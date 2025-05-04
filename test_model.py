#!/usr/bin/env python3

import os
import sys
import logging
import numpy as np
import torch
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_height_model")

def main():
    # 获取当前目录
    current_dir = Path(__file__).parent.absolute()
    
    # 添加CrowdNav_HEIGHT到PATH
    crowdnav_dir = current_dir / "nav2py_crowdnav_height_controller" / "CrowdNav_HEIGHT"
    if not crowdnav_dir.exists():
        # 尝试其他可能的位置
        crowdnav_dir = current_dir / "nav2py_crowdnav_height_controller" / "nav2py_crowdnav_height_controller" / "CrowdNav_HEIGHT"
    
    if not crowdnav_dir.exists():
        logger.error(f"找不到CrowdNav_HEIGHT目录: {crowdnav_dir}")
        sys.exit(1)
    
    logger.info(f"找到CrowdNav_HEIGHT目录: {crowdnav_dir}")
    sys.path.append(str(crowdnav_dir))
    
    try:
        # 导入模块
        from crowd_nav.configs.config import Config
        from training.networks.model import Policy
        
        logger.info("成功导入CrowdNav_HEIGHT模块")
        
        # 查找模型文件
        model_paths = []
        # 检查models目录
        models_dir = current_dir / "models"
        if models_dir.exists():
            for f in models_dir.glob("*.pt"):
                model_paths.append(f)
        
        # 检查CrowdNav_HEIGHT/trained_models目录
        trained_models_dir = crowdnav_dir / "trained_models"
        if trained_models_dir.exists():
            # 递归查找所有.pt文件
            for root, dirs, files in os.walk(trained_models_dir):
                for file in files:
                    if file.endswith(".pt"):
                        model_paths.append(Path(root) / file)
        
        # 没有找到模型文件
        if not model_paths:
            logger.error("找不到任何模型文件(.pt)")
            sys.exit(1)
        
        # 使用找到的第一个模型文件
        model_path = model_paths[0]
        logger.info(f"使用模型文件: {model_path}")
        
        # 初始化配置
        config = Config()
        
        # 创建观测空间
        obs_shape = {
            'robot_node': 5,  # [rel_x, rel_y, theta, v, w]
            'spatial_edges': config.sim.human_num * 4,  # x, y, vx, vy for each human
            'detected_human_num': 1,
            'point_clouds': 1  # lidar data
        }
        
        # 初始化模型
        logger.info("正在初始化Policy模型...")
        policy_model = Policy(
            obs_shape=obs_shape,
            action_space=2,  # [linear_velocity, angular_velocity]
            base=config.robot.policy
        )
        
        # 加载模型权重
        logger.info(f"正在加载模型权重: {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
        
        try:
            state_dict = torch.load(model_path, map_location=device)
            policy_model.load_state_dict(state_dict)
            policy_model.to(device)
            policy_model.eval()
            logger.info("成功加载模型权重!")
            
            # 创建测试输入
            batch_size = 1
            rnn_hidden_state = {
                'rnn': torch.zeros(1, batch_size, 1, config.SRNN.human_node_rnn_size).to(device)
            }
            
            # Robot状态: [rel_x, rel_y, theta, v, w]
            robot_node = torch.tensor([[[1.0, 2.0, 0.0, 0.0, 0.0]]], dtype=torch.float32).to(device)
            
            # Human状态: [batch, seq, human_num, 4]
            spatial_edges = torch.zeros((batch_size, 1, config.sim.human_num, 4), dtype=torch.float32).to(device)
            
            # Human数量
            detected_human_num = torch.tensor([[0]], dtype=torch.int32).to(device)
            
            # Lidar数据
            lidar_angular_res = config.lidar.angular_res
            num_angles = int(360.0 / lidar_angular_res)
            lidar_sequence_length = 4
            lidar_data = torch.ones((batch_size, 1, lidar_sequence_length, num_angles), dtype=torch.float32).to(device)
            
            # 组装输入
            inputs = {
                'robot_node': robot_node,
                'spatial_edges': spatial_edges,
                'detected_human_num': detected_human_num,
                'point_clouds': lidar_data
            }
            
            # 准备掩码
            masks = torch.ones(batch_size, 1).to(device)
            
            # 执行推理
            logger.info("正在执行模型推理...")
            with torch.no_grad():
                value, action, _, rnn_hidden_state_out = policy_model.act(
                    inputs,
                    rnn_hidden_state,
                    masks,
                    deterministic=True
                )
            
            # 显示结果
            logger.info(f"模型输出: value={value.item():.4f}, action={action.cpu().numpy()}")
            logger.info("模型测试成功!")
            
        except Exception as e:
            logger.error(f"加载模型权重时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)
        
    except ImportError as e:
        logger.error(f"导入CrowdNav_HEIGHT模块失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 