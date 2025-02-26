import yaml
import nav2py
import nav2py.interfaces
import base64
import numpy as np
import cv2
import rclpy
from rclpy.logging import get_logger

from .pas_controller import PaSController
from .PaS_CrowdNav.crowd_nav.configs.config import Config
import os
import torch

class nav2py_template_controller(nav2py.interfaces.nav2py_costmap_controller):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_callback('path', self._path_callback)
        # [新增] 注册新的回调
        self._register_callback('scan_data', self._scan_callback)

        config = Config()
        
        vae_path = "/home/zeng/nav_ws/src/nav2py_template_controller/nav2py_template_controller/nav2py_template_controller/PaS_CrowdNav/data/LabelVAE_CircleFOV30/label_vae_ckpt/label_vae_weight_300.pth"
        policy_path = "/home/zeng/nav_ws/src/nav2py_template_controller/nav2py_template_controller/nav2py_template_controller/PaS_CrowdNav/data/pas_rnn/checkpoints/33200.pt"
        
        if not os.path.exists(vae_path) or not os.path.exists(policy_path):
            print(f"Warning: Model files not found at {vae_path} or {policy_path}")
            print("Using simple obstacle avoidance instead.")
            vae_path = None
            policy_path = None
        
        self.pas_controller = PaSController(
            vae_path=vae_path,
            policy_path=policy_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            config=config
        )        
        
        self.logger = get_logger('nav2py_template_controller')
        
    def _path_callback(
        self,
        path_,
    ):
        try:
            if isinstance(path_, list) and len(path_) > 0:
                data_str = path_[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                
                path = yaml.safe_load(data_str)
            else:
                if isinstance(path_, bytes):
                    path = yaml.safe_load(path_.decode())
                else:
                    self.logger.error(f"Unexpected path data type: {type(path_)}")
                    return
                    
            # linear_x = 1
            # angular_v = 0
            # self._send_cmd_vel(linear_x, angular_v)
            # self.logger.info(f"Path callback processed - Command: linear={linear_x}, angular={angular_v}")
                
        except Exception as e:
            import traceback
            self.logger.error(f"Error processing path data: {e}")
            self.logger.error(traceback.format_exc())

    def _scan_callback(self, scan_data: bytes):
        try:
            if isinstance(scan_data, list) and len(scan_data) > 0:
                data_str = scan_data[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                ogm = scan_to_ogm(yaml.safe_load(data_str), save_path=None)
                
                if ogm is not None:
                    linear_x, angular_v = self.pas_controller.process_ogm(ogm)

                    self._send_cmd_vel(linear_x, angular_v)
                    self.logger.info(f"Sended cmd_vel: linear_x={linear_x}, angular_v={angular_v}")
            else:
                self.logger.error("Invalid scan data received")
        except Exception as e:
            self.logger.error(f"Error processing scan data: {e}")
        
if __name__ == "__main__":
    nav2py.main(nav2py_template_controller)