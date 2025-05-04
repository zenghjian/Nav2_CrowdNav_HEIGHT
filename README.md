# Nav2py_CrowdNav_HEIGHT_Controller

本项目实现了基于HEIGHT模型的ROS2导航控制器插件。HEIGHT (Heterogeneous Interaction Graph Transformer) 是一种高级的人群导航算法，能够在拥挤和受限环境中进行有效导航。

## 📌 安装

### **1. 准备HEIGHT模型**

将模型文件放在以下位置：

```bash
models/height_policy.pt
```

你可以使用CrowdNav_HEIGHT训练的模型。推荐使用的模型：

- 随机环境: `CrowdNav_HEIGHT/trained_models/ours_HH_RH_randEnv/237400.pt`
- 休息室环境: `CrowdNav_HEIGHT/trained_models/ours_RH_HH_loungeEnv_resumeFromRand/137400.pt`
- 走廊环境: `CrowdNav_HEIGHT/trained_models/ours_RH_HH_hallwayEnv/208200.pt`

### **2. 安装依赖**

使用以下命令安装依赖：

```bash
pip install -e .
```

## 🔧 使用方法

在ROS2的Nav2参数文件中，将控制器设置为`nav2py_crowdnav_height_controller::HeightCrowdNavController`：

```yaml
controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    failure_tolerance: 0.3
    progress_checker_plugin: "progress_checker"
    goal_checker_plugins: ["general_goal_checker"]
    controller_plugins: ["HeightCrowdNavController"]

    # Progress checker
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    # Goal checker
    general_goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True

    # HeightCrowdNavController
    HeightCrowdNavController:
      plugin: "nav2py_crowdnav_height_controller::HeightCrowdNavController"
      transform_tolerance: 0.1
```

## 🛠️ 开发历程

本项目从PaS_CrowdNav控制器迁移到HEIGHT控制器，完成了以下工作：

1. 更新了包名和插件名从nav2py_pas_crowdnav_controller到nav2py_crowdnav_height_controller
2. 创建了新的控制器实现，使用HEIGHT模型替代PaS模型
3. 修改了模型导入方式，能够动态查找和加载CrowdNav_HEIGHT模块
4. 优化了模型加载和推理代码，适配HEIGHT的输入输出格式
5. 添加了错误处理和回退机制，提高了控制器的稳定性

## 📝 注意事项

1. 确保CrowdNav_HEIGHT目录存在于正确位置，以便控制器能够找到并导入相关模块
2. 确保模型文件被正确放置在models目录或者CrowdNav_HEIGHT/trained_models目录
3. 使用前请确认已安装所有必要的依赖包

