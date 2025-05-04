import sys
import torch
import numpy as np
import os
from collections import deque
import math
import logging
import importlib.util
import pathlib


def set_log_level(logger: logging.Logger, level='info'):
    """
    Set the logging level for a ROS2 logger

    Args:
        logger: The ROS2 logger instance
        level: The log level ('debug', 'info', 'warn', 'error', 'fatal')
    """

    levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warn': logging.WARN,
        'error': logging.ERROR,
        'fatal': logging.FATAL
    }

    if level.lower() in levels:
        logger.setLevel(levels[level.lower()])
        logger.info(f"Log level set to {level.upper()}")
    else:
        logger.warn(f"Unknown log level '{level}'. Using INFO level.")
        logger.setLevel(logging.INFO)


class HeightController:
    """
    HEIGHT (Heterogeneous Interaction Graph Transformer) controller for navigation in crowded environments
    """

    def __init__(self, policy_path=None, device="cpu", config=None):
        """
        Initialize HEIGHT controller

        Args:
            policy_path: Path to Policy model
            device: Running device ('cpu' or 'cuda')
            config: Configuration object, if None, use default configuration
        """
        self.logger = logging.getLogger('height_controller')
        self.logger.addHandler(logging.StreamHandler(sys.stderr))

        set_log_level(self.logger, 'info')

        # Dynamically import CrowdNav_HEIGHT modules
        self.config = None
        self.Policy = None
        
        try:
            # Find CrowdNav_HEIGHT directory
            current_dir = pathlib.Path(__file__).parent.absolute()
            crowdnav_dir = current_dir.parent / "CrowdNav_HEIGHT"
            
            if not crowdnav_dir.exists():
                # Search in parent directories
                crowdnav_dir = current_dir.parent.parent / "CrowdNav_HEIGHT"
            
            if not crowdnav_dir.exists():
                self.logger.error(f"CrowdNav_HEIGHT directory not found at {crowdnav_dir}")
                raise ImportError("CrowdNav_HEIGHT module not found")
            
            self.logger.info(f"Found CrowdNav_HEIGHT at {crowdnav_dir}")
            
            # Add CrowdNav_HEIGHT to path
            sys.path.append(str(crowdnav_dir))
            
            # Now import the modules
            from crowd_nav.configs.config import Config
            from training.networks.model import Policy
            
            self.Config = Config
            self.Policy = Policy
            
            if config is None:
                self.config = Config()
            else:
                self.config = config
                
            self.logger.info("Successfully imported CrowdNav_HEIGHT modules")
            
        except Exception as e:
            self.logger.error(f"Error importing CrowdNav_HEIGHT modules: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            if config is None:
                # Create a minimal config object
                class DummyConfig:
                    def __init__(self):
                        self.sim = type('obj', (object,), {'human_num': 5})
                        self.lidar = type('obj', (object,), {'angular_res': 2.0})
                        self.SRNN = type('obj', (object,), {'human_node_rnn_size': 128})
                        self.robot = type('obj', (object,), {
                            'v_pref': 0.5, 
                            'v_max': 0.5, 
                            'w_max': 1.0,
                            'policy': 'selfAttn_merge_srnn_lidar'
                        })
                
                self.config = DummyConfig()
            else:
                self.config = config

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lidar_sequence_length = 4  # Number of lidar scans to keep
        self.lidar_sequence = deque(maxlen=self.lidar_sequence_length)

        # Expected input dimensions for the model
        self.expected_height = 96
        self.expected_width = 96

        # Model and hidden state initialization
        self.policy_model = None
        self.model_loaded = False
        self.rnn_hidden_state = None  # Store RNN hidden state

        # Load models
        if policy_path and os.path.exists(policy_path):
            try:
                self.load_model(policy_path)
                self.model_loaded = True
                self.logger.info(f"Model loaded successfully from {policy_path}")
            except Exception as e:
                self.logger.warn(f"Failed to load model: {e}")
                self.model_loaded = False
        else:
            # If no model or loading failed, use simple obstacle avoidance
            self.model_loaded = False
            self.logger.warn("No model specified or model not found. Using fallback strategy.")

    def load_model(self, policy_path):
        """
        Load pre-trained model

        Args:
            policy_path: Path to Policy model file
        """
        self.logger.info(f"Loading model from {policy_path}")
        policy_state_dict = torch.load(policy_path, map_location=self.device)

        # Check if Policy class was imported successfully
        if self.Policy is None:
            self.logger.error("Policy class not imported, cannot load model")
            return
            
        # Create observation space dictionary for HEIGHT model
        obs_shape = {
            'robot_node': 5,  # [rel_x, rel_y, theta, v, w] for differential drive robot
            'spatial_edges': self.config.sim.human_num * 4, # x, y, vx, vy for each human
            'detected_human_num': 1,
            'point_clouds': 1  # lidar data
        }

        # Use the actual config
        self.logger.info("Creating HEIGHT Policy model instance")
        self.policy_model = self.Policy(
            obs_shape=obs_shape,
            action_space=2,  # [linear_velocity, angular_velocity]
            base=self.config.robot.policy
        )

        # Load model weights
        self.policy_model.load_state_dict(policy_state_dict)
        self.policy_model.to(self.device)
        self.policy_model.eval()

        # Initialize RNN hidden state
        self.rnn_hidden_state = {
            'rnn': torch.zeros(1, 1, 1, self.config.SRNN.human_node_rnn_size).to(self.device)
        }

        self.logger.info("Model loading completed successfully")

    def prepare_robot_state(self, pose, current_velocity=None, goal_pose=None):
        """
        Prepare robot state vector for the network

        Args:
            pose: Robot pose dictionary
            current_velocity: Current robot velocity (optional)
            goal_pose: Goal pose (relative position from path's last point)

        Returns:
            tensor: Robot state vector formatted for the neural network
        """
        try:
            qx = pose['robot_pose']['orientation']['x']
            qy = pose['robot_pose']['orientation']['y']
            qz = pose['robot_pose']['orientation']['z']
            qw = pose['robot_pose']['orientation']['w']

            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

            rel_x = 0.0
            rel_y = 0.0
            if goal_pose is not None:
                rel_x = goal_pose.get('x', 0.0)
                rel_y = goal_pose.get('y', 0.0)
                self.logger.info(f"Using path endpoint as goal: x={rel_x:.2f}, y={rel_y:.2f}")
            else:
                self.logger.warn("No goal position available, using defaults (0,0)")

            v_linear = 0.0
            v_angular = 0.0

            if current_velocity is not None:
                v_linear = current_velocity.get('linear', 0.0)
                v_angular = current_velocity.get('angular', 0.0)

            self.logger.info(f"Robot velocities: linear={v_linear:.2f}, angular={v_angular:.2f}")

            #  [rel_x, rel_y, theta, v, w]
            robot_state = torch.tensor([[[rel_x, rel_y, yaw, v_linear, v_angular]]], dtype=torch.float32).to(self.device)
            self.logger.info(f"Created robot state: [{rel_x:.2f}, {rel_y:.2f}, {yaw:.2f}, {v_linear:.2f}, {v_angular:.2f}]")

            return robot_state

        except Exception as e:
            import traceback
            self.logger.error(f"Error in prepare_robot_state: {e}")
            self.logger.error(traceback.format_exc())
            return torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32).to(self.device)

    def prepare_lidar_data(self, costmap):
        """
        Convert costmap to simulated lidar scan data for the model

        Args:
            costmap: Occupancy grid (normalized [0, 1])

        Returns:
            tensor: Lidar scan data in appropriate format for the model
        """
        try:
            # Get costmap dimensions
            h, w = costmap.shape
            center_y, center_x = h // 2, w // 2
            
            # Number of lidar rays (based on angular resolution)
            lidar_angular_res = self.config.lidar.angular_res
            num_angles = int(360.0 / lidar_angular_res)
            max_range = min(h, w) // 2  # Maximum range in grid cells
            
            # Initialize lidar ranges
            lidar_data = np.ones(num_angles, dtype=np.float32) * max_range
            
            # For each angle, raycast to find obstacles
            for i in range(num_angles):
                angle_rad = math.radians(i * lidar_angular_res)
                # Cast ray from center until obstacle or max range
                for r in range(1, max_range):
                    x = int(center_x + r * math.cos(angle_rad))
                    y = int(center_y + r * math.sin(angle_rad))
                    
                    # Check bounds
                    if x < 0 or x >= w or y < 0 or y >= h:
                        lidar_data[i] = r
                        break
                    
                    # Check if cell is occupied (threshold at 0.5)
                    if costmap[y, x] > 0.5:
                        lidar_data[i] = r
                        break
            
            # Normalize to [0, 1]
            lidar_data = lidar_data / max_range
            
            # Add to sequence
            if len(self.lidar_sequence) < self.lidar_sequence_length:
                # If sequence is not complete, fill with current scan
                for _ in range(self.lidar_sequence_length - len(self.lidar_sequence)):
                    self.lidar_sequence.append(lidar_data.copy())
                self.logger.debug(f"Initialized sequence with {self.lidar_sequence_length} lidar scans")
            else:
                # Add new scan
                self.lidar_sequence.append(lidar_data.copy())
                self.logger.debug("Added new lidar scan to sequence")
            
            # Convert to tensor format expected by HEIGHT model
            # [batch, channels, sequence_length, num_angles]
            lidar_tensor = torch.tensor(np.stack(list(self.lidar_sequence), axis=0), 
                                       dtype=torch.float32).unsqueeze(0).to(self.device)
            
            self.logger.info(f"Prepared lidar data tensor with shape: {lidar_tensor.shape}")
            return lidar_tensor
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error in prepare_lidar_data: {e}")
            self.logger.error(traceback.format_exc())
            # Return empty lidar data with correct shape
            return torch.zeros((1, 1, self.lidar_sequence_length, int(360.0 / self.config.lidar.angular_res)), 
                             dtype=torch.float32).to(self.device)

    def prepare_human_data(self, costmap, resolution):
        """
        Extract simulated human data from the costmap
        This is a placeholder - in real implementation we would use actual human detections

        Args:
            costmap: Occupancy grid map
            resolution: Costmap resolution (meters/cell)

        Returns:
            tuple: (spatial_edges, detected_human_num) for the model
        """
        # For now, we're simulating "no humans detected"
        max_humans = self.config.sim.human_num
        
        # Create empty spatial edges tensor
        # [batch, sequence, human_num, 4] - (x, y, vx, vy) for each human
        spatial_edges = torch.zeros((1, 1, max_humans, 4), dtype=torch.float32).to(self.device)
        
        # Set detected human count to 0
        detected_human_num = torch.tensor([[0]], dtype=torch.int32).to(self.device)
        
        self.logger.info(f"Prepared empty human data with {max_humans} slots")
        return spatial_edges, detected_human_num

    def process_costmap(self, costmap, pose, resolution, origin_x, origin_y, velocity=None, goal=None):
        """
        Process a costmap and robot pose, return control commands

        Args:
            costmap: Occupancy grid map (normalized to [0, 1])
            pose: Robot pose dictionary
            resolution: Costmap resolution (meters/cell)
            origin_x: Costmap origin X (meters)
            origin_y: Costmap origin Y (meters)
            velocity: Current robot velocity (optional)
            goal: Goal position relative to robot frame (optional)

        Returns:
            tuple: (linear_x, angular_z) linear velocity and angular velocity
        """
        # Check if costmap has valid dimensions
        if costmap.size == 0:
            self.logger.error("Error: Empty costmap received")
            return None, None

        self.logger.info(f"Processing costmap of shape {costmap.shape}, resolution: {resolution}")

        # Log goal and pose information
        if goal is not None:
            self.logger.info(f"Goal pose (relative to robot): x={goal.get('x', 0.0):.2f}, y={goal.get('y', 0.0):.2f}")

        # Prepare inputs for the model
        robot_state = self.prepare_robot_state(pose, velocity, goal)
        lidar_data = self.prepare_lidar_data(costmap)
        spatial_edges, detected_human_num = self.prepare_human_data(costmap, resolution)

        if self.model_loaded:
            # If model is loaded, use model for inference
            try:
                self.logger.info("Using HEIGHT model for prediction")
                return self.predict_control(robot_state, lidar_data, spatial_edges, detected_human_num)
            except Exception as e:
                import traceback
                self.logger.error(f"Error during model prediction: {e}")
                self.logger.error(traceback.format_exc())
                return None, None
        else:
            # If model not loaded, return None to use the fallback
            self.logger.warn("Model not loaded, returning None to use fallback")
            return None, None

    def predict_control(self, robot_state, lidar_data, spatial_edges, detected_human_num):
        """
        Use HEIGHT model to predict control commands

        Args:
            robot_state: Tensor containing robot state information
            lidar_data: Tensor containing lidar scan data
            spatial_edges: Tensor containing human position and velocity data
            detected_human_num: Tensor containing number of detected humans

        Returns:
            tuple: (linear_x, angular_z) linear velocity and angular velocity
        """
        with torch.no_grad():
            try:
                # Add prominent frame delimiter for prediction logs
                frame_delimiter = "*" * 80
                self.logger.info(f"\n{frame_delimiter}")
                self.logger.info(f"***** STARTING MODEL PREDICTION *****")
                self.logger.info(f"{frame_delimiter}")

                # Prepare input dictionary
                inputs = {
                    'robot_node': robot_state,
                    'spatial_edges': spatial_edges,
                    'detected_human_num': detected_human_num,
                    'point_clouds': lidar_data
                }

                # Prepare mask
                masks = torch.ones(1, 1).to(self.device)

                # Use policy model for inference
                self.logger.info("Running policy model inference")
                value, action, _, self.rnn_hidden_state = self.policy_model.act(
                    inputs,
                    self.rnn_hidden_state,
                    masks,
                    deterministic=True  # Usually use deterministic behavior in deployment
                )

                # Extract raw action
                raw_action = action.cpu().numpy()
                self.logger.info(f"Action shape: {raw_action.shape}, Action values: {raw_action}")

                # Ensure raw_action is a 1D array
                if len(raw_action.shape) == 2:
                    raw_action = raw_action[0]

                # Initialize previous velocity (if not exist)
                if not hasattr(self, 'prev_v'):
                    self.prev_v = 0.0

                # Set parameters
                v_pref = self.config.robot.v_pref
                v_max = self.config.robot.v_max
                w_max = self.config.robot.w_max

                # Non-holonomic robot control
                linear_x = np.clip(raw_action[0], -v_max, v_max)
                angular_z = np.clip(raw_action[1], -w_max, w_max)

                self.logger.info(f"Computed control: linear_x={linear_x:.2f}, angular_z={angular_z:.2f}")

                # At the end, add another delimiter
                self.logger.info(f"\n{frame_delimiter}")
                self.logger.info(f"***** MODEL PREDICTION COMPLETED *****")
                self.logger.info(f"{frame_delimiter}")

                return linear_x, angular_z

            except Exception as e:
                import traceback
                self.logger.error(f"Error in predict_control: {e}")
                self.logger.error(traceback.format_exc())
                return None, None 