import torch
import numpy as np
import os
from collections import deque
import math
from .PaS_CrowdNav.crowd_nav.configs.config import Config
from .PaS_CrowdNav.rl.model import Policy
from .PaS_CrowdNav.rl.pas_rnn_model import Label_VAE
from rclpy.logging import get_logger

def set_log_level(logger, level='info'):
    """
    Set the logging level for a ROS2 logger
    
    Args:
        logger: The ROS2 logger instance
        level: The log level ('debug', 'info', 'warn', 'error', 'fatal')
    """
    from rclpy.logging import LoggingSeverity
    
    levels = {
        'debug': LoggingSeverity.DEBUG,
        'info': LoggingSeverity.INFO,
        'warn': LoggingSeverity.WARN,
        'error': LoggingSeverity.ERROR,
        'fatal': LoggingSeverity.FATAL
    }
    
    if level.lower() in levels:
        logger.set_level(levels[level.lower()])
        logger.info(f"Log level set to {level.upper()}")
    else:
        logger.warn(f"Unknown log level '{level}'. Using INFO level.")
        logger.set_level(LoggingSeverity.INFO)

class PaSController:
    """
    People as Sensors (PaS) controller, used to predict occluded areas and provide navigation decisions
    """
    def __init__(self, vae_path=None, policy_path=None, device="cpu", config=None):
        """
        Initialize PaS controller
        
        Args:
            vae_path: Path to VAE model
            policy_path: Path to Policy (PAS-RNN) model
            device: Running device ('cpu' or 'cuda')
            config: Configuration object, if None, use default configuration
        """
        self.logger = get_logger('pas_controller')
        
        set_log_level(self.logger, 'info')
        
        self.config = config if config is not None else Config()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sequence_length = self.config.pas.sequence
        self.costmap_sequence = deque(maxlen=self.sequence_length)
        self.grid_resolution = self.config.pas.grid_res
        
        # Expected input dimensions for the model
        self.expected_height = 96
        self.expected_width = 96
        
        # Model and hidden state initialization
        self.vae_model = None
        self.policy_model = None
        self.model_loaded = False
        self.rnn_hidden_state = None  # Store RNN hidden state
        
        # Load models
        if vae_path and os.path.exists(vae_path) and policy_path and os.path.exists(policy_path):
            try:
                self.load_model(vae_path, policy_path)
                self.model_loaded = True
                self.logger.info(f"Models loaded successfully from {vae_path} and {policy_path}")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                self.model_loaded = False
        else:
            # If no model or loading failed, use simple obstacle avoidance
            self.model_loaded = False
            self.logger.warn("No model specified or model not found. Using fallback strategy.")
    
    def load_model(self, vae_path, policy_path):
        """
        Load pre-trained models
        
        Args:
            vae_path: Path to VAE model file
            policy_path: Path to Policy model file
        """
        self.logger.info(f"Loading model from {vae_path} and {policy_path}")
        vae_state_dict = torch.load(vae_path, map_location=self.device)
        policy_state_dict = torch.load(policy_path, map_location=self.device)
        
        class Args:
            def __init__(self):
                self.rnn_output_size = 128  
                self.rnn_input_size = 64
                self.rnn_hidden_size = 128
                self.num_steps = 1
                self.num_processes = 1
                self.num_mini_batch = 1

        args = Args()
        
        self.logger.info("Creating Label_VAE instance")
        self.vae_model = Label_VAE(args)
        
        try:
            self.vae_model.load_state_dict(vae_state_dict)
            self.logger.info("VAE state dict loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading VAE state dict: {e}")
            raise
        
        self.vae_model.to(self.device)
        self.vae_model.eval()
        

        # Create a dummy action_space object
        class ActionSpace:
            def __init__(self, config):
                self.shape = [2]  # Assume action space is 2-dimensional [linear velocity, angular velocity]
                self.__class__.__name__ = "Box"
                self.kinematics = config.action_space.kinematics
        
        # Use the actual config
        self.logger.info("Creating Policy model instance")
        self.policy_model = Policy(
            action_space=ActionSpace(self.config),
            config=self.config,
            base_kwargs=args,
            base=self.config.robot.policy
        )
        
        # Load model weights
        self.policy_model.load_state_dict(policy_state_dict)
        self.policy_model.to(self.device)
        self.policy_model.eval()
        
        # Initialize RNN hidden state
        self.rnn_hidden_state = {
            'policy': torch.zeros(1, 1, 1, args.rnn_hidden_size).to(self.device)
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
                
            v_x = 0.0
            v_y = 0.0
            v_linear = 0.0
            v_angular = 0.0
            
            if current_velocity is not None:
                v_linear = current_velocity.get('linear', 0.0)
                v_angular = current_velocity.get('angular', 0.0)
                
                v_x = v_linear * math.cos(yaw)
                v_y = v_linear * math.sin(yaw)
            
            self.logger.info(f"Robot velocities: linear={v_linear:.2f}, angular={v_angular:.2f}")
            
            if self.config.action_space.kinematics == 'holonomic':
                robot_state = torch.tensor([[[rel_x, rel_y, v_x, v_y]]], dtype=torch.float32).to(self.device)
                self.logger.info(f"Created holonomic robot state: [{rel_x:.2f}, {rel_y:.2f}, {v_x:.2f}, {v_y:.2f}]")
            else:
                #  [rel_x, rel_y, theta, v, w]
                robot_state = torch.tensor([[[rel_x, rel_y, yaw, v_linear, v_angular]]], dtype=torch.float32).to(self.device)
                self.logger.info(f"Created differential robot state: [{rel_x:.2f}, {rel_y:.2f}, {yaw:.2f}, {v_linear:.2f}, {v_angular:.2f}]")
            
            return robot_state
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error in prepare_robot_state: {e}")
            self.logger.error(traceback.format_exc())
            
            if self.config.action_space.kinematics == 'holonomic':
                return torch.tensor([[[0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32).to(self.device)
            else:
                return torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32).to(self.device)
    
    def update_sequence(self, costmap):
        """
        Update costmap sequence
        
        Args:
            costmap: New occupancy grid map
        """
        # Resize costmap to match the expected input dimensions for the model
        resized_costmap = self.resize_costmap(costmap, self.expected_height, self.expected_width)
        
        if len(self.costmap_sequence) < self.sequence_length:
            # If sequence is not complete, fill with current costmap
            for _ in range(self.sequence_length - len(self.costmap_sequence)):
                self.costmap_sequence.append(resized_costmap.copy())
            self.logger.debug(f"Initialized sequence with {self.sequence_length} costmaps")
        else:
            # Add new costmap
            self.costmap_sequence.append(resized_costmap.copy())
            self.logger.debug("Added new costmap to sequence")
    
    def resize_costmap(self, costmap, target_height, target_width):
        """
        Resize costmap to target dimensions using simple interpolation
        
        Args:
            costmap: Input costmap array
            target_height: Target height
            target_width: Target width
            
        Returns:
            numpy array: Resized costmap
        """
        # Use simple resizing to match expected dimensions
        h, w = costmap.shape
        self.logger.debug(f"Resizing costmap from {h}x{w} to {target_height}x{target_width}")
        
        # Create a new empty costmap with target dimensions
        resized = np.zeros((target_height, target_width), dtype=np.float32)
        
        # Calculate scaling factors
        h_scale = h / target_height
        w_scale = w / target_width
        
        # Simple nearest neighbor interpolation
        for i in range(target_height):
            for j in range(target_width):
                orig_i = min(int(i * h_scale), h - 1)
                orig_j = min(int(j * w_scale), w - 1)
                resized[i, j] = costmap[orig_i, orig_j]
        
        return resized
    
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
        
        # Print costmap
        # self.log_costmap(costmap, self.logger, "Original Costmap")
        
        # Update costmap sequence
        self.update_sequence(costmap)
        
        # Log goal and pose information
        if goal is not None:
            self.logger.info(f"Goal pose (relative to robot): x={goal.get('x', 0.0):.2f}, y={goal.get('y', 0.0):.2f}")
        
        # Prepare robot state based on pose and goal
        robot_state = self.prepare_robot_state(pose, velocity, goal)
        
        if self.model_loaded:
            # If model is loaded, use model for inference
            try:
                self.logger.info("Using PaS model for prediction")
                return self.predict_control(robot_state)
            except Exception as e:
                import traceback
                self.logger.error(f"Error during model prediction: {e}")
                self.logger.error(traceback.format_exc())
                return None, None
        else:
            # If model not loaded, return None to use the fallback
            self.logger.warn("Model not loaded, returning None to use fallback")
            return None, None
    
    def predict_control(self, robot_state):
        """
        Use PaS model to predict control commands
        
        Args:
            robot_state: Tensor containing robot state information
            
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
                
                # Print robot state for debugging
                robot_state_np = robot_state.cpu().numpy()
                self.logger.info(f"Robot state shape: {robot_state.shape}")
                self.logger.info(f"Robot state values: {robot_state_np}")
                
                # Prepare input
                # Stack costmap sequence as a tensor
                costmap_array = np.stack(list(self.costmap_sequence), axis=0)  # [sequence_length, height, width]
                
                # Ensure costmap format is correct 
                if costmap_array.ndim == 3:
                    # Add channel dimension [sequence_length, 1, height, width]
                    costmap_tensor = torch.tensor(costmap_array, dtype=torch.float32).unsqueeze(1).to(self.device)
                else:
                    # Assume already in [sequence_length, channels, height, width] format
                    costmap_tensor = torch.tensor(costmap_array, dtype=torch.float32).to(self.device)
                
                # Debug information
                self.logger.info(f"Costmap tensor shape: {costmap_tensor.shape}")
                
                # If using sequence input, adjust shape
                if self.config.pas.seq_flag:
                    # [batch_size, sequence_length, height, width]
                    costmap_tensor_seq = costmap_tensor.permute(1, 0, 2, 3)
                    self.logger.debug("Using sequence input mode")
                else:
                    # Use latest costmap
                    costmap_tensor_seq = costmap_tensor[-1:].unsqueeze(0)
                    self.logger.debug("Using single frame input mode")
                
                # Debug information 
                self.logger.info(f"Costmap tensor sequence shape: {costmap_tensor_seq.shape}")
                self.logger.info(f"Robot state shape: {robot_state.shape}")
                
                # Prepare input dictionary
                inputs = {
                    'grid': costmap_tensor_seq,
                    'vector': robot_state
                }
                
                # Prepare mask
                masks = torch.ones(1, 1).to(self.device)
                
                # Use policy model for inference
                self.logger.info("Running policy model inference")
                value, action, _, self.rnn_hidden_state, decoded = self.policy_model.act(
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
                    if self.config.action_space.kinematics == 'holonomic':
                        self.prev_v = np.array([0.0, 0.0])
                    else:
                        self.prev_v = 0.0
                
                # Set parameters
                v_pref = self.config.robot.v_pref
                time_step = self.config.env.time_step
                a_pref = 1.0  # Default acceleration preference
                holonomic = self.config.action_space.kinematics == 'holonomic'
                
                self.logger.debug(f"Config parameters: v_pref={v_pref}, time_step={time_step}, a_pref={a_pref}, holonomic={holonomic}")
                
                # Implement clip_action logic
                if holonomic:
                    raw_action = np.array(raw_action)
                    
                    # Clip acceleration
                    a_norm = np.linalg.norm(raw_action - self.prev_v)
                    if a_norm > a_pref:
                        v_action = np.zeros(2)
                        raw_ax = raw_action[0] - self.prev_v[0]
                        raw_ay = raw_action[1] - self.prev_v[1]
                        v_action[0] = (raw_ax / a_norm * a_pref) * time_step + self.prev_v[0]
                        v_action[1] = (raw_ay / a_norm * a_pref) * time_step + self.prev_v[1]
                    else:
                        v_action = raw_action
                    
                    # Clip velocity
                    v_norm = np.linalg.norm(v_action)
                    if v_norm > v_pref:
                        v_action[0] = v_action[0] / v_norm * v_pref
                        v_action[1] = v_action[1] / v_norm * v_pref
                    
                    # Save current velocity as previous velocity for next step
                    self.prev_v = v_action.copy()
                    
                    # Convert from ActionXY to linear_x and angular_z
                    linear_x = np.linalg.norm(v_action)
                    angular_z = np.arctan2(v_action[1], v_action[0]) if linear_x > 0.01 else 0.0
                
                else:  # non-holonomic
                    linear_x = np.clip(raw_action[0], -0.1, 0.1)
                    angular_z = np.clip(raw_action[1], -0.1, 0.1)
                    
                    
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
            
    def log_costmap(self, costmap, logger, title="Costmap", threshold=0.5, max_size=20):
        """
        Print costmap in ASCII format to the logger
        
        Args:
            costmap: The costmap array to print
            logger: Logger object
            title: Title for the visualization
            threshold: Binarization threshold
            max_size: Maximum size to print (costmap will be downsampled if larger)
        """
        height, width = costmap.shape
        
        # Downsample if costmap is too large
        if height > max_size or width > max_size:
            h_step = max(1, height // max_size)
            w_step = max(1, width // max_size)
            sampled_height = height // h_step
            sampled_width = width // w_step
            
            # Create downsampled costmap
            sampled_costmap = np.zeros((sampled_height, sampled_width))
            for i in range(sampled_height):
                for j in range(sampled_width):
                    # Use average as the downsampled value
                    sampled_costmap[i, j] = np.mean(
                        costmap[i*h_step:min((i+1)*h_step, height), 
                            j*w_step:min((j+1)*w_step, width)]
                    )
            
            costmap = sampled_costmap
            height, width = costmap.shape
        
        # Print title and dimension info
        logger.info(f"\n{title} ({height}x{width}):")
        
        # Create visualization using characters for different values
        map_str = ""
        
        # Find robot position (usually the center of costmap)
        robot_y, robot_x = height // 2, width // 2
        
        # Create character representation for costmap values
        for i in range(height):
            row_str = ""
            for j in range(width):
                # If robot position
                if i == robot_y and j == robot_x:
                    row_str += "R "  # 'R' represents robot
                # If obstacle (based on threshold)
                elif costmap[i, j] > threshold:
                    val = int(min(9, costmap[i, j] * 9))  # Scale value to 0-9
                    row_str += f"{val} "  # Number represents obstacle intensity
                # If unknown area (medium value)
                elif costmap[i, j] > 0.2:
                    row_str += "· "  # Dot represents medium probability
                # If free space
                else:
                    row_str += "  "  # Space represents free space
            map_str += row_str + "\n"
        
        # Print the map
        logger.info(f"\n{map_str}")
        
        # Print legend
        logger.info("Legend: R=robot position, numbers(1-9)=obstacle intensity, ·=medium probability, space=free space")