import torch
import numpy as np
import os
from collections import deque
import yaml
import sys
import matplotlib.pyplot as plt
from skimage.draw import line as bresenham
import time
from .PaS_CrowdNav.crowd_nav.configs.config import Config
from .PaS_CrowdNav.rl.model import Policy
from .PaS_CrowdNav.rl.pas_rnn_model import Label_VAE

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
        self.config = config if config is not None else Config()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sequence_length = self.config.pas.sequence
        self.ogm_sequence = deque(maxlen=self.sequence_length)
        self.grid_resolution = self.config.pas.grid_res
        
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
                print(f"Models loaded successfully from {vae_path} and {policy_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.model_loaded = False
        else:
            # If no model or loading failed, use simple obstacle avoidance
            self.model_loaded = False
            print("No model specified or model not found. Using simple obstacle avoidance.")
    
    def load_model(self, vae_path, policy_path):
        """
        Load pre-trained models
        
        Args:
            vae_path: Path to VAE model file
            policy_path: Path to Policy model file
        """
        # First load model state dictionaries to check parameter sizes
        vae_state_dict = torch.load(vae_path, map_location=self.device)
        policy_state_dict = torch.load(policy_path, map_location=self.device)
        
        # Create args object that matches the model
        class Args:
            def __init__(self):
                self.rnn_output_size = 128  
                self.rnn_input_size = 64
                self.rnn_hidden_size = 128
                self.num_steps = 1
                self.num_processes = 1
                self.num_mini_batch = 1

        args = Args()
        # Create a Label_VAE instance that matches the model
        self.vae_model = Label_VAE(args)
        self.vae_model.load_state_dict(vae_state_dict)
        self.vae_model.to(self.device)
        self.vae_model.eval()
        

        # Create a dummy action_space object
        class ActionSpace:
            def __init__(self, config):
                self.shape = [2]  # Assume action space is 2-dimensional [linear velocity, angular velocity]
                self.__class__.__name__ = "Box"
                self.kinematics = config.action_space.kinematics
        
        # Use the actual config
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
        
        # Record recently processed OGM for visualization
        self.last_processed_ogm = None
        
    def update_sequence(self, ogm):
        """
        Update OGM sequence
        
        Args:
            ogm: New occupancy grid map
        """
        if len(self.ogm_sequence) < self.sequence_length:
            # If sequence is not complete, fill with current OGM
            for _ in range(self.sequence_length - len(self.ogm_sequence)):
                self.ogm_sequence.append(ogm.copy())
        else:
            # Add new OGM
            self.ogm_sequence.append(ogm.copy())
    
    def process_ogm(self, ogm):
        """
        Process a single OGM, return control commands
        
        Args:
            ogm: Occupancy grid map
            
        Returns:
            tuple: (linear_x, angular_z) linear velocity and angular velocity
        """
        # Update OGM sequence
        self.update_sequence(ogm)
        
        if self.model_loaded:
            # If model is loaded, use model for inference
            return self.predict_control()
        else:
            # Use simple obstacle avoidance strategy
            return self.simple_obstacle_avoidance(ogm)
    
    def prepare_robot_state(self):
        """
        Prepare robot state vector
        
        Returns:
            tensor: Robot state vector
        """
        if self.config.action_space.kinematics == 'holonomic':
            # For holonomic robot, state vector includes: [relative x, relative y, vx, vy]
            # In test environment, assume robot is at local coordinate origin with zero velocity
            robot_state = torch.zeros(1, 1, 4).to(self.device)
        else:
            # For differential drive robot, state vector includes: [relative x, relative y, theta, v, w]
            robot_state = torch.zeros(1, 1, 5).to(self.device)
        
        return robot_state
    
    def predict_control(self):
        """
        Use PaS model to predict control commands
        
        Returns:
            tuple: (linear_x, angular_z) linear velocity and angular velocity
        """
        with torch.no_grad():
            # Prepare input
            # Stack OGM sequence as a tensor
            ogm_array = np.stack(list(self.ogm_sequence), axis=0)  # [sequence_length, height, width]
            
            # Ensure OGM format is correct
            if ogm_array.ndim == 3:
                # Add channel dimension [sequence_length, 1, height, width]
                ogm_tensor = torch.tensor(ogm_array, dtype=torch.float32).unsqueeze(1).to(self.device)
            else:
                # Assume already in [sequence_length, channels, height, width] format
                ogm_tensor = torch.tensor(ogm_array, dtype=torch.float32).to(self.device)
                
            # If using sequence input, adjust shape
            if self.config.pas.seq_flag:
                # [batch_size, sequence_length, height, width]
                ogm_tensor_seq = ogm_tensor.permute(1, 0, 2, 3)
            else:
                # Use latest OGM
                ogm_tensor_seq = ogm_tensor[-1:].unsqueeze(0)
            
            # Prepare robot state
            robot_state = self.prepare_robot_state()
            
            # Prepare input dictionary
            inputs = {
                'grid': ogm_tensor_seq,
                'vector': robot_state
            }
            
            # Prepare mask
            masks = torch.ones(1, 1).to(self.device)
            
            # Use policy model for inference
            value, action, _, self.rnn_hidden_state, decoded = self.policy_model.act(
                inputs, 
                self.rnn_hidden_state, 
                masks, 
                deterministic=True  # Usually use deterministic behavior in deployment
            )
            
            # Save decoded OGM for visualization
            if decoded is not None:
                self.last_processed_ogm = decoded.squeeze().cpu().numpy()
            
            # Extract raw action
            raw_action = action.cpu().numpy()
            print(f"Action shape: {raw_action.shape}, Action values: {raw_action}")
            
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
                # Clip action (changes in v and w)
                clipped_v_change = np.clip(raw_action[0], -0.1, 0.1)
                clipped_w_change = np.clip(raw_action[1], -0.25, 0.25)
                
                # If need to track current v and w
                if not hasattr(self, 'current_v'):
                    self.current_v = 0.0
                if not hasattr(self, 'current_w'):
                    self.current_w = 0.0
                
                # Update v and w
                self.current_v += clipped_v_change
                self.current_w = clipped_w_change  # Directly set angular velocity instead of accumulating
                
                # Ensure velocity is within valid range
                self.current_v = np.clip(self.current_v, 0.0, v_pref)
                
                # Set output
                linear_x = self.current_v
                angular_z = self.current_w
            
        return linear_x, angular_z
    
    def simple_obstacle_avoidance(self, ogm):
        """
        Simple obstacle avoidance strategy
        
        Args:
            ogm: Occupancy grid map
            
        Returns:
            tuple: (linear_x, angular_z) linear velocity and angular velocity
        """
        # Get map center (robot position)
        center_y, center_x = ogm.shape[0] // 2, ogm.shape[1] // 2
        
        # Define front area
        front_width = 20  # Front area width
        front_length = 30  # Front area length
        
        # Ensure coordinates are in valid range
        front_y_start = max(0, center_y - front_length)
        front_x_start = max(0, center_x - front_width // 2)
        front_x_end = min(ogm.shape[1], center_x + front_width // 2)
        
        front_area = ogm[front_y_start:center_y, front_x_start:front_x_end]
        
        # Calculate left and right areas
        left_y_start = max(0, center_y - 20)
        left_x_start = min(ogm.shape[1], center_x)
        left_x_end = min(ogm.shape[1], center_x + 20)
        
        right_y_start = max(0, center_y - 20)
        right_x_start = max(0, center_x - 20)
        right_x_end = max(0, center_x)
        
        left_area = ogm[left_y_start:center_y, left_x_start:left_x_end]
        right_area = ogm[right_y_start:center_y, right_x_start:right_x_end]
        
        # Prevent division by zero
        front_size = max(1, front_area.size)
        left_size = max(1, left_area.size)
        right_size = max(1, right_area.size)
        
        # Calculate obstacle ratio in areas
        front_obstacle_ratio = np.sum(front_area > 0.7) / front_size
        left_obstacle_ratio = np.sum(left_area > 0.7) / left_size
        right_obstacle_ratio = np.sum(right_area > 0.7) / right_size
        
        # Decision logic
        if front_obstacle_ratio > 0.1:  # If obstacles in front
            linear_x = 0.2  # Slow down
            if left_obstacle_ratio < right_obstacle_ratio:
                angular_z = 0.5  # Turn left
            else:
                angular_z = -0.5  # Turn right
        else:
            linear_x = 0.5  # Normal speed
            angular_z = 0.0  # Go straight
        
        return linear_x, angular_z

def generate_test_ogm(width=100, height=100, obstacle_positions=None):
    """
    Generate test occupancy grid map
    
    Args:
        width: Map width
        height: Map height
        obstacle_positions: List of obstacle positions [(x1, y1), (x2, y2), ...]
        
    Returns:
        numpy array: Occupancy grid map
    """
    # Create an empty OGM (0: free, 0.5: unknown, 1: occupied)
    ogm = np.ones((height, width), dtype=np.float32) * 0.5
    
    # Robot position (map center)
    center_x, center_y = width // 2, height // 2
    
    # Create a free area around the robot
    radius = 10
    for y in range(height):
        for x in range(width):
            if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                ogm[y, x] = 0.0
    
    # Add obstacles
    if obstacle_positions is None:
        # Default obstacle positions
        obstacle_positions = [
            (center_x - 30, center_y - 20),
            (center_x + 30, center_y - 25),
            (center_x, center_y - 40),
            (center_x - 20, center_y + 30),
            (center_x + 15, center_y + 25)
        ]
    
    for x, y in obstacle_positions:
        if 0 <= x < width and 0 <= y < height:
            # Set occupied areas around obstacle position
            obstacle_radius = 5
            for oy in range(max(0, y - obstacle_radius), min(height, y + obstacle_radius + 1)):
                for ox in range(max(0, x - obstacle_radius), min(width, x + obstacle_radius + 1)):
                    if (ox - x)**2 + (oy - y)**2 < obstacle_radius**2:
                        ogm[oy, ox] = 1.0
    
    # Use Bresenham algorithm to mark paths from robot to obstacles as free
    for x, y in obstacle_positions:
        if 0 <= x < width and 0 <= y < height:
            rr, cc = bresenham(center_y, center_x, y, x)
            for r_idx, c_idx in zip(rr[:-1], cc[:-1]):  # Not including the last point (obstacle)
                if 0 <= r_idx < height and 0 <= c_idx < width:
                    ogm[r_idx, c_idx] = 0.0
    
    return ogm

def load_scan_data_from_yaml(yaml_file):
    """
    Load laser scan data from YAML file
    
    Args:
        yaml_file: YAML file path
        
    Returns:
        dict: Laser scan data dictionary
    """
    with open(yaml_file, 'r') as file:
        scan_data = yaml.safe_load(file)
    return scan_data

def scan_to_ogm(scan_data, grid_size=100, resolution=0.1, max_range=10.0):
    """
    Convert laser scan data to occupancy grid map
    
    Args:
        scan_data: Dictionary containing laser scan information
        grid_size: Grid map size (grid_size x grid_size)
        resolution: Physical size of each grid (meters)
        max_range: Maximum valid laser distance
        
    Returns:
        numpy array: Occupancy grid map
    """
    # Create blank OGM, 0.5 represents unknown space
    ogm = np.ones((grid_size, grid_size), dtype=np.float32) * 0.5
    
    # Robot position (map center)
    center_x, center_y = grid_size // 2, grid_size // 2
    
    # Get laser data
    angle_min = scan_data['angle_min']
    angle_max = scan_data['angle_max']
    angle_increment = scan_data['angle_increment']
    ranges = scan_data['ranges']
    
    # Fill OGM
    for i, r in enumerate(ranges):
        # Ignore invalid values or values out of range
        if isinstance(r, (int, float)) and (r > max_range or r <= 0):
            continue
            
        # Calculate current laser beam angle
        angle = angle_min + i * angle_increment
        
        # Calculate endpoint coordinates
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        
        # Convert to grid coordinates
        grid_x = int(center_x + x / resolution)
        grid_y = int(center_y + y / resolution)
        
        # Boundary check
        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
            # Mark as occupied
            ogm[grid_y, grid_x] = 1.0
            
            # Use Bresenham algorithm to mark path from robot to obstacle as free
            rr, cc = bresenham(center_y, center_x, grid_y, grid_x)
            for r_idx, c_idx in zip(rr[:-1], cc[:-1]):  # Not including the last point (obstacle)
                if 0 <= r_idx < grid_size and 0 <= c_idx < grid_size:
                    ogm[r_idx, c_idx] = 0.0
    
    return ogm

def visualize_results(ogm, linear_x, angular_z, decoded_ogm=None, save_path=None):
    """
    Visualize OGM and control commands
    
    Args:
        ogm: Original occupancy grid map
        linear_x: Linear velocity
        angular_z: Angular velocity
        decoded_ogm: Decoded OGM (can be None)
        save_path: Path to save results (can be None)
    """
    # Create image
    if decoded_ogm is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    else:
        fig, ax1 = plt.subplots(figsize=(8, 8))
    
    # Draw original OGM
    im1 = ax1.imshow(ogm, cmap='binary', vmin=0.0, vmax=1.0)
    plt.colorbar(im1, ax=ax1, label='Occupancy')
    ax1.set_title('Original OGM')
    ax1.grid(False)
    
    # Mark robot position (center)
    center_y, center_x = ogm.shape[0] // 2, ogm.shape[1] // 2
    ax1.plot(center_x, center_y, 'ro', markersize=10)
    
    # Draw robot's forward direction arrow
    if linear_x > 0:
        arrow_length = 20 * linear_x
        dx = arrow_length * np.cos(angular_z)
        dy = arrow_length * np.sin(angular_z)
        ax1.arrow(center_x, center_y, dx, dy, head_width=5, head_length=7, fc='red', ec='red')
    
    # Add control command information to the image
    ax1.text(10, 10, f'Linear: {linear_x:.2f} m/s', color='white', backgroundcolor='black')
    ax1.text(10, 30, f'Angular: {angular_z:.2f} rad/s', color='white', backgroundcolor='black')
    
    # Draw decoded OGM (if provided)
    if decoded_ogm is not None:
        im2 = ax2.imshow(decoded_ogm, cmap='binary', vmin=0.0, vmax=1.0)
        plt.colorbar(im2, ax=ax2, label='Occupancy')
        ax2.set_title('Decoded OGM')
        ax2.grid(False)
        
        # Mark robot position (center)
        ax2.plot(center_x, center_y, 'ro', markersize=10)
    
    plt.tight_layout()
    
    # Save image (if path provided)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Results saved to {save_path}")
    
    plt.show()

def main():
    """
    Main function, used to test PaSController
    """
    print("Starting PaSController test...")
    
    # Create configuration object
    config = Config()
    
    # Set model paths
    vae_path = "/home/zeng/nav_ws/src/nav2py_template_controller/nav2py_template_controller/nav2py_template_controller/PaS_CrowdNav/data/LabelVAE_CircleFOV30/label_vae_ckpt/label_vae_weight_300.pth"
    # policy_path = "/home/zeng/nav_ws/src/nav2py_template_controller/nav2py_template_controller/nav2py_template_controller/PaS_CrowdNav/data/pas_rnn/checkpoints/33200.pt"
    policy_path = None
    
    # Check if model files exist
    if not os.path.exists(vae_path) or not os.path.exists(policy_path):
        print(f"Warning: Model files not found at {vae_path} or {policy_path}")
        print("Using simple obstacle avoidance instead.")
        vae_path = None
        policy_path = None
    
    # Create PaSController instance
    controller = PaSController(
        vae_path=vae_path,
        policy_path=policy_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        config=config
    )
    
    # Define test mode
    test_mode = "generated"  # "generated" or "yaml"
    
    if test_mode == "yaml":
        # Load laser scan data from YAML file
        yaml_file = "path/to/scan_data.yaml"
        if os.path.exists(yaml_file):
            scan_data = load_scan_data_from_yaml(yaml_file)
            ogm = scan_to_ogm(scan_data)
        else:
            print(f"Warning: YAML file not found at {yaml_file}")
            print("Using generated OGM instead.")
            test_mode = "generated"
    
    if test_mode == "generated":
        # Generate test OGM
        ogm = generate_test_ogm()
    
    # Create results save directory
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Simulate real-time processing
    num_frames = 5
    for i in range(num_frames):
        print(f"Processing frame {i+1}/{num_frames}...")
        
        # If there are multiple OGMs, can update here, here we use the same OGM
        current_ogm = ogm.copy()
        
        # Add some random noise to simulate different frames
        noise = np.random.normal(0, 0.05, current_ogm.shape)
        current_ogm = np.clip(current_ogm + noise, 0.0, 1.0)
        
        # Process OGM
        start_time = time.time()
        linear_x, angular_z = controller.process_ogm(current_ogm)
        end_time = time.time()
        
        print(f"Processing time: {(end_time - start_time) * 1000:.2f} ms")
        print(f"Control command: linear_x={linear_x:.2f}, angular_z={angular_z:.2f}")
        
        # Visualize results
        results_dir = "test_results"
        # save_path = os.path.join(results_dir, f"frame_{i+1}.png")
        # visualize_results(
        #     current_ogm, 
        #     linear_x, 
        #     angular_z, 
        #     controller.last_processed_ogm if controller.model_loaded else None,
        #     save_path
        # )
        
        # Pause, simulate real-time running
        time.sleep(0.5)
    
    print("Test completed. Check 'test_results' directory for visualizations.")

if __name__ == "__main__":
    main()