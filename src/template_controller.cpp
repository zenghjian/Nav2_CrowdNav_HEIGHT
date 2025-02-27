/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Author(s): Shrijit Singh <shrijitsingh99@gmail.com>
 *  Contributor: Pham Cong Trang <phamcongtranghd@gmail.com>
 *  Contributor: Mitchell Sayer <mitchell4408@gmail.com>
 */

 #include <algorithm>
 #include <string>
 #include <memory>
 
 #include "nav2_core/exceptions.hpp"
 #include "nav2_util/node_utils.hpp"
 #include "nav2py_template_controller/template_controller.hpp"
 #include "nav2_util/geometry_utils.hpp"
 #include "ament_index_cpp/get_package_share_directory.hpp"
 
 using std::hypot;
 using std::min;
 using std::max;
 using std::abs;
 using nav2_util::declare_parameter_if_not_declared;
 using nav2_util::geometry_utils::euclidean_distance;
 
 namespace nav2py_template_controller
 {
 
 void TemplateController::configure(
   const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
   std::string name, const std::shared_ptr<tf2_ros::Buffer> tf,
   const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
 {
   node_ = parent;
 
   auto node = node_.lock();
 
   costmap_ros_ = costmap_ros;
   tf_ = tf;
   plugin_name_ = name;
   logger_ = node->get_logger();
   clock_ = node->get_clock();
 
   // Set transform tolerance parameter
   double transform_tolerance;
   nav2_util::declare_parameter_if_not_declared(
     node, plugin_name_ + ".transform_tolerance", rclcpp::ParameterValue(0.1));
   node->get_parameter(plugin_name_ + ".transform_tolerance", transform_tolerance);
   transform_tolerance_ = rclcpp::Duration::from_seconds(transform_tolerance);
 
   // Initialize nav2py
   std::string nav2py_script = ament_index_cpp::get_package_share_directory("nav2py_template_controller") + "/../../lib/nav2py_template_controller/nav2py_run" ;
   nav2py_bootstrap(nav2py_script +
     " --host 127.0.0.1" +
     " --port 0");
 
   // Create publisher for global plan
   global_pub_ = node->create_publisher<nav_msgs::msg::Path>("received_global_plan", 1);
   
   RCLCPP_INFO(
     logger_,
     "Configured controller: %s of type nav2py_template_controller::TemplateController",
     plugin_name_.c_str());
 }
 
 void TemplateController::sendCostmapAndPose(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity)
{
  static int frame_count = 0;
  frame_count++;
  
  // Add prominent frame delimiter for logs
  std::string frame_delimiter(80, '=');
  RCLCPP_INFO(logger_, "\n%s", frame_delimiter.c_str());
  RCLCPP_INFO(logger_, "===== SENDING FRAME %d =====", frame_count);
  RCLCPP_INFO(logger_, "%s", frame_delimiter.c_str());
  
  // Get the costmap
  nav2_costmap_2d::Costmap2D* costmap = costmap_ros_->getCostmap();
  unsigned int width = costmap->getSizeInCellsX();
  unsigned int height = costmap->getSizeInCellsY();
  double resolution = costmap->getResolution();
  double origin_x = costmap->getOriginX();
  double origin_y = costmap->getOriginY();
  
  RCLCPP_INFO(
    logger_,
    "Sending costmap: %dx%d, resolution: %.3f, origin: (%.2f, %.2f)",
    width, height, resolution, origin_x, origin_y);
  
  // Create a structured costmap data
  std::stringstream ss;
  ss << "costmap_info:\n";
  ss << "  width: " << width << "\n";
  ss << "  height: " << height << "\n";
  ss << "  resolution: " << resolution << "\n";
  ss << "  origin_x: " << origin_x << "\n";
  ss << "  origin_y: " << origin_y << "\n";
  
  // Add costmap data - avoid including all cells for debug log to prevent overflow
  RCLCPP_INFO(logger_, "Adding costmap data array of size %dx%d = %d cells", width, height, width * height);
  
  ss << "costmap_data: [";
  for (unsigned int i = 0; i < height; ++i) {
    for (unsigned int j = 0; j < width; ++j) {
      unsigned char cost = costmap->getCost(j, i);
      ss << static_cast<int>(cost);
      if (i < height - 1 || j < width - 1) {
        ss << ", ";
      }
      
      // For debug: log a sample of the costs at the center
      if (frame_count % 10 == 0 && 
          i >= height/2-2 && i <= height/2+2 && 
          j >= width/2-2 && j <= width/2+2) {
        RCLCPP_DEBUG(
          logger_,
          "Costmap value at (%d, %d): %d", 
          i, j, static_cast<int>(cost));
      }
    }
  }
  ss << "]\n";
  
  // Add robot pose
  RCLCPP_INFO(
    logger_,
    "Robot pose: (%.2f, %.2f, %.2f)",
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);
  
  ss << "robot_pose:\n";
  ss << "  position:\n";
  ss << "    x: " << pose.pose.position.x << "\n";
  ss << "    y: " << pose.pose.position.y << "\n";
  ss << "    z: " << pose.pose.position.z << "\n";
  ss << "  orientation:\n";
  ss << "    x: " << pose.pose.orientation.x << "\n";
  ss << "    y: " << pose.pose.orientation.y << "\n";
  ss << "    z: " << pose.pose.orientation.z << "\n";
  ss << "    w: " << pose.pose.orientation.w << "\n";
  
  // Add robot velocity
  RCLCPP_INFO(
    logger_,
    "Robot velocity: linear=(%.2f, %.2f, %.2f), angular=(%.2f, %.2f, %.2f)",
    velocity.linear.x, velocity.linear.y, velocity.linear.z,
    velocity.angular.x, velocity.angular.y, velocity.angular.z);
  
  ss << "robot_velocity:\n";
  ss << "  linear:\n";
  ss << "    x: " << velocity.linear.x << "\n";
  ss << "    y: " << velocity.linear.y << "\n";
  ss << "    z: " << velocity.linear.z << "\n";
  ss << "  angular:\n";
  ss << "    x: " << velocity.angular.x << "\n";
  ss << "    y: " << velocity.angular.y << "\n";
  ss << "    z: " << velocity.angular.z << "\n";
  
  // Generate timestamp
  ss << "timestamp: " << clock_->now().nanoseconds() << "\n";
  ss << "frame_id: " << frame_count << "\n";
  
  // Send the data
  nav2py_send("costmap_pose", {ss.str()});
  
  // Add closing delimiter
  RCLCPP_INFO(logger_, "\n%s", frame_delimiter.c_str());
  RCLCPP_INFO(logger_, "===== FRAME %d SENT =====", frame_count);
  RCLCPP_INFO(logger_, "%s", frame_delimiter.c_str());
}
 
 void TemplateController::cleanup()
 {
   RCLCPP_INFO(
     logger_,
     "Cleaning up controller: %s of type template_controller::TemplateController",
     plugin_name_.c_str());
   nav2py_cleanup();
   global_pub_.reset();
 }
 
 void TemplateController::activate()
 {
   RCLCPP_INFO(
     logger_,
     "Activating controller: %s of type template_controller::TemplateController\"  %s",
     plugin_name_.c_str(),plugin_name_.c_str());
   global_pub_->on_activate();
 }
 
 void TemplateController::deactivate()
 {
   RCLCPP_INFO(
     logger_,
     "Dectivating controller: %s of type template_controller::TemplateController\"  %s",
     plugin_name_.c_str(),plugin_name_.c_str());
   global_pub_->on_deactivate();
 }
 
 void TemplateController::setSpeedLimit(const double& speed_limit, const bool& percentage)
 {
   (void) speed_limit;
   (void) percentage;
 }
 
 geometry_msgs::msg::TwistStamped TemplateController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker * goal_checker)
{

  (void)goal_checker;

  auto transformed_plan = transformGlobalPlan(pose);
  
  try {
    std::string path_yaml = nav_msgs::msg::to_yaml(transformed_plan, true);
    
    nav2py_send("path", {path_yaml});
    RCLCPP_INFO(logger_, "Sent path data to Python controller");
  } catch (const std::exception& e) {
    RCLCPP_ERROR(
      logger_,
      "Error sending path to Python controller: %s", e.what());
  }
  
  try {
    sendCostmapAndPose(pose, velocity);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(
      logger_,
      "Error sending costmap and pose data: %s", e.what());
  }
  
  geometry_msgs::msg::TwistStamped cmd_vel;
  cmd_vel.header.frame_id = pose.header.frame_id;
  cmd_vel.header.stamp = clock_->now();
  
  try {
    RCLCPP_INFO(logger_, "Waiting for velocity command from Python controller...");
    cmd_vel.twist = wait_for_cmd_vel();
    
    RCLCPP_INFO(
      logger_, 
      "Received velocity command: linear_x=%.2f, angular_z=%.2f", 
      cmd_vel.twist.linear.x, cmd_vel.twist.angular.z);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(
      logger_,
      "Error receiving velocity command: %s", e.what());
    
    cmd_vel.twist.linear.x = 0.0;
    cmd_vel.twist.angular.z = 0.0;
  }

  return cmd_vel;
}
 
 void TemplateController::setPlan(const nav_msgs::msg::Path & path)
 {
   global_pub_->publish(path);
   global_plan_ = path;
 }
 
 nav_msgs::msg::Path
 TemplateController::transformGlobalPlan(
   const geometry_msgs::msg::PoseStamped & pose)
 {
   // Original mplementation taken fron nav2_dwb_controller
 
   if (global_plan_.poses.empty()) {
     throw nav2_core::PlannerException("Received plan with zero length");
   }
 
   // let's get the pose of the robot in the frame of the plan
   geometry_msgs::msg::PoseStamped robot_pose;
   if (!transformPose(
       tf_, global_plan_.header.frame_id, pose,
       robot_pose, transform_tolerance_))
   {
     throw nav2_core::PlannerException("Unable to transform robot pose into global plan's frame");
   }
 
   // We'll discard points on the plan that are outside the local costmap
   nav2_costmap_2d::Costmap2D * costmap = costmap_ros_->getCostmap();
   double dist_threshold = std::max(costmap->getSizeInCellsX(), costmap->getSizeInCellsY()) *
     costmap->getResolution() / 2.0;
 
   // First find the closest pose on the path to the robot
   auto transformation_begin = nav2_util::geometry_utils::min_by(
     global_plan_.poses.begin(), global_plan_.poses.end(),
     [&robot_pose](const geometry_msgs::msg::PoseStamped & ps) {
       return euclidean_distance(robot_pose, ps);
     });
 
   // From the closest point, look for the first point that's further then dist_threshold from the
   // robot. These points are definitely outside of the costmap so we won't transform them.
   auto transformation_end = std::find_if(
     transformation_begin, end(global_plan_.poses),
     [&](const auto & global_plan_pose) {
       return euclidean_distance(robot_pose, global_plan_pose) > dist_threshold;
     });
 
   // Helper function for the transform below. Transforms a PoseStamped from global frame to local
   auto transformGlobalPoseToLocal = [&](const auto & global_plan_pose) {
       // We took a copy of the pose, let's lookup the transform at the current time
       geometry_msgs::msg::PoseStamped stamped_pose, transformed_pose;
       stamped_pose.header.frame_id = global_plan_.header.frame_id;
       stamped_pose.header.stamp = pose.header.stamp;
       stamped_pose.pose = global_plan_pose.pose;
       transformPose(
         tf_, costmap_ros_->getBaseFrameID(),
         stamped_pose, transformed_pose, transform_tolerance_);
       return transformed_pose;
     };
 
   // Transform the near part of the global plan into the robot's frame of reference.
   nav_msgs::msg::Path transformed_plan;
   std::transform(
     transformation_begin, transformation_end,
     std::back_inserter(transformed_plan.poses),
     transformGlobalPoseToLocal);
   transformed_plan.header.frame_id = costmap_ros_->getBaseFrameID();
   transformed_plan.header.stamp = pose.header.stamp;
 
   // Remove the portion of the global plan that we've already passed so we don't
   // process it on the next iteration (this is called path pruning)
   global_plan_.poses.erase(begin(global_plan_.poses), transformation_begin);
   global_pub_->publish(transformed_plan);
 
   if (transformed_plan.poses.empty()) {
     throw nav2_core::PlannerException("Resulting plan has 0 poses in it.");
   }
 
   return transformed_plan;
 }
 
 bool TemplateController::transformPose(
   const std::shared_ptr<tf2_ros::Buffer> tf,
   const std::string frame,
   const geometry_msgs::msg::PoseStamped & in_pose,
   geometry_msgs::msg::PoseStamped & out_pose,
   const rclcpp::Duration & transform_tolerance
 ) const
 {
   // Implementation taken as is fron nav_2d_utils in nav2_dwb_controller
 
   if (in_pose.header.frame_id == frame) {
     out_pose = in_pose;
     return true;
   }
 
   try {
     tf->transform(in_pose, out_pose, frame);
     return true;
   } catch (tf2::ExtrapolationException & ex) {
     auto transform = tf->lookupTransform(
       frame,
       in_pose.header.frame_id,
       tf2::TimePointZero
     );
     if (
       (rclcpp::Time(in_pose.header.stamp) - rclcpp::Time(transform.header.stamp)) >
       transform_tolerance)
     {
       RCLCPP_ERROR(
         rclcpp::get_logger("tf_help"),
         "Transform data too old when converting from %s to %s",
         in_pose.header.frame_id.c_str(),
         frame.c_str()
       );
       RCLCPP_ERROR(
         rclcpp::get_logger("tf_help"),
         "Data time: %ds %uns, Transform time: %ds %uns",
         in_pose.header.stamp.sec,
         in_pose.header.stamp.nanosec,
         transform.header.stamp.sec,
         transform.header.stamp.nanosec
       );
       return false;
     } else {
       tf2::doTransform(in_pose, out_pose, transform);
       return true;
     }
   } catch (tf2::TransformException & ex) {
     RCLCPP_ERROR(
       rclcpp::get_logger("tf_help"),
       "Exception in transformPose: %s",
       ex.what()
     );
     return false;
   }
   return false;
 }
 
 }  // namespace nav2py_template_controller
 
 // Register this controller as a nav2_core plugin
 PLUGINLIB_EXPORT_CLASS(nav2py_template_controller::TemplateController, nav2_core::Controller)