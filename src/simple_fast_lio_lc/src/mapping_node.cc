#include "common.hpp"
#include "esekf.hpp"
#include "measure_sync.hpp"
#include "simple_fast_lio_lc.hpp"
#include "tf/transform_broadcaster.h"
#include <functional>
#include <map>
#include <memory>
#include <nav_msgs/Path.h>
#include <ros/publisher.h>
#include <ros/rate.h>
#include <ros/ros.h>
#include <ros/time.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <signal.h>
#include <unistd.h>
#include <vector>
#include <visualization_msgs/MarkerArray.h>

// 信号处理回调函数
std::atomic<bool> g_quit(false);

simple_fast_lio2_lc::SimpleFastLio2Lc slamer;
MessageSync measure_syncer(
    std::bind(&simple_fast_lio2_lc::SimpleFastLio2Lc::SlamProcess, &slamer,
              std::placeholders::_1));

void pubTf(simple_fast_lio2_lc::state_ikfom state) {
  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(state.pos.x(), state.pos.y(), state.pos.z()));
  q.setW(state.rot.unit_quaternion().w());
  q.setX(state.rot.unit_quaternion().x());
  q.setY(state.rot.unit_quaternion().y());
  q.setZ(state.rot.unit_quaternion().z());
  transform.setRotation(q);
  tf::StampedTransform stamped_transform(
      transform, ros::Time::now(),
      "camera_init", // parent frame
      "body"         // child frame (change as appropriate)
  );
  br.sendTransform(stamped_transform);
}
ros::Publisher pub_cloud;
ros::Publisher pub_imu;
std::shared_ptr<ros::Publisher> pub_loop_map;
std::shared_ptr<ros::Publisher> pub_loop_curpcl;
ros::Publisher pubPath_;
ros::Publisher pub_loop;
ros::Publisher pub_global_map;
ros::Publisher pub_corr_frame;
void sigIntHandler(int sig) {
  g_quit = true; // 设置退出标志
  ROS_INFO("收到Ctrl+C，准备退出...");

  // 停止所有子线程
  slamer.stopThreads();

  // 强制ROS节点关闭
  ros::shutdown();
}
void pubPath(const std::vector<PathPose> &path) {
  if (!pubPath_) {
    return;
  }
  if (pubPath_.getNumSubscribers() == 0) {
    return;
  }

  nav_msgs::Path path_msg;
  path_msg.header.frame_id = "camera_init";
  path_msg.header.stamp = ros::Time::now();
  for (auto &po : path) {

    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = po.frame_id;
    // pose.header.stamp = ros::Time(po.time);
    pose.pose.position.x = po.position_x;
    pose.pose.position.y = po.position_y;
    pose.pose.position.z = po.position_z;
    pose.pose.orientation.w = po.q.w();
    pose.pose.orientation.x = po.q.x();
    pose.pose.orientation.y = po.q.y();
    pose.pose.orientation.z = po.q.z();
    path_msg.poses.push_back(pose);
  }

  pubPath_.publish(path_msg);
}
void visualizeLoopClosure(const std::map<int, int> &loop_index_contanier,
                          const pcl::PointCloud<PointType> cloud_key_poses_3D) {
  if (loop_index_contanier.empty())
    return;
  if (!pub_loop) {
    return;
  }
  if (pub_loop.getNumSubscribers() == 0) {
    return;
  }

  visualization_msgs::MarkerArray markerArray;
  // loop nodes
  visualization_msgs::Marker markerNode;
  markerNode.header.frame_id = "camera_init";
  markerNode.header.stamp = ros::Time::now();
  markerNode.action = visualization_msgs::Marker::ADD;
  markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
  markerNode.ns = "loop_nodes";
  markerNode.id = 0;
  markerNode.pose.orientation.w = 1;
  markerNode.scale.x = 0.3;
  markerNode.scale.y = 0.3;
  markerNode.scale.z = 0.3;
  markerNode.color.r = 0;
  markerNode.color.g = 0.8;
  markerNode.color.b = 1;
  markerNode.color.a = 1;
  // loop edges
  visualization_msgs::Marker markerEdge;
  markerEdge.header.frame_id = "camera_init";
  markerEdge.header.stamp = ros::Time::now();
  markerEdge.action = visualization_msgs::Marker::ADD;
  markerEdge.type = visualization_msgs::Marker::LINE_LIST;
  markerEdge.ns = "loop_edges";
  markerEdge.id = 1;
  markerEdge.pose.orientation.w = 1;
  markerEdge.scale.x = 0.1;
  markerEdge.color.r = 0.9;
  markerEdge.color.g = 0.9;
  markerEdge.color.b = 0;
  markerEdge.color.a = 1;

  for (auto it = loop_index_contanier.begin(); it != loop_index_contanier.end();
       ++it) {
    int key_cur = it->first;
    int key_pre = it->second;
    geometry_msgs::Point p;
    p.x = cloud_key_poses_3D.points[key_cur].x;
    p.y = cloud_key_poses_3D.points[key_cur].y;
    p.z = cloud_key_poses_3D.points[key_cur].z;
    markerNode.points.push_back(p);
    markerEdge.points.push_back(p);
    p.x = cloud_key_poses_3D.points[key_pre].x;
    p.y = cloud_key_poses_3D.points[key_pre].y;
    p.z = cloud_key_poses_3D.points[key_pre].z;
    markerNode.points.push_back(p);
    markerEdge.points.push_back(p);
  }

  markerArray.markers.push_back(markerNode);
  markerArray.markers.push_back(markerEdge);
  pub_loop.publish(markerArray);
}
void pubGlobalMap(const pcl::PointCloud<PointType>::Ptr &global_map_cloud) {
  if (!pub_global_map) {
    return;
  }
  if (pub_global_map.getNumSubscribers() == 0) {
    return;
  }
  if (global_map_cloud == nullptr) {
    return;
  }
  sensor_msgs::PointCloud2 tempCloud;
  pcl::toROSMsg(*global_map_cloud, tempCloud);
  tempCloud.header.stamp = ros::Time::now();
  tempCloud.header.frame_id = "camera_init";
  pub_global_map.publish(tempCloud);
}

void pubCorrFrame(const PointCloudXYZI::Ptr &CorrFrame) {
  if (!pub_corr_frame) {
    return;
  }
  if (pub_corr_frame.getNumSubscribers() == 0) {
    return;
  }
  if (CorrFrame == nullptr) {
    return;
  }
  sensor_msgs::PointCloud2 tempCloud;
  pcl::toROSMsg(*CorrFrame, tempCloud);
  tempCloud.header.stamp = ros::Time::now();
  tempCloud.header.frame_id = "camera_init";
  pub_corr_frame.publish(tempCloud);
}
int main(int argc, char **argv) {
  ros::init(argc, argv, "bag_reader");
  signal(SIGINT, sigIntHandler);
  ros::NodeHandle nh;
  pubPath_ = nh.advertise<nav_msgs::Path>("/path", 10);
  pub_loop =
      nh.advertise<visualization_msgs::MarkerArray>("/loop_node_edge", 10);
  pub_global_map =
      nh.advertise<sensor_msgs::PointCloud2>("/global_cloud_map", 10);
  pub_corr_frame =
      nh.advertise<sensor_msgs::PointCloud2>("/corr_frame_cloud", 10);
  slamer.pub_odometry_callback_ = std::bind(&pubTf, std::placeholders::_1);
  slamer.pub_corr_frame_callback_ =
      std::bind(&pubCorrFrame, std::placeholders::_1);
  slamer.pub_path_callback_ = std::bind(&pubPath, std::placeholders::_1);
  slamer.visualize_loop_closure_callback_ = std::bind(
      &visualizeLoopClosure, std::placeholders::_1, std::placeholders::_2);
  slamer.visualize_global_map_callback_ =
      std::bind(&pubGlobalMap, std::placeholders::_1);
  slamer.startThreads();

  std::cout << "init this3\n";

  rosbag::Bag bag;
  // ros::Publisher pubLaserCloudMap =
  //     nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 1);
  // pub_cloud =
  //     nh.advertise<sensor_msgs::PointCloud2>("/velodyne_points", 100000);
  // pub_imu = nh.advertise<sensor_msgs::Imu>("/imu/data", 100000);
  // pub_loop_map = std::make_shared<ros::Publisher>(
  //     nh.advertise<sensor_msgs::PointCloud2>("/loop_map", 100000));
  // pub_loop_curpcl = std::make_shared<ros::Publisher>(
  //     nh.advertise<sensor_msgs::PointCloud2>("/loop_curpcl", 100000));

  // slamer.p_map_pub = pub_loop_map;
  // slamer.p_pcl_pub = pub_loop_curpcl;

  try {
    bag.open("/home/ling/simple_fast_lio_lc/src/"
             "utbm_robocar_dataset_20180713_noimage.bag",
             rosbag::bagmode::Read);
  } catch (rosbag::BagException &e) {
    ROS_ERROR_STREAM("无法打开bag文件: " << e.what());
    return -1;
  }
  // 创建视图，指定要读取的话题
  std::vector<std::string> topics = {"/velodyne_points", "/nmea_sentence",
                                     "/imu/data"};
  rosbag::View full_view(bag);
  ros::Time bag_start = full_view.getBeginTime();

  ros::Time bag_end = full_view.getEndTime();
  ROS_INFO_STREAM("Bag file tole time : " << (bag_end - bag_start).toSec()
                                          << " sec");

  double skip_seconds = 60 * 4.0;
  // double skip_seconds = 0;
  ros::Time start_time = bag_start + ros::Duration(skip_seconds);

  // 检查起始时间是否有效
  if (start_time >= bag_end) {
    ROS_ERROR("指定的起始时间超过bag文件的结束时间");
    bag.close();
    return -1;
  }
  rosbag::View view(bag, rosbag::TopicQuery(topics), start_time, bag_end);
  ros::Rate rate(100);
  for (const rosbag::MessageInstance &msg : view) {
    if (!ros::ok()) {
      break;
    }
    if (msg.getTopic() == "/velodyne_points") {
      sensor_msgs::PointCloud2::ConstPtr pcl2_msg =
          msg.instantiate<sensor_msgs::PointCloud2>();
      if (pcl2_msg != nullptr) {
        // pub_cloud.publish(*pcl2_msg);
        // std::cout<<"time: "<<pcl2_msg->header.stamp.toSec()<<" start time:
        // "<<start_time.toSec()<<"\n";
        measure_syncer.ProcessCloud(pcl2_msg);
        // sensor_msgs::PointCloud2 laserCloudMap;
        // pcl::toROSMsg(slamer.getPointcloud(), laserCloudMap);
        // // laserCloudMap.header.stamp = ros::Time().fromSec(1);
        // laserCloudMap.header.frame_id = "camera_init";
        // pubLaserCloudMap.publish(laserCloudMap);
      }
    }
    if (msg.getTopic() == "/imu/data") {
      sensor_msgs::Imu::ConstPtr imu = msg.instantiate<sensor_msgs::Imu>();
      if (imu != nullptr) {
        // pub_imu.publish(*imu);
        measure_syncer.ProcessIMU(imu);
      }
    }

    rate.sleep();
    ros::spinOnce();
  }
  slamer.stopThreads();

  return 0;
}