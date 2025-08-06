//
// Created by xiang on 22-9-9.
//

#ifndef SLAM_IN_AUTO_DRIVING_MEASURE_SYNC_H
#define SLAM_IN_AUTO_DRIVING_MEASURE_SYNC_H

// #include "cloud_convert.h"
// #include "common/imu.h"
// #include "common/point_types.h"

#include "common.hpp"
#include "preprocess.hpp"
#include <deque>
#include <sensor_msgs/Imu.h>
/**
 * 将激光数据和IMU数据同步
 */
class MessageSync {
public:
  using Callback = std::function<void(const MeasureGroup &)>;

  MessageSync(Callback cb) : callback_(cb), p_pre_(new Preprocess()) {
    p_pre_->set(VELO32, 2, 16);
    p_pre_->N_SCANS=32;
    p_pre_->SCAN_RATE=10;
    p_pre_->time_unit=US;
  }

  /// 初始化
  void Init(const std::string &yaml);

  /// 处理IMU数据
  void ProcessIMU(sensor_msgs::Imu::ConstPtr _msg_in) {
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*_msg_in));
    msg->header.stamp = get_ros_time(get_time_sec(_msg_in->header.stamp));

    double timestamp = get_time_sec(msg->header.stamp);

    if (timestamp < last_timestamp_imu_) {
      std::cerr << "imu loop back, clear buffer\n";
      imu_buffer_.clear();
    }

    last_timestamp_imu_ = timestamp;
    imu_buffer_.push_back(msg);
    // std::cout<<"ProcessIMU: " << last_timestamp_imu_ << std::endl;
  }

  /**
   * 处理sensor_msgs::PointCloud2点云
   * @param msg
   */
  void ProcessCloud(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    double cur_time = get_time_sec(msg->header.stamp);
    if (cur_time < last_timestamp_lidar_) {
      std::cerr << "lidar loop back, clear buffer\n";
      lidar_buffer_.clear();
    }
    last_timestamp_lidar_ = cur_time;
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());

    p_pre_->process(msg, ptr);
    lidar_buffer_.push_back(ptr);
    time_buffer_.push_back(last_timestamp_lidar_);

    Sync();
    // std::cout<<"ProcessCloud: " << last_timestamp_lidar_ << std::endl;
  }


private:
  /// 尝试同步IMU与激光数据，成功时返回true
  bool Sync() {
    if (lidar_buffer_.empty() || imu_buffer_.empty()) {
      return false;
    }

    if (!lidar_pushed_) {
      measures_.lidar = lidar_buffer_.front();
      measures_.lidar_beg_time = time_buffer_.front();

      lidar_end_time_ = measures_.lidar_beg_time +
                        measures_.lidar->points.back().curvature / double(1000);

      measures_.lidar_end_time = lidar_end_time_;
      lidar_pushed_ = true;
    }

    if (last_timestamp_imu_ < lidar_end_time_) {
      return false;
    }

    double imu_time = get_time_sec(imu_buffer_.front()->header.stamp);
    measures_.imu.clear();
    while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_)) {
      imu_time = get_time_sec(imu_buffer_.front()->header.stamp);
      if (imu_time > lidar_end_time_) {
        break;
      }
      measures_.imu.push_back(imu_buffer_.front());
      imu_buffer_.pop_front();
    }

    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
    lidar_pushed_ = false;

    if (callback_) {
      callback_(measures_);
    }

    return true;
  }

  Callback callback_;                            // 同步数据后的回调函数
  
  std::deque<PointCloudXYZI::Ptr> lidar_buffer_; // 雷达数据缓冲
  std::deque<sensor_msgs::Imu::Ptr> imu_buffer_; // imu数据缓冲
  double last_timestamp_imu_ = -1.0;             // 最近imu时间
  double last_timestamp_lidar_ = 0;              // 最近lidar时间
  std::shared_ptr<Preprocess> p_pre_ = nullptr;  // 点云转换
  std::deque<double> time_buffer_;
  bool lidar_pushed_ = false;
  MeasureGroup measures_;
  double lidar_end_time_ = 0;
};

#endif // SLAM_IN_AUTO_DRIVING_MEASURE_SYNC_H
