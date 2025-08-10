#include "common.hpp"
#include "esekf.hpp"
#include "ikd-Tree/ikd_Tree.h"
#include "imu_processing.hpp"
#include "sophus/so3.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <omp.h>
#include <ostream>
#include <pcl/features/normal_3d.h> // 法向量计算
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/filter.h> // 移除无效点
#include <pcl/filters/voxel_grid.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/publisher.h>
#include <sensor_msgs/PointCloud2.h>
#include <small_gicp/benchmark/read_points.hpp>
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/pcl/pcl_registration.hpp>
#include <small_gicp/registration/registration_helper.hpp>
#include <small_gicp/util/downsampling_omp.hpp>
#include <thread>
#include <vector>
#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
namespace simple_fast_lio2_lc {
const float MOV_THRESHOLD = 1.5f;
inline double DET_RANGE = 100;
class SimpleFastLio2Lc {
private:
  ESEKalmanFilter kf_;
  PointCloudXYZI::Ptr p_feats_undistort_;
  PointCloudXYZI::Ptr p_feats_down_body_;
  PointCloudXYZI::Ptr p_feats_down_world_;
  KD_TREE<PointType> ikdtree_;
  pcl::ApproximateVoxelGrid<PointType> filter_cloud_;
  pcl::ApproximateVoxelGrid<PointType> filter_map_;
  // pcl::VoxelGrid<PointType> filter_cloud_;
  // pcl::VoxelGrid<PointType> filter_map_;
  vector<PointVector> nearest_points_;

  int feats_down_size_ = 0;

  bool flg_first_scan_ = true;
  double first_lidar_time_ = 0;
  bool flg_EKF_inited_ = false;
  double filter_size_cloud_min_ = 0.1;
  double filter_size_map_min_ = 0.1;
  double cube_len_ = 100;

  double gyr_cov_ = 0.1;
  double acc_cov_ = 0.1;
  double b_gyr_cov_ = 0.0001;
  double b_acc_cov_ = 0.0001;
  vector<double> extrin_T_{0, 0, 0.28};
  vector<double> extrin_R_{1., 0., 0., 0., 1., 0., 0., 0., 1.};
  V3D lidar_T_wrt_imu_;
  M3D lidar_R_wrt_imu_;
  std::shared_ptr<ImuProcess> p_imu_;
  Eigen::Vector3d pos_lid_;

  vector<pcl::PointCloud<PointType>::Ptr> cloud_key_frames_;
  pcl::PointCloud<PointType>::Ptr cloud_key_poses_3D_;
  pcl::PointCloud<PointTypePose>::Ptr cloud_key_poses_6D_;
  pcl::PointCloud<PointType>::Ptr copy_cloud_key_poses_3D_;
  pcl::PointCloud<PointTypePose>::Ptr copy_cloud_key_poses_6D_;
  double key_frame_adding_angle_threshold_ = 0.2;
  double key_frame_adding_distance_threshold_ = 0.5;
  pcl::KdTreeFLANN<PointType>::Ptr kdtree_history_key_poses_;
  float history_keyframe_search_radius_ = 15.0;
  float history_keyframe_search_time_diff_ = 30.0;
  int history_keyframe_search_num_ = 25;
  float history_keyframe_fitness_score_ = 0.5;
  bool localmap_initialized_ = false;
  bool point_selected_surf_[100000] = {0};
  float res_last_[100000] = {0.0};
  pcl::PointCloud<PointType>::Ptr pcl_wait_pub_;
  std::mutex mtx_;
  PointCloudXYZI::Ptr normvec_;
  Eigen::Quaterniond geo_quat_;
  BoxPointType localmap_points_;
  PointCloudXYZI::Ptr laser_cloud_ori_;
  PointCloudXYZI::Ptr corr_normvect_;
  PointCloudXYZI::Ptr featsFromMap;

  int lidar_time_ = 0;

  gtsam::NonlinearFactorGraph gtsam_graph_;
  gtsam::Values initial_estimate_;

  // 回环的索引字典，从当前帧到回环节点的索引
  map<int, int> loop_index_container_;
  // 所有回环配对关系
  vector<pair<int, int>> loop_index_queue_;
  // 所有回环的姿态配对关系
  vector<gtsam::Pose3> loop_pose_queue_;
  // 每个回环因子的噪声模型
  vector<gtsam::noiseModel::Diagonal::shared_ptr> loop_noise_queue_;
  // 回环锁
  std::mutex mtx_loop_info_;
  // 非线性优化器
  gtsam::ISAM2 *isam;
  bool loop_closure_enable_flag_ = true;

  // 做回环检测时使用ICP时的点云降采样器
  pcl::VoxelGrid<PointType> down_size_filter_icp_;
  // 构建局部地图时都挑选的关键帧做降采样
  pcl::VoxelGrid<PointType> down_size_filter_surrounding_key_poses_;

  std::thread loop_thread_;
  std::thread global_map_thread_;
  bool aLoopIsClosed = false;
  gtsam::Values isamCurrentEstimate;
  std::atomic<bool> exit_flag_{false};
  std::vector<PathPose> global_path_;

public:
  // std::shared_ptr<ros::Publisher> p_map_pub;
  // std::shared_ptr<ros::Publisher> p_pcl_pub;
  std::function<void(const state_ikfom &)> pub_odometry_callback_;
  std::function<void(const std::vector<PathPose> &)> pub_path_callback_;
  std::function<void(const std::map<int, int> &,
                     const pcl::PointCloud<PointType>)>
      visualize_loop_closure_callback_;
  std::function<void(const pcl::PointCloud<PointType>::Ptr &)>
      visualize_global_map_callback_;
  std::function<void(const PointCloudXYZI::Ptr &)> pub_corr_frame_callback_;
  state_ikfom state_point_;

  void SlamProcess(const MeasureGroup &measures) {
    lidar_time_ = measures.lidar_beg_time;
    if (flg_first_scan_) {
      first_lidar_time_ = measures.lidar_beg_time;
      p_imu_->first_lidar_time_ = first_lidar_time_;
      flg_first_scan_ = false;
      return;
    }
    p_imu_->process(measures, kf_, p_feats_undistort_);
    state_point_ = kf_.getState();
    pos_lid_ = state_point_.pos +
               state_point_.rot.matrix() * state_point_.offset_T_L_I;

    if (p_feats_undistort_->empty() || (p_feats_undistort_ == NULL)) {
      std::cerr << "No point, skip this scan!\n";
      return;
    }
    flg_EKF_inited_ = (measures.lidar_beg_time - first_lidar_time_) < INIT_TIME
                          ? false
                          : true;
    /*** Segment the map in lidar FOV ***/
    lasermapFovSegment();
    // std::cout<<p_feats_undistort_->size()<<"\n";

    /*** downsample the feature points in a scan ***/
    // filter_cloud_.setDownsampleAllData(bool downsample)
    filter_cloud_.setInputCloud(p_feats_undistort_);
    filter_cloud_.filter(*p_feats_down_body_);

    feats_down_size_ = p_feats_down_body_->points.size();
    p_feats_down_world_->resize(feats_down_size_);
    if (feats_down_size_ < 5) {
      std::cerr << "points num too little,skip \n";
      return;
    }
    // trans body_point to world_point
    for (int i = 0; i < feats_down_size_; i++) {
      pointBodyToWorld(&(p_feats_down_body_->points[i]),
                       &(p_feats_down_world_->points[i]));
    }
    /*** initialize the map kdtree ***/
    if (ikdtree_.Root_Node == nullptr) {
      std::cerr << "Initialize the map kdtree\n";
      ikdtree_.set_downsample_param(filter_size_map_min_);
      ikdtree_.Build(p_feats_down_world_->points);
      // saveKeyFrame();

      return;
    }
    normvec_->resize(feats_down_size_);
    nearest_points_.resize(feats_down_size_);
    double solve_H_time = 0;
    kf_.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
    state_point_ = kf_.getState();
    saveKeyFramesAndFactor();
    correctPoses();

    pos_lid_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;
    geo_quat_ = state_point_.rot.unit_quaternion();

    mapIncremental();

    if (pub_odometry_callback_) {
      pub_odometry_callback_(state_point_);
    }
    if (pub_path_callback_) {
      pub_path_callback_(getPath());
    }
    if (pub_corr_frame_callback_) {
      pub_corr_frame_callback_(p_feats_down_world_);
    }
  }
  PointCloudXYZI getPointcloud() {

    PointCloudXYZI featsFromMap;
    // featsFromMap.resize(feats_down_size_);
    // for (int i = 0; i < feats_down_size_; i++) {
    //   pointBodyToWorld(&(p_feats_down_body_->points[i]),
    //                    &(featsFromMap.points[i]));
    // }
    pcl::PointCloud<PointTypePose>::Ptr copy_cloud_key_poses_6D(
        new pcl::PointCloud<PointTypePose>());
    mtx_.lock();
    *copy_cloud_key_poses_6D = *cloud_key_poses_6D_;
    mtx_.unlock();

    for (int i = 0; i < cloud_key_frames_.size(); i++) {
      featsFromMap += *transformPointCloud(cloud_key_frames_[i],
                                           &copy_cloud_key_poses_6D->points[i]);
    }
    // PointVector().swap(ikdtree_.PCL_Storage);
    // ikdtree_.flatten(ikdtree_.Root_Node, ikdtree_.PCL_Storage, NOT_RECORD);
    // featsFromMap.clear();
    // featsFromMap.points = ikdtree_.PCL_Storage;

    return featsFromMap;
  }
  SimpleFastLio2Lc() {
    allocateMemory();

    auto f = [this](const double &dt, const state_ikfom &x,
                    const input_ikfom &in) {
      Eigen::Vector3d omega = in.gyro - x.bg;
      Eigen::Vector3d a_inertial = x.rot.matrix() * (in.acc - x.ba);
      state_ikfom out;
      out.pos = x.pos + x.vel * dt;
      out.rot = x.rot * Sophus::SO3d::exp(omega * dt);
      out.offset_R_L_I =
          x.offset_R_L_I * Sophus::SO3d::exp(Eigen::Vector3d(0, 0, 0));
      out.vel = x.vel + (a_inertial + x.grav) * dt;
      out.ba = x.ba + Eigen::Vector3d(0, 0, 0);
      out.bg = x.bg + Eigen::Vector3d(0, 0, 0);
      out.grav = x.grav + Eigen::Vector3d(0, 0, 0);

      return out;
    };
    auto getJacbi_f = [this](const double &dt, const state_ikfom &x,
                             const input_ikfom &in) {
      // clang-format off
/*
x(delta_pos__, delta_theta, delta_RLI__, delta_TLI__, delta_vel__, delta_bgyro, delta_bacc_, delta_grav_)
I            0            0            0           I*dt          0            0            0

0       exp(-W_i*dt)      0            0            0        -A(W_i*dt)       0            0

0            0            I            0            0            0            0            0

0            0            0            I            0            0            0            0

0        -R(acc^)*dt      0            0            I            0          -R*dt         I*dt

0            0            0            0            0            I            0            0

0            0            0            0            0            0            I            0

0            0            0            0            0            0            0            I
*/
      // clang-format on
      Eigen::Matrix<double, 24, 24> jacbi =
          Eigen::Matrix<double, 24, 24>::Identity();
      jacbi.block<3, 3>(0, 12) = Eigen::Matrix3d::Identity() * dt;
      Eigen::Vector3d acc_ = in.acc - x.ba; // 测量加速度 = a_m - bias
      Eigen::Vector3d omega = in.gyro - x.bg;
      Eigen::Vector3d delta_omega = omega * dt;
      jacbi.block<3, 3>(3, 3) = Sophus::SO3d::exp(delta_omega * (-1)).matrix();
      jacbi.block<3, 3>(3, 15) = get_A_matrix(delta_omega) * (-1);
      jacbi.block<3, 3>(12, 3) =
          x.rot.matrix() * Sophus::SO3d::hat(acc_) * dt * (-1);
      jacbi.block<3, 3>(12, 18) = x.rot.matrix() * dt * (-1);
      jacbi.block<3, 3>(12, 21) = Eigen::Matrix<double, 3, 3>::Identity() * dt;
      return jacbi;
    };
    auto getJacbi_f_w = [this](const double &dt,
                               const simple_fast_lio2_lc::state_ikfom &x,
                               const simple_fast_lio2_lc::input_ikfom &in) {
      // clang-format off
/*
x(delta_pos__, delta_theta, delta_RLI__, delta_TLI__, delta_vel__, delta_bgyro, delta_bacc_, delta_grav_)
w(  noise_gyry_    noise_acc__    noise_b_gyr    noise_b_acc)
   0              0              0              0

-A(W_i*dt)*dt        0              0              0

   0              0              0              0

   0              0              0              0

   0            -R*dt            0              0

   0              0             I*dt            0

   0              0              0             I*dt

   0              0              0              0
*/
      // clang-format on
      Eigen::Matrix<double, 24, 12> jacbi_f_w =
          Eigen::Matrix<double, 24, 12>::Zero();
      Eigen::Vector3d omega = in.gyro - x.bg;
      Eigen::Vector3d delta_omega = omega * dt;
      jacbi_f_w.template block<3, 3>(12, 3) = -x.rot.matrix() * dt;
      jacbi_f_w.template block<3, 3>(3, 0) = -get_A_matrix(delta_omega) * dt;
      jacbi_f_w.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity() * dt;
      jacbi_f_w.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity() * dt;
      return jacbi_f_w;
    };

    auto h_share_model = [this](const state_ikfom &s,
                                dyn_share_datastruct &ekfom_data) {
      laser_cloud_ori_->clear();
      corr_normvect_->clear();
      double total_residual = 0.0;

/** closest surface search and residual computation **/
#ifdef MP_EN
      omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
      for (int i = 0; i < feats_down_size_; i++) {
        PointType &point_body = p_feats_down_body_->points[i];
        PointType &point_world = p_feats_down_world_->points[i];

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) +
                     s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = nearest_points_[i];

        if (ekfom_data.converge) {
          /** Find the closest surfaces in the map **/
          ikdtree_.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near,
                                  pointSearchSqDis);
          point_selected_surf_[i] =
              points_near.size() < NUM_MATCH_POINTS        ? false
              : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                           : true;
        }

        if (!point_selected_surf_[i])
          continue;

        VF(4) pabcd;
        point_selected_surf_[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f)) {
          float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y +
                      pabcd(2) * point_world.z + pabcd(3);
          float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

          if (s > 0.9) {
            point_selected_surf_[i] = true;
            normvec_->points[i].x = pabcd(0);
            normvec_->points[i].y = pabcd(1);
            normvec_->points[i].z = pabcd(2);
            normvec_->points[i].intensity = pd2;
            res_last_[i] = abs(pd2);
          }
        }
      }

      int effct_feat_num = 0;

      for (int i = 0; i < feats_down_size_; i++) {
        if (point_selected_surf_[i]) {
          laser_cloud_ori_->points[effct_feat_num] =
              p_feats_down_body_->points[i];
          corr_normvect_->points[effct_feat_num] = normvec_->points[i];
          total_residual += res_last_[i];
          effct_feat_num++;
        }
      }

      if (effct_feat_num < 24) {
        ekfom_data.valid = false;
        std::cerr << "No Effective Points!" << std::endl;
        // ROS_WARN("No Effective Points! \n");
        return;
      }

      double res_mean_last = total_residual / effct_feat_num;
      // double match_time  += omp_get_wtime() - match_start;
      double solve_start_ = omp_get_wtime();

      /*** Computation of Measuremnt Jacobian matrix H and measurents vector
       * ***/
      ekfom_data.jacbi_H = Eigen::MatrixXd::Zero(effct_feat_num, 12); // 23
      ekfom_data.h.resize(effct_feat_num);

      for (int i = 0; i < effct_feat_num; i++) {
        const PointType &laser_p = laser_cloud_ori_->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect_->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.unit_quaternion().conjugate() * norm_vec);
        V3D A(point_crossmat * C);

        ekfom_data.jacbi_H.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z,
            VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
      }
    };
    kf_.init(f, getJacbi_f, getJacbi_f_w, h_share_model, 4);
    std::cout << "init this1\n";
    lidar_T_wrt_imu_ << VEC_FROM_ARRAY(extrin_T_);

    lidar_R_wrt_imu_ << MAT_FROM_ARRAY(extrin_R_);

    filter_cloud_.setLeafSize(filter_size_cloud_min_, filter_size_cloud_min_,
                              filter_size_cloud_min_);
    filter_map_.setLeafSize(filter_size_map_min_, filter_size_map_min_,
                            filter_size_map_min_);
    // 回环检测时ICP匹配前点云降采样器
    down_size_filter_icp_.setLeafSize(0.5, 0.5, 0.5);
    // 构建局部地图时对检索出的位姿点做降采样
    down_size_filter_surrounding_key_poses_.setLeafSize(2, 2, 2);
    p_imu_->setParam(lidar_T_wrt_imu_, lidar_R_wrt_imu_,
                     V3D(gyr_cov_, gyr_cov_, gyr_cov_),
                     V3D(acc_cov_, acc_cov_, acc_cov_),
                     V3D(b_gyr_cov_, b_gyr_cov_, b_gyr_cov_),
                     V3D(b_acc_cov_, b_acc_cov_, b_acc_cov_));

    std::cout << "init this\n";
  }
  ~SimpleFastLio2Lc() {
    if (loop_thread_.joinable()) {
      loop_thread_.join();
    }
    if (global_map_thread_.joinable()) {
      global_map_thread_.join();
    }
    if (isam != nullptr) {
      delete isam;
      isam = nullptr;
    }
  }

  void stopThreads() {
    exit_flag_ = true; // 设置退出标志
    if (loop_thread_.joinable()) {
      loop_thread_.join(); // 等待子线程结束
    }
    if (global_map_thread_.joinable()) {
      global_map_thread_.join();
    }
  }
  void startThreads() {
    loop_thread_ = std::thread(&SimpleFastLio2Lc::loopClosureThread, this);
    global_map_thread_ =
        std::thread(&SimpleFastLio2Lc::visualizeGlobalMapThread, this);
  }
  void setCallBack(
      std::function<void(const state_ikfom &)> pub_odometry_cb = nullptr,
      std::function<void(const std::vector<PathPose> &)> pub_path_cb = nullptr,
      std::function<void(const std::map<int, int> &,
                         const pcl::PointCloud<PointType>)>
          vis_loop_closure_cb = nullptr,
      std::function<void(const pcl::PointCloud<PointType>::Ptr &)>
          vis_global_map_cb = nullptr,
      std::function<void(const PointCloudXYZI::Ptr &)> pub_corr_frame_cb =
          nullptr) {
    pub_odometry_callback_ = pub_odometry_cb;
    pub_path_callback_ = pub_path_cb;
    visualize_loop_closure_callback_ = vis_loop_closure_cb;
    visualize_global_map_callback_ = vis_global_map_cb;
    pub_corr_frame_callback_ = pub_corr_frame_cb;
  }

private:
  state_ikfom getState() { return state_point_; }

  pcl::PointCloud<PointType>::Ptr getGlobalMap() {

    if (cloud_key_poses_3D_->points.empty() == true)
      return nullptr;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(
        new pcl::KdTreeFLANN<PointType>());
    ;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(
        new pcl::PointCloud<PointType>());

    // kd-tree to find near key frames to visualize
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    // search near key frames to visualize
    mtx_.lock();
    kdtreeGlobalMap->setInputCloud(cloud_key_poses_3D_);
    kdtreeGlobalMap->radiusSearch(cloud_key_poses_3D_->back(), 1000,
                                  pointSearchIndGlobalMap,
                                  pointSearchSqDisGlobalMap, 0);
    mtx_.unlock();

    for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
      globalMapKeyPoses->push_back(
          cloud_key_poses_3D_->points[pointSearchIndGlobalMap[i]]);
    // downsample near selected key frames
    pcl::VoxelGrid<PointType>
        downSizeFilterGlobalMapKeyPoses; // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setLeafSize(
        1, 1,
        1); // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
    for (auto &pt : globalMapKeyPosesDS->points) {
      kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap,
                                      pointSearchSqDisGlobalMap);
      pt.intensity =
          cloud_key_poses_3D_->points[pointSearchIndGlobalMap[0]].intensity;
    }

    // extract visualized and downsampled key frames
    for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i) {
      if (pointDistance(globalMapKeyPosesDS->points[i],
                        cloud_key_poses_3D_->back()) > 5000)
        continue;
      int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
      *globalMapKeyFrames +=
          *transformPointCloud(cloud_key_frames_[thisKeyInd],
                               &cloud_key_poses_6D_->points[thisKeyInd]);
    }
    // downsample visualized points
    pcl::VoxelGrid<PointType>
        downSizeFilterGlobalMapKeyFrames; // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(0.5, 0.5, 0.5);
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
    return globalMapKeyFramesDS;
  }
  std::vector<PathPose> getPath() { return global_path_; }

  void lasermapFovSegment() {
    std::vector<BoxPointType> cub_needrm;
    cub_needrm.clear();
    // pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid_;
    if (!localmap_initialized_) {
      for (int i = 0; i < 3; i++) {
        localmap_points_.vertex_min[i] = pos_LiD(i) - cube_len_ / 2.0;
        localmap_points_.vertex_max[i] = pos_LiD(i) + cube_len_ / 2.0;
      }
      localmap_initialized_ = true;
      return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++) {
      dist_to_map_edge[i][0] =
          fabs(pos_LiD(i) - localmap_points_.vertex_min[i]);
      dist_to_map_edge[i][1] =
          fabs(pos_LiD(i) - localmap_points_.vertex_max[i]);
      if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE ||
          dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        need_move = true;
    }
    if (!need_move)
      return;
    BoxPointType new_localmap_points, tmp_boxpoints;
    new_localmap_points = localmap_points_;
    float mov_dist =
        max((cube_len_ - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9,
            double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++) {
      tmp_boxpoints = localmap_points_;
      if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
        new_localmap_points.vertex_max[i] -= mov_dist;
        new_localmap_points.vertex_min[i] -= mov_dist;
        tmp_boxpoints.vertex_min[i] = localmap_points_.vertex_max[i] - mov_dist;
        cub_needrm.push_back(tmp_boxpoints);
      } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
        new_localmap_points.vertex_max[i] += mov_dist;
        new_localmap_points.vertex_min[i] += mov_dist;
        tmp_boxpoints.vertex_max[i] = localmap_points_.vertex_min[i] + mov_dist;
        cub_needrm.push_back(tmp_boxpoints);
      }
    }
    localmap_points_ = new_localmap_points;

    PointVector points_history;
    ikdtree_.acquire_removed_points(points_history);
    if (cub_needrm.size() > 0)
      int kdtree_delete_counter = ikdtree_.Delete_Point_Boxes(cub_needrm);
  }
  void allocateMemory() {
    // p_pre_.reset(new PreProcess());
    p_imu_.reset(new ImuProcess());

    extrin_T_.resize(3, 0);
    extrin_R_.resize(9, 0);
    lidar_T_wrt_imu_.setZero();
    lidar_R_wrt_imu_.setIdentity();
    memset(point_selected_surf_, true, sizeof(point_selected_surf_));
    memset(res_last_, -1000.0f, sizeof(res_last_));
    p_feats_undistort_.reset(new PointCloudXYZI());
    p_feats_down_body_.reset(new PointCloudXYZI());
    p_feats_down_world_.reset(new PointCloudXYZI());
    cloud_key_poses_3D_.reset(new pcl::PointCloud<PointType>());
    cloud_key_poses_6D_.reset(new pcl::PointCloud<PointTypePose>());
    copy_cloud_key_poses_3D_.reset(new pcl::PointCloud<PointType>());
    copy_cloud_key_poses_6D_.reset(new pcl::PointCloud<PointTypePose>());
    cloud_key_frames_.clear();
    normvec_.reset(new PointCloudXYZI(100000, 1));
    laser_cloud_ori_.reset(new PointCloudXYZI(100000, 1));
    corr_normvect_.reset(new PointCloudXYZI(100000, 1));
    featsFromMap.reset(new PointCloudXYZI());
    kdtree_history_key_poses_.reset(
        new pcl::KdTreeFLANN<pcl::PointXYZINormal>());
    // isam 优化器参数
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    isam = new gtsam::ISAM2(parameters);
  }

  void mapIncremental() {
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size_);
    PointNoNeedDownsample.reserve(feats_down_size_);
    for (int i = 0; i < feats_down_size_; i++) {
      /* transform to world frame */
      pointBodyToWorld(&(p_feats_down_body_->points[i]),
                       &(p_feats_down_world_->points[i]));
      /* decide if need add to map */
      if (!nearest_points_[i].empty() && flg_EKF_inited_) {
        const PointVector &points_near = nearest_points_[i];
        bool need_add = true;
        BoxPointType Box_of_Point;
        PointType downsample_result, mid_point;
        mid_point.x =
            floor(p_feats_down_world_->points[i].x / filter_size_map_min_) *
                filter_size_map_min_ +
            0.5 * filter_size_map_min_;
        mid_point.y =
            floor(p_feats_down_world_->points[i].y / filter_size_map_min_) *
                filter_size_map_min_ +
            0.5 * filter_size_map_min_;
        mid_point.z =
            floor(p_feats_down_world_->points[i].z / filter_size_map_min_) *
                filter_size_map_min_ +
            0.5 * filter_size_map_min_;
        float dist = calc_dist(p_feats_down_world_->points[i], mid_point);
        if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min_ &&
            fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min_ &&
            fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min_) {
          PointNoNeedDownsample.push_back(p_feats_down_world_->points[i]);
          continue;
        }
        for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
          if (points_near.size() < NUM_MATCH_POINTS)
            break;
          if (calc_dist(points_near[readd_i], mid_point) < dist) {
            need_add = false;
            break;
          }
        }
        if (need_add)
          PointToAdd.push_back(p_feats_down_world_->points[i]);
      } else {
        PointToAdd.push_back(p_feats_down_world_->points[i]);
      }
    }

    // double st_time = omp_get_wtime();
    int add_point_size = ikdtree_.Add_Points(PointToAdd, true);
    ikdtree_.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    // kdtree_incremental_time = omp_get_wtime() - st_time;
  }
  void pointBodyToWorld(PointType const *const pi, PointType *const po) {
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body +
                                     state_point_.offset_T_L_I) +
                 state_point_.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
  }

  bool isKeyFrame() {
    if (cloud_key_poses_3D_->points.empty())
      return true;

    Eigen::Affine3f transStart =
        pclPointToAffine3f(cloud_key_poses_6D_->back());
    Eigen::Affine3f transFinal =
        Eigen::Translation3f(state_point_.pos.cast<float>()) *
        state_point_.rot.matrix().cast<float>();

    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

    if (abs(roll) < 0.2 && abs(pitch) < 0.2 && abs(yaw) < 0.2 &&
        sqrt(x * x + y * y + z * z) < 2) {
      return false;
    }

    return true;
  }

  void addOdomFactor() {
    if (cloud_key_poses_3D_->empty()) {
      gtsam::noiseModel::Diagonal::shared_ptr priorNoise =
          gtsam::noiseModel::Diagonal::Variances(
              (gtsam::Vector(6) << 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8)
                  .finished()); // rad*rad, meter*meter
      gtsam_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(
          0, trans2gtsamPose(state_point_.pos, state_point_.rot), priorNoise));
      initial_estimate_.insert(
          0, trans2gtsamPose(state_point_.pos, state_point_.rot));
    } else {
      gtsam::noiseModel::Diagonal::shared_ptr odometryNoise =
          gtsam::noiseModel::Diagonal::Variances(
              (gtsam::Vector(6) << 1e-5, 1e-5, 1e-5, 1e5, 1e5, 1e5).finished());
      gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloud_key_poses_6D_->back());

      gtsam::Pose3 poseTo = trans2gtsamPose(state_point_.pos, state_point_.rot);
      gtsam_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
          cloud_key_poses_3D_->size() - 1, cloud_key_poses_3D_->size(),
          poseFrom.between(poseTo), odometryNoise));
      initial_estimate_.insert(cloud_key_poses_3D_->size(), poseTo);
    }
  }
  void addLoopFactor() {

    if (loop_index_queue_.empty()) {
      return;
    }

    for (int i = 0; i < (int)loop_index_queue_.size(); ++i) {
      int indexFrom = loop_index_queue_[i].first;
      int indexTo = loop_index_queue_[i].second;
      gtsam::Pose3 poseBetween = loop_pose_queue_[i];
      gtsam::noiseModel::Diagonal::shared_ptr noiseBetween =
          loop_noise_queue_[i];
      gtsam_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
          indexFrom, indexTo, poseBetween, noiseBetween));
    }

    loop_index_queue_.clear();
    loop_pose_queue_.clear();
    loop_noise_queue_.clear();
    aLoopIsClosed = true;
  }
  void loopClosureThread() {
    if (loop_closure_enable_flag_ == false)
      return;

    const auto sleep_duration =
        std::chrono::milliseconds(static_cast<int>(1000.0 / 5));
    while (!exit_flag_) {
      std::this_thread::sleep_for(sleep_duration);
      performLoopClosure();
      if (!visualize_loop_closure_callback_) {
        continue;
      }
      visualize_loop_closure_callback_(loop_index_container_,
                                       *copy_cloud_key_poses_3D_);
    }
  }
  void performLoopClosure() {
    // 1. 关键帧队列为空，直接返回
    if (cloud_key_poses_3D_->points.empty() == true)
      return;

    // 2. 加锁拷贝关键帧3D和6D位姿，避免多线程干扰
    mtx_.lock();
    *copy_cloud_key_poses_3D_ = *cloud_key_poses_3D_;
    *copy_cloud_key_poses_6D_ = *cloud_key_poses_6D_;
    mtx_.unlock();

    // 3.
    // 将最后一帧关键帧作为当前帧，如果当前帧已经在回环对应关系中，则返回（已经处理过这一帧了）。如果找到的回环对应帧相差时间过短也返回false。回环关系用一个全局map缓存
    // 4.
    // 对关键帧3D位姿构建kd树，并用当前帧位置从kd树寻找距离最近的几帧，挑选时间间隔最远的那一帧作为匹配帧
    int loopKeyCur;
    int loopKeyPre;
    if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
      return;

    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(
        new pcl::PointCloud<PointType>());
    {
      // 5. 将当前帧转到Map坐标系并降采样，注意这里第三个参数是0,
      // 也就是不加上前后其他帧
      loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
      // 6. 对匹配帧前后几帧转换到Map坐标系下，融合并降采样，构建局部地图
      loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, 25);
      if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
        return;
    }
    // if (p_map_pub != nullptr) {
    //   sensor_msgs::PointCloud2 laserCloudMap;
    //   pcl::toROSMsg(*prevKeyframeCloud, laserCloudMap);
    //   // laserCloudMap.header.stamp = ros::Time().fromSec(1);
    //   laserCloudMap.header.frame_id = "camera_init";
    //   p_map_pub->publish(laserCloudMap);
    // }
    // if (p_pcl_pub != nullptr) {
    //   sensor_msgs::PointCloud2 laserCloudMap;
    //   pcl::toROSMsg(*cureKeyframeCloud, laserCloudMap);
    //   // laserCloudMap.header.stamp = ros::Time().fromSec(1);
    //   laserCloudMap.header.frame_id = "camera_init";
    //   p_pcl_pub->publish(laserCloudMap);
    // }
    // 2. 移除无效点（包含NaN/Inf的点）
    double t1 = omp_get_wtime();
    // pcl::PointCloud<PointType>::Ptr source_filtered(
    //     new pcl::PointCloud<PointType>());
    // pcl::PointCloud<PointType>::Ptr target_filtered(
    //     new pcl::PointCloud<PointType>());
    // std::vector<int> indices;
    // pcl::removeNaNFromPointCloud(*cureKeyframeCloud, *source_filtered,
    // indices); pcl::removeNaNFromPointCloud(*prevKeyframeCloud,
    // *target_filtered, indices);

    // static pcl::NormalEstimation<PointType, PointType> ne;
    // pcl::search::KdTree<PointType>::Ptr tree(
    //     new pcl::search::KdTree<PointType>());
    // ne.setSearchMethod(tree);
    // ne.setKSearch(20); // 使用20个最近邻计算法向量

    // ne.setInputCloud(source_filtered);
    // ne.compute(*source_filtered);

    // // 计算目标点云法向量
    // ne.setInputCloud(target_filtered);
    // ne.compute(*target_filtered);

    // static pcl::IterativeClosestPointWithNormals<PointType, PointType> icp;

    // // icp.setMaxCorrespondenceDistance(history_keyframe_search_radius_ * 2);
    // icp.setMaxCorrespondenceDistance(history_keyframe_search_radius_ * 2);
    // icp.setMaximumIterations(300);
    // icp.setTransformationEpsilon(1e-8);
    // icp.setEuclideanFitnessEpsilon(1e-8);
    // icp.setRANSACIterations(0);

    // // Align clouds
    // icp.setInputSource(source_filtered);
    // icp.setInputTarget(target_filtered);
    // pcl::PointCloud<PointType>::Ptr unused_result(
    //     new pcl::PointCloud<PointType>());
    // icp.align(*unused_result);
    // // if (p_pcl_pub != nullptr) {
    // //   sensor_msgs::PointCloud2 laserCloudMap;
    // //   pcl::toROSMsg(*unused_result, laserCloudMap);
    // //   // laserCloudMap.header.stamp = ros::Time().fromSec(1);
    // //   laserCloudMap.header.frame_id = "camera_init";
    // //   p_pcl_pub->publish(laserCloudMap);
    // // }
    // std::cout << "分数：" << icp.getFitnessScore() <<"
    // 配准耗时："<<omp_get_wtime()-t1<<std::endl;

    // if (icp.hasConverged() == false ||
    //     icp.getFitnessScore() > history_keyframe_fitness_score_)
    //   return;

    // float x, y, z;
    // Eigen::Quaternionf quat;
    // Eigen::Affine3f correctionLidarFrame;
    // correctionLidarFrame = icp.getFinalTransformation();
    // // transform from world origin to wrong pose
    // Eigen::Affine3f tWrong =
    //     pclPointToAffine3f(copy_cloud_key_poses_6D_->points[loopKeyCur]);
    // // transform from world origin to corrected pose
    // Eigen::Affine3f tCorrect =
    //     correctionLidarFrame *
    //     tWrong; // pre-multiplying -> successive rotation about a fixed frame
    // x = tCorrect.translation().x();
    // y = tCorrect.translation().y();
    // z = tCorrect.translation().z();
    // quat = Eigen::Quaternionf(tCorrect.rotation());

    // gtsam::Pose3 poseFrom = gtsam::Pose3(
    //     gtsam::Rot3::Quaternion(quat.w(), quat.x(), quat.y(), quat.z()),
    //     gtsam::Point3(x, y, z));
    // gtsam::Pose3 poseTo =
    //     pclPointTogtsamPose3(copy_cloud_key_poses_6D_->points[loopKeyPre]);
    // gtsam::Vector Vector6(6);
    // float noiseScore = 1e-6;
    // Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore,
    //     noiseScore;
    // gtsam::noiseModel::Diagonal::shared_ptr constraintNoise =
    //     gtsam::noiseModel::Diagonal::Variances(Vector6);

    pcl::PointCloud<PointType>::Ptr source_cloud(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr target_cloud(
        new pcl::PointCloud<PointType>());


    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cureKeyframeCloud, *source_cloud, indices);
    pcl::removeNaNFromPointCloud(*prevKeyframeCloud, *target_cloud, indices);

 
    std::vector<Eigen::Vector4f> eigen_points_source;
    std::vector<Eigen::Vector4f> eigen_points_target;
    eigen_points_source.reserve(source_cloud->size());
    eigen_points_target.reserve(target_cloud->size());

    for (const auto &point : *source_cloud) {
      // 过滤NaN点（如果开启）
      if ((std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z))) {
        continue;
      }
      // 转换为齐次坐标（w=1.0）
      Eigen::Vector4f eigen_point;
      eigen_point << point.x, point.y, point.z, 1.0f;
      eigen_points_source.push_back(eigen_point);
    }
    for (const auto &point : *target_cloud) {
      // 过滤NaN点（如果开启）
      if ((std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z))) {
        continue;
      }

      // 转换为齐次坐标（w=1.0）
      Eigen::Vector4f eigen_point;
      eigen_point << point.x, point.y, point.z, 1.0f;
      eigen_points_target.push_back(eigen_point);
    }

    small_gicp::RegistrationSetting setting;
    setting.num_threads = 1;               // Number of threads to be used
    setting.downsampling_resolution = 0.25; // Downsampling resolution
    setting.max_correspondence_distance = 3.0;
    // setting.type=small_gicp::RegistrationSetting::RegistrationType::ICP;

    Eigen::Isometry3d init_T_target_source = Eigen::Isometry3d::Identity();
    small_gicp::RegistrationResult result =
        align(eigen_points_target, eigen_points_source, init_T_target_source,
              setting);

    // std::cout << "--- T_target_source ---" << std::endl
    //           << result.T_target_source.matrix() << std::endl;
    // std::cout << "converged:" << result.converged << std::endl;
    // std::cout << "error:" << result.error << std::endl;
    // std::cout << "iterations:" << result.iterations << std::endl;
    // std::cout << "num_inliers:" << result.num_inliers << std::endl;
    // std::cout << "--- H ---" << std::endl << result.H << std::endl;
    // std::cout << "--- b ---" << std::endl << result.b.transpose() << std::endl;

    // 计算适应度分数（这里使用平均误差作为参考）
    auto fitness_score = result.error / std::max(1, (int)result.num_inliers);
    std::cout << "适应度分数：" << fitness_score << " 配准耗时："
              << omp_get_wtime() - t1 << std::endl;

    // 检查配准是否成功
    if (!result.converged || fitness_score > history_keyframe_fitness_score_)
      return;

    // 8. 处理配准结果，获取修正后的位姿
    Eigen::Matrix4f correctionMatrix =
        result.T_target_source.matrix().cast<float>();
    Eigen::Affine3f correctionLidarFrame(correctionMatrix);

    // 以下部分保持不变
    Eigen::Affine3f tWrong =
        pclPointToAffine3f(copy_cloud_key_poses_6D_->points[loopKeyCur]);
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;

    float x = tCorrect.translation().x();
    float y = tCorrect.translation().y();
    float z = tCorrect.translation().z();
    Eigen::Quaternionf quat(tCorrect.rotation());

    gtsam::Pose3 poseFrom = gtsam::Pose3(
        gtsam::Rot3::Quaternion(quat.w(), quat.x(), quat.y(), quat.z()),
        gtsam::Point3(x, y, z));
    gtsam::Pose3 poseTo =
        pclPointTogtsamPose3(copy_cloud_key_poses_6D_->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    float noiseScore = 1e-6;
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore,
        noiseScore;
    gtsam::noiseModel::Diagonal::shared_ptr constraintNoise =
        gtsam::noiseModel::Diagonal::Variances(Vector6);

    // 9. 将回环索引、回环间相对位姿、回环噪声模型加入全局变量
    mtx_.lock();
    loop_index_queue_.push_back(make_pair(loopKeyCur, loopKeyPre));
    loop_pose_queue_.push_back(poseFrom.between(poseTo));
    loop_noise_queue_.push_back(constraintNoise);
    mtx_.unlock();
    loop_index_container_[loopKeyCur] = loopKeyPre;
  }

  bool detectLoopClosureDistance(int *latestID, int *closestID) {
    int loopKeyCur = copy_cloud_key_poses_3D_->size() - 1;
    int loopKeyPre = -1;

    // 确认最后一帧关键帧没有被加入过回环关系中
    auto it = loop_index_container_.find(loopKeyCur);
    if (it != loop_index_container_.end())
      return false;

    // 将关键帧的3D位置构建kdtree，并检索空间位置相近的关键帧
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtree_history_key_poses_->setInputCloud(copy_cloud_key_poses_3D_);
    // 寻找空间距离相近的关键帧
    kdtree_history_key_poses_->radiusSearch(
        copy_cloud_key_poses_3D_->back(), history_keyframe_search_radius_,
        pointSearchIndLoop, pointSearchSqDisLoop, 0);

    // 确保空间距离相近的帧是较久前采集的，排除是前面几个关键帧
    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i) {
      int id = pointSearchIndLoop[i];
      if (abs(copy_cloud_key_poses_6D_->points[id].time - lidar_time_) >
          history_keyframe_search_time_diff_) {
        // 行程小于100m的跳过
        double s = 0;
        for (int i = id; i < loopKeyCur; i++) {
          s += std::sqrt(std::pow((cloud_key_poses_3D_->points[id + 1].x -
                                   cloud_key_poses_3D_->points[id].x),
                                  2) +
                         std::pow((cloud_key_poses_3D_->points[id + 1].y -
                                   cloud_key_poses_3D_->points[id].y),
                                  2) +
                         std::pow((cloud_key_poses_3D_->points[id + 1].z -
                                   cloud_key_poses_3D_->points[id].z),
                                  2)

          );
        }
        if (s >= 100) {
          loopKeyPre = id;
          break;
        }
      }
    }

    // 如果没有找到位置关系、时间关系都符合要求的关键帧，则返回false
    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
      return false;

    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
  }
  void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes,
                             const int &key, const int &searchNum) {
    // extract near keyframes
    nearKeyframes->clear();
    int cloudSize = copy_cloud_key_poses_6D_->size();
    for (int i = -searchNum; i <= searchNum; ++i) {
      int keyNear = key + i;
      if (keyNear < 0 || keyNear >= cloudSize)
        continue;
      *nearKeyframes +=
          *transformPointCloud(cloud_key_frames_[keyNear],
                               &copy_cloud_key_poses_6D_->points[keyNear]);
    }

    if (nearKeyframes->empty())
      return;

    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(
        new pcl::PointCloud<PointType>());
    down_size_filter_icp_.setInputCloud(nearKeyframes);
    down_size_filter_icp_.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
  }

  void saveKeyFramesAndFactor() {
    // 是否将当前帧采纳为关键帧
    // 如果距离（小于1米）和角度同时不符合要求，不采纳为关键帧
    if (isKeyFrame() == false)
      return;

    // 添加激光里程计因子
    addOdomFactor();

    // 添加回环因子
    addLoopFactor();

    // 迭代一次优化器
    isam->update(gtsam_graph_, initial_estimate_);

    // 如果当前帧有新的GPS因子或者回环因子加入，执行多次迭代更新，且后面会更新所有历史帧位姿
    if (aLoopIsClosed == true) {
      isam->update();
      isam->update();
      isam->update();
      isam->update();
      isam->update();
      isam->update();
    }
    isamCurrentEstimate = isam->calculateEstimate();

    // 清空因子图和初始值（标准做法），因子已经加入了优化器
    gtsam_graph_.resize(0);
    initial_estimate_.clear();

    // 从优化器拿出当前帧的位姿存入关键帧位姿序列
    PointType thisPose3D;
    PointTypePose thisPose6D;
    gtsam::Pose3 latestEstimate;

    // 从优化器中拿出最近一帧的优化结果

    latestEstimate =
        isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size() - 1);
    // cout << "****************************************************" << endl;
    // isamCurrentEstimate.print("Current estimate: ");
    if (aLoopIsClosed != true) {
      // 弱国没有回环，用ekf的结果，优化的结果会精度不如ekf的高
      latestEstimate = trans2gtsamPose(state_point_.pos, state_point_.rot);
    }

    // 将当前帧经过优化的结果存入关键帧位姿序列（3D/6D）
    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity =
        cloud_key_poses_3D_->size(); // this can be used as index
    cloud_key_poses_3D_->push_back(thisPose3D);

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity; // this can be used as index
    thisPose6D.roll = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw = latestEstimate.rotation().yaw();
    thisPose6D.time = lidar_time_;
    cloud_key_poses_6D_->push_back(thisPose6D);

    if (aLoopIsClosed == true) {
      state_ikfom state_updated = kf_.getState();
      Eigen::Vector3d pos(latestEstimate.translation().x(),
                          latestEstimate.translation().y(),
                          latestEstimate.translation().z());
      Eigen::Quaterniond q = EulerToQuat(latestEstimate.rotation().roll(),
                                         latestEstimate.rotation().pitch(),
                                         latestEstimate.rotation().yaw());

      //  更新状态量
      state_updated.pos = pos;
      state_updated.rot = Sophus::SO3d(q);
      state_point_ = state_updated;
      kf_.setState(state_updated);
    }

    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(
        new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*p_feats_undistort_,
                        *thisSurfKeyFrame); // 存储关键帧,没有降采样的点云

    cloud_key_frames_.push_back(thisSurfKeyFrame);

    updatePath(thisPose6D);
  }
  void correctPoses() {
    if (cloud_key_poses_3D_->points.empty()) {
      return;
    }

    if (aLoopIsClosed == true) {
      global_path_.clear();
      int numPoses = isamCurrentEstimate.size();
      for (int i = 0; i < numPoses; ++i) {
        cloud_key_poses_3D_->points[i].x =
            isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
        cloud_key_poses_3D_->points[i].y =
            isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
        cloud_key_poses_3D_->points[i].z =
            isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();

        cloud_key_poses_6D_->points[i].x = cloud_key_poses_3D_->points[i].x;
        cloud_key_poses_6D_->points[i].y = cloud_key_poses_3D_->points[i].y;
        cloud_key_poses_6D_->points[i].z = cloud_key_poses_3D_->points[i].z;
        cloud_key_poses_6D_->points[i].roll =
            isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
        cloud_key_poses_6D_->points[i].pitch =
            isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
        cloud_key_poses_6D_->points[i].yaw =
            isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

        // 更新里程计轨迹
        updatePath(cloud_key_poses_6D_->points[i]);
      }
      // 清空局部map， 重新构建  ikdtree
      for (int i = 0; i < feats_down_size_; i++) {
        pointBodyToWorld(&(p_feats_down_body_->points[i]),
                         &(p_feats_down_world_->points[i]));
      }

      ikdtree_.Build(p_feats_down_world_->points);
      aLoopIsClosed = false;
    }
  }
  void updatePath(const PointTypePose &pose_in) {
    PathPose pose_stamp;
    pose_stamp.time = pose_in.time * 1e9;
    pose_stamp.frame_id = "camera_init";
    pose_stamp.position_x = pose_in.x;
    pose_stamp.position_y = pose_in.y;
    pose_stamp.position_z = pose_in.z;
    Eigen::Quaternionf q;
    pose_stamp.q = Eigen::AngleAxisf(pose_in.yaw, Eigen::Vector3f::UnitZ()) *
                   Eigen::AngleAxisf(pose_in.pitch, Eigen::Vector3f::UnitY()) *
                   Eigen::AngleAxisf(pose_in.roll, Eigen::Vector3f::UnitX());
    global_path_.push_back(pose_stamp);
  }

  void visualizeGlobalMapThread() {
    const auto sleep_duration =
        std::chrono::milliseconds(static_cast<int>(1000.0 / 5));
    if (!visualize_global_map_callback_) {
      return;
    }
    while (!exit_flag_) {
      std::this_thread::sleep_for(sleep_duration);
      visualize_global_map_callback_(getGlobalMap());
    }
  }
};
} // namespace simple_fast_lio2_lc
