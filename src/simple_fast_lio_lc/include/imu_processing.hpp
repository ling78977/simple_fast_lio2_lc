#ifndef SIMPLE_FAST_LIO2_LC_IMU_PRECESSING_HPP
#define SIMPLE_FAST_LIO2_LC_IMU_PRECESSING_HPP
#include "common.hpp"
#include "esekf.hpp"
#include "sophus/so3.hpp"
#include <omp.h>
#include <sensor_msgs/Imu.h>

#define MAX_INI_COUNT (10)
inline bool time_list(PointType &x, PointType &y) {
  return (x.curvature < y.curvature);
};
namespace simple_fast_lio2_lc {

class ImuProcess {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess()
      : start_timestamp_(-1), is_first_frame_(true), imu_need_init_(true) {
    init_iter_num_ = 1; // 初始化迭代次数
    Q_ = process_noise_cov();
    cov_acc_ = V3D(0.1, 0.1, 0.1);               // 加速度协方差初始化
    cov_gyr_ = V3D(0.1, 0.1, 0.1);               // 角速度协方差初始化
    cov_bias_gyr_ = V3D(0.0001, 0.0001, 0.0001); // 角速度bias协方差初始化
    cov_bias_acc_ = V3D(0.0001, 0.0001, 0.0001); // 加速度bias协方差初始化
    mean_acc_ = V3D(0, 0, -1.0);
    mean_gyr_ = V3D(0, 0, 0);
    angvel_last_ = Zero3d;                   // 上一帧角速度初始化
    lidar_T_wrt_imu_ = Zero3d;               // lidar到IMU的位置外参初始化
    lidar_R_wrt_imu_ = Eye3d;                // lidar到IMU的旋转外参初始化
    last_imu_.reset(new sensor_msgs::Imu()); // 上一帧imu初始化
  }
  ~ImuProcess() {}
  void reset() {
    mean_acc_ = V3D(0, 0, -1.0);
    mean_gyr_ = V3D(0, 0, 0);
    angvel_last_ = Zero3d;
    imu_need_init_ = true;                   // 是否需要初始化imu
    start_timestamp_ = -1;                   // 开始时间戳
    init_iter_num_ = 1;                      // 初始化迭代次数
    imu_pose_.clear();                       // imu位姿清空
    last_imu_.reset(new sensor_msgs::Imu()); // 上一帧imu初始化
    // cur_pcl_un_.reset(new PointCloudXYZI());      // 当前帧点云未去畸变初始化
  }
  void setParam(const V3D &_transl, const M3D &_rot, const V3D &_gyr,
                const V3D &_acc, const V3D &_gyr_bias, const V3D &_acc_bias) {
    lidar_T_wrt_imu_ = _transl;
    lidar_R_wrt_imu_ = _rot;
    cov_gyr_scale_ = _gyr;
    cov_acc_scale_ = _acc;
    cov_bias_gyr_ = _gyr_bias;
    cov_bias_acc_ = _acc_bias;
  }
  void process(const MeasureGroup &_meas, ESEKalmanFilter &_kf_state,
               PointCloudXYZI::Ptr &_cur_pcl_un) {
    // double T1 = omp_get_wtime();

    if (_meas.imu.empty()) {
      return;
    };
    assert(_meas.lidar != nullptr);

    if (imu_need_init_) {
      // The very first lidar frame
      imuInit(_meas, _kf_state,
              init_iter_num_); // 如果开头几帧 需要初始化IMU参数

      imu_need_init_ = true;

      last_imu_ = _meas.imu.back();

      if (init_iter_num_ > MAX_INI_COUNT) {
        cov_acc_ *= pow(G_m_s2 / mean_acc_.norm(), 2);
        imu_need_init_ = false;
        cov_acc_ = cov_acc_scale_;
        cov_gyr_ = cov_gyr_scale_;
        std::cout << "IMU Initial Done" << std::endl;
      }
      return;
    }
    // double T1 = omp_get_wtime();
    undistortPcl(_meas, _kf_state, *_cur_pcl_un);

    // int T2 = omp_get_wtime();
    // std::cout << "[ IMU Process ]: Time: " << T2 - T1 << std::endl;
  }
  Eigen::Matrix<double, 12, 12> Q_; // 噪声协方差矩阵  对应论文式(8)中的Q
  V3D cov_acc_;                     // 加速度协方差
  V3D cov_gyr_;                     // 角速度协方差
  V3D cov_acc_scale_;               // 外部传入的 初始加速度协方差
  V3D cov_gyr_scale_;               // 外部传入的 初始角速度协方差
  V3D cov_bias_gyr_;                // 角速度bias的协方差
  V3D cov_bias_acc_;                // 加速度bias的协方差
  double first_lidar_time_;         // 当前帧第一个点云时间

private:
  void imuInit(const MeasureGroup &_meas, ESEKalmanFilter &_kf_state, int &_N) {
    // MeasureGroup这个struct表示当前过程中正在处理的所有数据，包含IMU队列和一帧lidar的点云
    // 以及lidar的起始和结束时间 初始化重力、陀螺仪偏差、acc和陀螺仪协方差
    // 将加速度测量值归一化为单位重力   **/
    V3D cur_acc, cur_gyr;

    if (is_first_frame_) // 如果为第一帧IMU
    {
      reset(); // 重置IMU参数
      _N = 1;  // 将迭代次数置1
      is_first_frame_ = false;
      // IMU初始时刻的加速度
      const auto &imu_acc = _meas.imu.front()->linear_acceleration;
      // IMU初始时刻的角速度
      const auto &gyr_acc = _meas.imu.front()->angular_velocity;
      // 第一帧加速度值作为初始化均值
      mean_acc_ << imu_acc.x, imu_acc.y, imu_acc.z;
      // 第一帧角速度值作为初始化均值
      mean_gyr_ << gyr_acc.x, gyr_acc.y, gyr_acc.z;
      // 将当前IMU帧对应的lidar起始时间 作为初始时间
      first_lidar_time_ = _meas.lidar_beg_time;
    }

    for (const auto &imu : _meas.imu) // 根据所有IMU数据，计算平均值和方差
    {
      const auto &imu_acc = imu->linear_acceleration;
      const auto &gyr_acc = imu->angular_velocity;
      cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
      cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

      mean_acc_ +=
          (cur_acc - mean_acc_) / _N; // 根据当前帧和均值差作为均值的更新
      mean_gyr_ += (cur_gyr - mean_gyr_) / _N;

      cov_acc_ = cov_acc_ * (_N - 1.0) / _N +
                 (cur_acc - mean_acc_).cwiseProduct(cur_acc - mean_acc_) / _N;
      cov_gyr_ = cov_gyr_ * (_N - 1.0) / _N +
                 (cur_gyr - mean_gyr_).cwiseProduct(cur_gyr - mean_gyr_) / _N /
                     _N * (_N - 1);

      _N++;
    }

    state_ikfom init_state = _kf_state.getState(); // 在esekfom.hpp获得x_的状态
    init_state.grav = -mean_acc_ / mean_acc_.norm() *
                      G_m_s2; // 得平均测量的单位方向向量 * 重力加速度预设值

    init_state.bg = mean_gyr_;                  // 角速度测量作为陀螺仪偏差
    init_state.offset_T_L_I = lidar_T_wrt_imu_; // 将lidar和imu外参传入
    init_state.offset_R_L_I = Sophus::SO3d(lidar_R_wrt_imu_);
    _kf_state.setState(init_state);

    Eigen::Matrix<double, 24, 24> init_P = Eigen::MatrixXd::Identity(24, 24);
    init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
    init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
    init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
    init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
    init_P(21, 21) = init_P(22, 22) = init_P(23, 23) = 0.00001;
    _kf_state.setP(init_P);
    last_imu_ = _meas.imu.back();
  }
  void undistortPcl(const MeasureGroup &_meas, ESEKalmanFilter &_kf_state,
                    PointCloudXYZI &_pcl_out) {
    /*** add the imu of the last frame-tail to the of current frame-head ***/
    auto v_imu = _meas.imu;
    v_imu.push_front(last_imu_);
    const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
    const double &imu_end_time = v_imu.back()->header.stamp.toSec();

    double pcl_beg_time = _meas.lidar_beg_time;
    double pcl_end_time = _meas.lidar_end_time;

    /*** sort point clouds by offset time ***/
    _pcl_out = *(_meas.lidar);
    sort(_pcl_out.points.begin(), _pcl_out.points.end(), time_list);
    // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

    /*** Initialize IMU pose ***/
    state_ikfom imu_state = _kf_state.getState();
    imu_pose_.clear();
    imu_pose_.push_back(
        set_pose6d(0.0, acc_s_last_, angvel_last_, imu_state.vel, imu_state.pos,
                   imu_state.rot.unit_quaternion().toRotationMatrix()));

    /*** forward propagation at each imu point ***/
    V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
    M3D R_imu;

    double dt = 0;

    input_ikfom in;
    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
      auto &&head = *(it_imu);
      auto &&tail = *(it_imu + 1);

      if (tail->header.stamp.toSec() < last_lidar_end_time_)
        continue;

      angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
          0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
          0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
      acc_avr << 0.5 * (head->linear_acceleration.x +
                        tail->linear_acceleration.x),
          0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
          0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

      // fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time
      // << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;

      acc_avr = acc_avr * G_m_s2 / mean_acc_.norm(); // - state_inout.ba;

      if (head->header.stamp.toSec() < last_lidar_end_time_) {
        dt = tail->header.stamp.toSec() - last_lidar_end_time_;
        // dt = tail->header.stamp.toSec() - pcl_beg_time;
      } else {
        dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
      }

      in.acc = acc_avr;
      in.gyro = angvel_avr;
      Q_.block<3, 3>(0, 0).diagonal() = cov_gyr_;
      Q_.block<3, 3>(3, 3).diagonal() = cov_acc_;
      Q_.block<3, 3>(6, 6).diagonal() = cov_bias_gyr_;
      Q_.block<3, 3>(9, 9).diagonal() = cov_bias_acc_;
      _kf_state.predict(dt, Q_, in);

      /* save the poses at each IMU measurements */
      imu_state = _kf_state.getState();
      angvel_last_ = angvel_avr - imu_state.bg;
      acc_s_last_ = imu_state.rot * (acc_avr - imu_state.ba);
      for (int i = 0; i < 3; i++) {
        acc_s_last_[i] += imu_state.grav[i];
      }
      double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
      imu_pose_.push_back(set_pose6d(
          offs_t, acc_s_last_, angvel_last_, imu_state.vel, imu_state.pos,
          imu_state.rot.unit_quaternion().toRotationMatrix()));
    }

    /*** calculated the pos and attitude prediction at the frame-end ***/
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    _kf_state.predict(dt, Q_, in);

    imu_state = _kf_state.getState();
    last_imu_ = _meas.imu.back();
    last_lidar_end_time_ = pcl_end_time;

    /*** undistort each lidar point (backward propagation) ***/
    if (_pcl_out.points.begin() == _pcl_out.points.end())
      return;

    auto it_pcl = _pcl_out.points.end() - 1;
    for (auto it_kp = imu_pose_.end() - 1; it_kp != imu_pose_.begin(); it_kp--) {
      auto head = it_kp - 1;
      auto tail = it_kp;
      R_imu << MAT_FROM_ARRAY(head->rot);
      // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
      vel_imu << VEC_FROM_ARRAY(head->vel);
      pos_imu << VEC_FROM_ARRAY(head->pos);
      acc_imu << VEC_FROM_ARRAY(tail->acc);
      angvel_avr << VEC_FROM_ARRAY(tail->gyr);

      for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
        dt = it_pcl->curvature / double(1000) - head->offset_time;

        /* Transform to the 'end' frame, using only the rotation
         * Note: Compensation direction is INVERSE of Frame's moving direction
         * So if we want to compensate a point at timestamp-i to the frame-e
         * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is
         * represented in global frame */
        M3D R_i(R_imu *  Sophus::SO3d::exp(angvel_avr*dt).matrix());

        V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
        V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt -
                 imu_state.pos);
        V3D P_compensate =
            imu_state.offset_R_L_I.unit_quaternion().conjugate() *
            (imu_state.rot.unit_quaternion().conjugate() * (R_i * (imu_state.offset_R_L_I * P_i +
                                                 imu_state.offset_T_L_I) +
                                          T_ei) -
             imu_state.offset_T_L_I); // not accurate!

        // save Undistorted points and their rotation
        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);

        if (it_pcl == _pcl_out.points.begin())
          break;
      }
    }
  }

  // PointCloudXYZI::Ptr cur_pcl_un_;                 // 当前帧点云未去畸变
  sensor_msgs::Imu::Ptr last_imu_; // 上一帧imu
  std::vector<Pose6D> imu_pose_;   // 存储imu位姿(反向传播用)
  M3D lidar_R_wrt_imu_;            // lidar到IMU的旋转外参
  V3D lidar_T_wrt_imu_;            // lidar到IMU的平移外参
  V3D mean_acc_;                   // 加速度均值,用于计算方差
  V3D mean_gyr_;                   // 角速度均值，用于计算方差
  V3D angvel_last_;                // 上一帧角速度
  V3D acc_s_last_;                 // 上一帧加速度
  double start_timestamp_;         // 开始时间戳
  double last_lidar_end_time_;     // 上一帧结束时间戳
  int init_iter_num_ = 1;          // 初始化迭代次数
  bool is_first_frame_ = true;     // 是否是第一帧
  bool imu_need_init_ = true;      // 是否需要初始化imu
};

} // namespace simple_fast_lio2_lc

#endif