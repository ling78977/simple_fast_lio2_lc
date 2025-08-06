#ifndef SIMPLE_FAST_LIO2_COMMON_HPP
#define SIMPLE_FAST_LIO2_COMMON_HPP

// #include "simple_fast_lio2_lc/msg/detail/pose6_d__struct.hpp"
#include "sophus/so3.hpp"
#include <Eigen/Eigen>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <cmath>
#include <deque>
#include <gtsam/geometry/Pose3.h>
#include <nav_msgs/Odometry.h>
#include <pcl/common/eigen.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/time.h>
#include <sensor_msgs/Imu.h>
#include <vector>

typedef struct {
  float offset_time;
  float acc[3];
  float gyr[3];
  float vel[3];
  float pos[3];
  float rot[9];
} Pose6D;

typedef struct{
double time;
std::string frame_id;
float position_x;
float position_y;
float position_z;
Eigen::Quaternionf q;
}PathPose;

#define VF(a) Eigen::Matrix<float, (a), 1>

#define PI_M (3.14159265358)
#define G_m_s2 (9.82)   // Gravaty const in GuangDong/China
#define DIM_STATE (18)  // Dimension of states (Let Dim(SO(3)) = 3)
#define DIM_PROC_N (12) // Dimension of process noise (Let Dim(SO(3)) = 3)
#define CUBE_LEN (6.0)
#define LIDAR_SP_LEN (2)
#define INIT_COV (1)
#define NUM_MATCH_POINTS (5)
#define MAX_MEAS_DIM (10000)

#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
#define SKEW_SYM_MATRX(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0
#define DEBUG_FILE_DIR(name) (string(string(ROOT_DIR) + "Log/" + name))

// typedef simple_fast_lio2_lc::msg::Pose6D Pose6D;
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
typedef std::vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;
typedef Eigen::Vector3d V3D;
typedef Eigen::Matrix3d M3D;
typedef Eigen::Vector3f V3F;
typedef Eigen::Matrix3f M3F;

struct PointXYZIRPYT {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY; // preferred way of adding a XYZ+padding
  float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16; // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIRPYT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
        float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time,
                                                                 time))

typedef PointXYZIRPYT PointTypePose;

inline M3D Eye3d(M3D::Identity());
inline M3F Eye3f(M3F::Identity());
inline V3D Zero3d(0, 0, 0);
inline V3F Zero3f(0, 0, 0);

struct MeasureGroup // Lidar data and imu dates for the current process
{
  MeasureGroup() {
    lidar_beg_time = 0.0;
    this->lidar.reset(new PointCloudXYZI());
  };
  double lidar_beg_time;
  double lidar_end_time;
  PointCloudXYZI::Ptr lidar;
  std::deque<sensor_msgs::Imu::Ptr> imu;
};

inline Eigen::Matrix<double, 12, 12> process_noise_cov() {
  Eigen::Matrix<double, 12, 12> Q = Eigen::MatrixXd::Zero(12, 12);
  Q.block<3, 3>(0, 0) = 0.0001 * Eigen::Matrix3d::Identity();
  Q.block<3, 3>(3, 3) = 0.0001 * Eigen::Matrix3d::Identity();
  Q.block<3, 3>(6, 6) = 0.00001 * Eigen::Matrix3d::Identity();
  Q.block<3, 3>(9, 9) = 0.00001 * Eigen::Matrix3d::Identity();

  return Q;
}

inline double get_time_sec(const ros::Time &time) { return time.toSec(); }

template <typename T>
auto set_pose6d(const double t, const Eigen::Matrix<T, 3, 1> &a,
                const Eigen::Matrix<T, 3, 1> &g,
                const Eigen::Matrix<T, 3, 1> &v,
                const Eigen::Matrix<T, 3, 1> &p,
                const Eigen::Matrix<T, 3, 3> &R) {
  Pose6D rot_kp;
  rot_kp.offset_time = t;
  for (int i = 0; i < 3; i++) {
    rot_kp.acc[i] = a(i);
    rot_kp.gyr[i] = g(i);
    rot_kp.vel[i] = v(i);
    rot_kp.pos[i] = p(i);
    for (int j = 0; j < 3; j++)
      rot_kp.rot[i * 3 + j] = R(i, j);
  }
  return rot_kp;
}

inline float calc_dist(PointType p1, PointType p2) {
  float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) +
            (p1.z - p2.z) * (p1.z - p2.z);
  return d;
}

template <typename T>
bool esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point,
                const T &threshold) {
  Eigen::Matrix<T, NUM_MATCH_POINTS, 3> A;
  Eigen::Matrix<T, NUM_MATCH_POINTS, 1> b;
  A.setZero();
  b.setOnes();
  b *= -1.0f;

  // 求A/Dx + B/Dy + C/Dz + 1 = 0 的参数
  for (int j = 0; j < NUM_MATCH_POINTS; j++) {
    A(j, 0) = point[j].x;
    A(j, 1) = point[j].y;
    A(j, 2) = point[j].z;
  }

  Eigen::Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

  T n = normvec.norm();
  // pca_result是平面方程的4个参数  /n是为了归一化
  pca_result(0) = normvec(0) / n;
  pca_result(1) = normvec(1) / n;
  pca_result(2) = normvec(2) / n;
  pca_result(3) = 1.0 / n;

  // 如果几个点中有距离该平面>threshold的点 认为是不好的平面 返回false
  for (int j = 0; j < NUM_MATCH_POINTS; j++) {
    if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y +
             pca_result(2) * point[j].z + pca_result(3)) > threshold) {
      return false;
    }
  }
  return true;
}

inline ros::Time get_ros_time(double timestamp) {
  int32_t sec = static_cast<int32_t>(std::floor(timestamp));
  uint32_t nsec = static_cast<uint32_t>((timestamp - sec) * 1e9);
  return ros::Time(sec, nsec);
}

inline Eigen::Matrix<double, 3, 3> get_A_matrix(Eigen::Vector3d &v) {
  Eigen::Matrix<double, 3, 3> res;
  double squaredNorm = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  double norm = std::sqrt(squaredNorm);
  if (norm < 1e-11) {
    res = Eigen::Matrix<double, 3, 3>::Identity();
  } else {
    res = Eigen::Matrix<double, 3, 3>::Identity() +
          (1 - std::cos(norm)) / squaredNorm * Sophus::SO3d::hat(v) +
          (1 - std::sin(norm) / norm) / squaredNorm * Sophus::SO3d::hat(v) *
              Sophus::SO3d::hat(v);
  }
  return res;
}
inline float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

inline Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint) {

  return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z,
                                thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}
inline pcl::PointCloud<PointType>::Ptr
transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn,
                    PointTypePose *transformIn) {
  pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

  int cloudSize = cloudIn->size();
  cloudOut->resize(cloudSize);

  Eigen::Affine3f transCur = pcl::getTransformation(
      transformIn->x, transformIn->y, transformIn->z, transformIn->roll,
      transformIn->pitch, transformIn->yaw);

#pragma omp parallel for num_threads(2)
  for (int i = 0; i < cloudSize; ++i) {
    const auto &pointFrom = cloudIn->points[i];
    cloudOut->points[i].x = transCur(0, 0) * pointFrom.x +
                            transCur(0, 1) * pointFrom.y +
                            transCur(0, 2) * pointFrom.z + transCur(0, 3);
    cloudOut->points[i].y = transCur(1, 0) * pointFrom.x +
                            transCur(1, 1) * pointFrom.y +
                            transCur(1, 2) * pointFrom.z + transCur(1, 3);
    cloudOut->points[i].z = transCur(2, 0) * pointFrom.x +
                            transCur(2, 1) * pointFrom.y +
                            transCur(2, 2) * pointFrom.z + transCur(2, 3);
    cloudOut->points[i].intensity = pointFrom.intensity;
  }
  return cloudOut;
}
inline void quaternion_to_rpy_eigen(double x, double y, double z, double w,
                                    float &roll, float &pitch, float &yaw) {
  // 创建四元数对象（注意Eigen构造函数参数顺序为w,x,y,z）
  Eigen::Quaterniond quat(w, x, y, z);

  // 转换为旋转矩阵
  Eigen::Matrix3d rot = quat.toRotationMatrix();

  // 计算欧拉角（XYZ顺序，即Roll-Pitch-Yaw）
  Eigen::Vector3d euler = rot.eulerAngles(0, 1, 2);
  roll = euler[0];  // 绕X轴旋转
  pitch = euler[1]; // 绕Y轴旋转
  yaw = euler[2];   // 绕Z轴旋转
}
inline gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint) {
  return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll),
                                          double(thisPoint.pitch),
                                          double(thisPoint.yaw)),
                      gtsam::Point3(double(thisPoint.x), double(thisPoint.y),
                                    double(thisPoint.z)));
}
inline gtsam::Pose3 trans2gtsamPose(Eigen::Vector3d pos, Sophus::SO3d rot) {
  return gtsam::Pose3(gtsam::Rot3::Quaternion(
                          rot.unit_quaternion().w(), rot.unit_quaternion().x(),
                          rot.unit_quaternion().y(), rot.unit_quaternion().z()),
                      gtsam::Point3(pos.x(), pos.y(), pos.z()));
}

//  eulerAngle 2 Quaterniond
inline Eigen::Quaterniond  EulerToQuat(float roll_, float pitch_, float yaw_)
{
    Eigen::Quaterniond q ;            //   四元数 q 和 -q 是相等的
    Eigen::AngleAxisd roll(double(roll_), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch(double(pitch_), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw(double(yaw_), Eigen::Vector3d::UnitZ());
    q = yaw * pitch * roll ;
    q.normalize();
    return q ;
}
#endif