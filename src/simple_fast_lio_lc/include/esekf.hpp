#ifndef SIMPLE_FAST_LIO_ESEKF_HPP
#define SIMPLE_FAST_LIO_ESEKF_HPP

#include "common.hpp"
#include "sophus/so3.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/src/Core/Matrix.h>
#include <boost/bind.hpp>
#include <cstdlib>
#include <functional>
#include <omp.h>

namespace simple_fast_lio2_lc {
// 24维的状态量x
struct state_ikfom {
  // delta_p, delta_theta, delta_RLI, delta_TLI, delta_v, delta_bg, delta_ba,
  // delta_g
  Eigen::Vector3d pos = Eigen::Vector3d(0, 0, 0);
  Sophus::SO3d rot = Sophus::SO3d();
  Sophus::SO3d offset_R_L_I = Sophus::SO3d();
  Eigen::Vector3d offset_T_L_I = Eigen::Vector3d(0, 0, 0);
  Eigen::Vector3d vel = Eigen::Vector3d(0, 0, 0);
  Eigen::Vector3d bg = Eigen::Vector3d(0, 0, 0);
  Eigen::Vector3d ba = Eigen::Vector3d(0, 0, 0);
  Eigen::Vector3d grav = Eigen::Vector3d(0, 0, -G_m_s2);
};
// 输入u
struct input_ikfom {
  Eigen::Vector3d acc = Eigen::Vector3d(0, 0, 0);
  Eigen::Vector3d gyro = Eigen::Vector3d(0, 0, 0);
};
struct dyn_share_datastruct {
  bool valid;
  bool converge;
  Eigen::Matrix<double, Eigen::Dynamic, 1> z;
  Eigen::Matrix<double, Eigen::Dynamic, 1> h;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_v;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> jacbi_H;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> R;
};

class ESEKalmanFilter {
public:
  ESEKalmanFilter(const state_ikfom &x = state_ikfom(),
                  const Eigen::Matrix<double, 24, 24> &P =
                      Eigen::Matrix<double, 24, 24>::Identity()) {
    this->x_ = x;
    this->P_ = P;
  }
  void
  init(const std::function<state_ikfom(const double &, const state_ikfom &,
                                       const input_ikfom &)> &_f,
       const std::function<Eigen::Matrix<double, 24, 24>(
           const double &, const state_ikfom &, const input_ikfom &)> &_jacbi_f,
       const std::function<Eigen::Matrix<double, 24, 12>(
           const double &, const state_ikfom &, const input_ikfom &)>
           &_jacbi_f_w,
       const std::function<void(state_ikfom &, dyn_share_datastruct &)>
           &_getjacbi_H_h,
       int _maximum_iter) {
    processFunction_ = _f;
    this->jacbi_f = _jacbi_f;
    this->jacbi_f_w = _jacbi_f_w;
    this->get_jacbi_H_h = _getjacbi_H_h;
    maximum_iter = _maximum_iter;
  };
  ~ESEKalmanFilter() {};
  state_ikfom getState() { return x_; }
  void setState(state_ikfom &x) { this->x_ = x; }
  Eigen::Matrix<double, 24, 24> getP() { return P_; }
  void setP(Eigen::Matrix<double, 24, 24> &p) { this->P_ = p; }
  void predict(double &dt, Eigen::Matrix<double, 12, 12> &Q,
               const input_ikfom &i_in) {

    F_x_ = jacbi_f(dt, x_, i_in);
    F_w_ = jacbi_f_w(dt, x_, i_in);
    x_ = processFunction_(dt, x_, i_in);
    P_ = (F_x_)*P_ * (F_x_).transpose() + F_w_ * Q * F_w_.transpose();
  }
  void update_iterated_dyn_share_modified(double R, double &solve_time) {
    dyn_share_datastruct dyn_share;
    dyn_share.valid = true;
    dyn_share.converge = true;
    int t = 0;
    state_ikfom x_propagated = x_;
    Eigen::Matrix<double, 24, 24> P_propagated = P_;
    // int dof_Measurement;
    Eigen::Matrix<double, 24, 1> K_h;
    Eigen::Matrix<double, 24, 24> K_x;
    Eigen::Matrix<double, 24, 1> dx_new = Eigen::Matrix<double, 24, 1>::Zero();
    for (int i = -1; i < maximum_iter; i++) {
      dyn_share.valid = true;
      // h_dyn_share(x_, dyn_share);
      get_jacbi_H_h(x_, dyn_share);

      if (!dyn_share.valid) {
        continue;
      }

      // 广义减法
      Eigen::Matrix<double, Eigen::Dynamic, 12> jacbi_H = dyn_share.jacbi_H;
      // double solve_start = omp_get_wtime();
      Eigen::Matrix<double, 24, 1> dx;
      dx.block<3, 1>(0, 0) = x_.pos - x_propagated.pos;

      dx.block<3, 1>(3, 0) =
          Sophus::SO3d(x_propagated.rot.matrix().transpose() * x_.rot.matrix())
              .log();
      dx.block<3, 1>(6, 0) =
          Sophus::SO3d(x_propagated.offset_R_L_I.matrix().transpose() *
                       x_.offset_R_L_I.matrix())
              .log();
      dx.block<3, 1>(9, 0) = x_.offset_T_L_I - x_propagated.offset_T_L_I;
      dx.block<3, 1>(12, 0) = x_.vel - x_propagated.vel;
      dx.block<3, 1>(15, 0) = x_.bg - x_propagated.bg;
      dx.block<3, 1>(18, 0) = x_.ba - x_propagated.ba;
      dx.block<3, 1>(21, 0) = x_.grav - x_propagated.grav;
      //
      dx_new = dx;
      P_ = P_propagated;
      Eigen::Vector3d seg_rot = Eigen::Vector3d(dx(3), dx(4), dx(5));
      Eigen::Vector3d seg_RLI = Eigen::Vector3d(dx(6), dx(7), dx(8));
      // Eigen::Matrix<double, 3, 3> seg_rot_A_T =
      //     get_A_matrix(seg_rot).transpose();
      // Eigen::Matrix<double, 3, 3> seg_RLI_A_T =
      //     get_A_matrix(seg_RLI).transpose();
      // dx_new.template block<3, 1>(3, 0) =
      //     seg_rot_A_T * dx_new.template block<3, 1>(3, 0); // delta_theta
      // dx_new.template block<3, 1>(6, 0) =
      //     seg_RLI_A_T * dx_new.template block<3, 1>(6, 0); // delta_RLI
      // for (int i = 0; i < 24; i++) {
      //   P_.template block<3, 1>(3, i) =
      //       seg_rot_A_T * (P_.template block<3, 1>(3, i));
      //   P_.template block<3, 1>(6, i) =
      //       seg_RLI_A_T * (P_.template block<3, 1>(6, i));
      // }
      // for (int i = 0; i < 24; i++) {
      //   P_.template block<1, 3>(i, 3) =
      //       (P_.template block<1, 3>(i, 3)) * seg_rot_A_T.transpose();
      //   P_.template block<1, 3>(i, 6) =
      //       (P_.template block<1, 3>(i, 6)) * seg_RLI_A_T.transpose();
      // }
      Eigen::Matrix<double, 24, 24> HTH =
          Eigen::Matrix<double, 24, 24>::Zero(); // 矩阵 H^T * H
      HTH.block<12, 12>(0, 0) = jacbi_H.transpose() * jacbi_H;
      auto K_front = (HTH / R + P_.inverse()).inverse();
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> K;
      K = K_front.block<24, 12>(0, 0) * jacbi_H.transpose() /
          R; // 卡尔曼增益  这里R视为常数
      Eigen::Matrix<double, 24, 24> KH =
          Eigen::Matrix<double, 24, 24>::Zero(); // 矩阵 K * H
      KH.block<24, 12>(0, 0) = K * jacbi_H;
      Eigen::Matrix<double, 24, 1> dx_ =
          K * dyn_share.h +
          (KH - Eigen::Matrix<double, 24, 24>::Identity()) * dx_new; // 公式(18)
      // 广义加法
      x_.pos = x_.pos + dx_.block<3, 1>(0, 0);
      x_.rot = x_.rot * Sophus::SO3d::exp(dx_.block<3, 1>(3, 0));
      x_.offset_R_L_I =
          x_.offset_R_L_I * Sophus::SO3d::exp(dx_.block<3, 1>(6, 0));
      x_.offset_T_L_I = x_.offset_T_L_I + dx_.block<3, 1>(9, 0);
      x_.vel = x_.vel + dx_.block<3, 1>(12, 0);
      x_.bg = x_.bg + dx_.block<3, 1>(15, 0);
      x_.ba = x_.ba + dx_.block<3, 1>(18, 0);
      x_.grav = x_.grav + dx_.block<3, 1>(21, 0);
      dyn_share.converge = true;
      for (int j = 0; j < 24; j++) {
        if (std::fabs(dx_[j]) > epsi) // 如果dx>epsi 认为没有收敛
        {
          dyn_share.converge = false;
          break;
        }
      }

      if (dyn_share.converge)
        t++;

      if (!t && i == maximum_iter -
                         2) // 如果迭代了3次还没收敛
                            // 强制令成true，h_share_model函数中会重新寻找近邻点
      {
        dyn_share.converge = true;
      }

      if (t > 1 || i == maximum_iter - 1) {
        P_ = (Eigen::Matrix<double, 24, 24>::Identity() - KH) * P_; // 公式(19)
        return;
      }
    }
  }



private:
  Eigen::Matrix<double, 24, 24> L_ = Eigen::Matrix<double, 24, 24>::Identity();
  int maximum_iter = 1;
  double epsi = 0.001;

  std::function<void(state_ikfom &, dyn_share_datastruct &)> get_jacbi_H_h;
  // 过程处理函数
  std::function<state_ikfom(const double &, const state_ikfom &,
                            const input_ikfom &)>
      processFunction_;
  // 求误差状态量的雅可比矩阵
  std::function<Eigen::Matrix<double, 24, 24>(
      const double &, const state_ikfom &, const input_ikfom &)>
      jacbi_f;
  // 求误差状态量对噪声的雅可比矩阵
  std::function<Eigen::Matrix<double, 24, 12>(
      const double &, const state_ikfom &, const input_ikfom &)>
      jacbi_f_w;
  // Eigen::Matrix<double, 24, 12>
  // getJacbi_f_w(const double &dt, const state_ikfom &x, const input_ikfom &in)
  // {
  //   // clang-format off
  //     /*
  //     x(delta_pos__, delta_theta, delta_RLI__, delta_TLI__, delta_vel__,
  //     delta_bgyro, delta_bacc_, delta_grav_) w(  noise_gyry_    noise_acc__
  //     noise_b_gyr    noise_b_acc)
  //              0              0              0              0

  //        -A(W_i*dt)*dt        0              0              0

  //              0              0              0              0

  //              0              0              0              0

  //              0            -R*dt            0              0

  //              0              0             I*dt            0

  //              0              0              0             I*dt

  //              0              0              0              0
  //     */
  //   // clang-format on
  //   Eigen::Matrix<double, 24, 12> jacbi_f_w =
  //       Eigen::Matrix<double, 24, 12>::Zero();
  //   Eigen::Vector3d omega = in.gyro - x.bg;
  //   Eigen::Vector3d delta_omega = omega * dt;
  //   jacbi_f_w.template block<3, 3>(12, 3) = -x.rot.matrix() * dt;
  //   jacbi_f_w.template block<3, 3>(3, 0) = -get_A_matrix(delta_omega) * dt;
  //   jacbi_f_w.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity() * dt;
  //   jacbi_f_w.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity() * dt;
  //   return jacbi_f_w;
  // }
  // // 运动过程
  // state_ikfom process(const double &dt, const state_ikfom &x,
  //                     const input_ikfom &in) {
  //   Eigen::Vector3d omega = in.gyro - x.bg;
  //   Eigen::Vector3d a_inertial = x.rot.matrix() * (in.acc - x.ba);
  //   state_ikfom out;
  //   // out.pos = x.pos + x.vel * dt;
  //   // out.rot = x.rot * Sophus::SO3d::exp(omega * dt);
  //   // out.offset_R_L_I =
  //   //     x.offset_R_L_I * Sophus::SO3d::exp(Eigen::Vector3d(0, 0, 0));
  //   // out.vel = x.vel + (a_inertial + x.grav) * dt;
  //   // out.ba = x.ba + Eigen::Vector3d(0, 0, 0);
  //   // out.bg = x.bg + Eigen::Vector3d(0, 0, 0);
  //   // out.grav = x.grav + Eigen::Vector3d(0, 0, 0);

  //   return out;
  // }
  // Eigen::Matrix<double, 24, 24>
  // getJacbi_f(const double &dt, const state_ikfom &x, const input_ikfom &in) {
  //   // clang-format off
  //   /*
  //   x(delta_pos__, delta_theta, delta_RLI__, delta_TLI__, delta_vel__,
  //   delta_bgyro, delta_bacc_, delta_grav_)
  //          I            0            0            0           I*dt          0
  //          0            0

  //          0       exp(-W_i*dt)      0            0            0 -A(W_i*dt)
  //          0            0

  //          0            0            I            0            0            0
  //          0            0

  //          0            0            0            I            0            0
  //          0            0

  //          0        -R(acc^)*dt      0            0            I            0
  //          -R*dt         I*dt

  //          0            0            0            0            0            I
  //          0            0

  //          0            0            0            0            0            0
  //          I            0

  //          0            0            0            0            0            0
  //          0            I
  //   */
  //   // clang-format on
  //   Eigen::Matrix<double, 24, 24> jacbi =
  //       Eigen::Matrix<double, 24, 24>::Identity();
  //   jacbi.block<3, 3>(0, 12) = Eigen::Matrix3d::Identity() * dt;
  //   Eigen::Vector3d acc_ = in.acc - x.ba; // 测量加速度 = a_m - bias
  //   Eigen::Vector3d omega = in.gyro - x.bg;
  //   Eigen::Vector3d delta_omega = omega * dt;
  //   jacbi.block<3, 3>(3, 3) = Sophus::SO3d::exp(delta_omega * (-1)).matrix();
  //   jacbi.block<3, 3>(3, 15) = get_A_matrix(delta_omega) * (-1);
  //   jacbi.block<3, 3>(12, 3) =
  //       x.rot.matrix() * Sophus::SO3d::hat(acc_) * dt * (-1);
  //   jacbi.block<3, 3>(12, 18) = x.rot.matrix() * dt * (-1);
  //   jacbi.block<3, 3>(12, 21) = Eigen::Matrix<double, 3, 3>::Identity() * dt;
  //   return jacbi;
  // }

  state_ikfom x_;
  Eigen::Matrix<double, 24, 12> F_w_; // 状态量对噪声的雅可比矩阵
  Eigen::Matrix<double, 24, 24> F_x_; // 误差状态量的雅可比矩阵
  Eigen::Matrix<double, 24, 24> P_ = Eigen::Matrix<double, 24, 24>::Identity();
};




} // namespace simple_fast_lio2_lc

#endif
