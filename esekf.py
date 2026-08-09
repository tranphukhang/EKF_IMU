"""Incremental ESEKF implementation for the right-foot IMU."""

from __future__ import annotations

import mujoco
import numpy as np


class ESEKF:
  """Hold the nominal ESEKF state for the right-foot IMU."""

  def __init__(
      self,
      model: mujoco.MjModel,
      data: mujoco.MjData,
      site_name: str = "imu_right_foot",
  ) -> None:
    self.model = model
    self.data = data
    self.site_name = site_name

    self.site_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_SITE, site_name
    )
    if self.site_id < 0:
      raise ValueError(f"Khong tim thay site MuJoCo: {site_name!r}")

    # Initial nominal state x0_hat = [p, v, q], with q = [w, x, y, z].
    # The ESEKF origin is chosen at the initial IMU position; therefore p0 = [0, 0, 0.05].
    self.x0_hat = np.array([
      0.0, 0.0, 0.05,       # p0 [m]
      0.0, 0.0, 0.0,       # v0 [m/s]
      1.0, 0.0, 0.0, 0.0,  # q0: IMU -> world [w, x, y, z]
    ], dtype=float)

    # Trạng thái danh định hiện tại
    self.x_hat = self.x0_hat.copy()

    self.position = self.x_hat[0:3]
    self.velocity = self.x_hat[3:6]
    self.quaternion = self.x_hat[6:10]

    # Độ lệch chuẩn ban đầu của trạng thái sai số
    sigma_p = np.array([0.001, 0.001, 0.001])       # [m]
    sigma_v = np.array([0.01, 0.01, 0.01])          # [m/s]
    sigma_theta = np.deg2rad([1.0, 1.0, 5.0])       # roll, pitch, yaw [rad]

    self.P0 = np.diag(
        np.concatenate([
            sigma_p**2,
            sigma_v**2,
            sigma_theta**2,
        ])
    )

    # Hiệp phương sai ban đầu
    self.P = self.P0.copy()

    # Jacobian phép đo ZUPT
    # Error state: [delta_p, delta_v, delta_theta]
    self.H_zupt = np.zeros((3, 9), dtype=float)
    self.H_zupt[:, 3:6] = np.eye(3)

    # Độ lệch chuẩn của pseudo-measurement ZUPT
    sigma_zupt = 0.01  # [m/s]
    # Measurement covariance của ZUPT
    self.R_zupt = np.eye(3, dtype=float) * sigma_zupt**2

    # Chu kỳ lấy mẫu IMU: 200 Hz
    self.delta_t = 0.005  # [s]

    # Continuous-time noise covariance density từ VN-100
    q_acc_c = 1.88494e-6
    q_gyro_c = 3.73156e-9
    self.Qc = np.diag([
        q_acc_c, q_acc_c, q_acc_c,
        q_gyro_c, q_gyro_c, q_gyro_c,
    ])
    # Covariance của từng mẫu IMU tại dt = 0.005 s (200 Hz)
    self.Q = self.Qc / self.delta_t

    # Vector trọng lực trong hệ tọa độ world
    self.gravity = np.asarray(
        self.model.opt.gravity,
        dtype=float,
    ).copy()

    self.initialized = False

    # In kết quả dự đoán với tần số 5 Hz
    self.predict_print_period = 1.0 / 5.0
    self.last_predict_print_time = -np.inf

  @staticmethod
  def _skew(vector: np.ndarray) -> np.ndarray:
      x, y, z = vector

      return np.array([
          [0.0, -z,   y],
          [z,    0.0, -x],
          [-y,   x,   0.0],
      ])

  @staticmethod
  def _global_attitude_error(
        q_true: np.ndarray,
        q_est: np.ndarray,
    ) -> np.ndarray:
        """Rotation-vector error với global/left angular error."""

        q_true = np.asarray(q_true, dtype=float).copy()
        q_est = np.asarray(q_est, dtype=float).copy()

        mujoco.mju_normalize4(q_true)
        mujoco.mju_normalize4(q_est)

        # q_est^{-1} = conjugate(q_est)
        q_est_inv = q_est.copy()
        q_est_inv[1:4] *= -1.0

        # Global error:
        # delta_q = q_true ⊗ q_est^{-1}
        delta_q = np.empty(4, dtype=float)

        mujoco.mju_mulQuat(
            delta_q,
            q_true,
            q_est_inv,
        )

        mujoco.mju_normalize4(delta_q)

        # q và -q biểu diễn cùng một rotation.
        # Chọn representation có góc nhỏ nhất.
        if delta_q[0] < 0.0:
            delta_q *= -1.0

        vector_part = delta_q[1:4]
        sin_half_angle = np.linalg.norm(vector_part)

        if sin_half_angle < 1e-12:
            return 2.0 * vector_part

        angle = 2.0 * np.arctan2(
            sin_half_angle,
            delta_q[0],
        )

        axis = vector_part / sin_half_angle

        return angle * axis

  def predict(
        self,
        acceleration: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> np.ndarray:
    """Lan truyen trang thai danh dinh p, v, q."""
    acceleration = np.asarray(acceleration, dtype=float)
    angular_velocity = np.asarray(angular_velocity, dtype=float)

    # Lưu trạng thái tại k-1, tránh dùng giá trị vừa cập nhật
    position_previous = self.position.copy()
    velocity_previous = self.velocity.copy()
    quaternion_previous = self.quaternion.copy()

    # R(q_k-1): ma trận xoay từ IMU sang world
    rotation_flat = np.empty(9, dtype=float)

    mujoco.mju_quat2Mat(
        rotation_flat,
        quaternion_previous,
    )

    rotation_imu_to_world = rotation_flat.reshape(3, 3)

    # R_{k-1} a_m,k
    rotated_acceleration = (
        rotation_imu_to_world @ acceleration
    )

    # Jacobian trạng thái sai số F_x,k
    Fx = np.eye(9, dtype=float)
    Fx[0:3, 3:6] = (
        np.eye(3) * self.delta_t
    )
    Fx[3:6, 6:9] = (
        -self._skew(rotated_acceleration)
        * self.delta_t
    )
    self.Fx = Fx

    # Jacobian nhiễu quá trình F_i,k
    Fi = np.zeros((9, 6), dtype=float)
    Fi[3:6, 0:3] = (
        -rotation_imu_to_world
        * self.delta_t
    )
    Fi[6:9, 3:6] = (
        -rotation_imu_to_world
        * self.delta_t
    )
    self.Fi = Fi

    # Dự đoán ma trận hiệp phương sai
    self.P = (
        self.Fx @ self.P @ self.Fx.T
        + self.Fi @ self.Q @ self.Fi.T
    )
    # Giữ P đối xứng do sai số số học
    self.P = 0.5 * (self.P + self.P.T)

    # a^w_k = R(q_k-1) a_k + g
    acceleration_world = (
        rotated_acceleration
        + self.gravity
    )

    # p_k = p_k-1 + v_k-1 * delta_t
    position_new = (
        position_previous
        + velocity_previous * self.delta_t
    )

    # v_k = v_k-1 + a^w_k * delta_t
    velocity_new = (
        velocity_previous
        + acceleration_world * self.delta_t
    )

    # q_k = q_k-1 ⊗ q{omega_k * delta_t}
    quaternion_new = quaternion_previous.copy()
    mujoco.mju_quatIntegrate(
        quaternion_new,
        angular_velocity,
        self.delta_t,
    )

    # Chuẩn hóa để quaternion luôn có norm bằng 1
    mujoco.mju_normalize4(quaternion_new)

    # Ghi kết quả vào trạng thái danh định hiện tại
    self.position[:] = position_new
    self.velocity[:] = velocity_new
    self.quaternion[:] = quaternion_new

    # self._print_prediction()

    return self.x_hat.copy()
    

  def correct_zupt(self) -> np.ndarray:
    """Bước hiệu chỉnh ZUPT."""

    ######
    # Quaternion ground truth IMU -> world tại thời điểm hiện tại
    q_true = np.empty(4, dtype=float)

    mujoco.mju_mat2Quat(
        q_true,
        self.data.site_xmat[self.site_id],
    )

    # Quaternion estimate trước ZUPT correction
    q_est_before = self.quaternion.copy()

    # Attitude error thật trước correction
    attitude_error_before = self._global_attitude_error(
        q_true,
        q_est_before,
    )
    #######

    # Phép đo giả ZUPT: vận tốc chân bằng 0
    z = np.zeros(3, dtype=float)

    # h(x_hat) = v_hat
    predicted_measurement = self.velocity.copy()

    # Residual / innovation:
    # r = z - h(x_hat)
    r = z - predicted_measurement

    self.zupt_residual = r.copy()

    # Innovation covariance
    H = self.H_zupt
    R = self.R_zupt
    S = H @ self.P @ H.T + R
    self.zupt_innovation_covariance = S.copy()

    # Kalman gain
    PHt = self.P @ H.T
    K = np.linalg.solve(
        S,
        PHt.T,
    ).T
    self.K_zupt = K.copy()

    # Ước lượng trạng thái sai số hiệu chỉnh
    # delta_x = K r
    delta_x = K @ r
    # Kiểm tra delta_x
    assert delta_x.shape == (9,)
    assert np.all(np.isfinite(delta_x))
    self.delta_x_zupt = delta_x.copy()
    delta_p = delta_x[0:3]
    delta_v = delta_x[3:6]
    delta_theta = delta_x[6:9]

    # Inject position và velocity error vào nominal state
    self.position[:] += delta_p
    self.velocity[:] += delta_v
    # Chuyển delta_theta thành delta quaternion
    quaternion_before = self.quaternion.copy()
    angle = np.linalg.norm(delta_theta)
    delta_q = np.empty(4, dtype=float)
    if angle < 1e-12:
        delta_q[:] = np.array([
            1.0,
            0.5 * delta_theta[0],
            0.5 * delta_theta[1],
            0.5 * delta_theta[2],
        ])
    else:
        axis = delta_theta / angle
        mujoco.mju_axisAngle2Quat(
            delta_q,
            axis,
            angle,
        )
    # Global angular error:
    # q_plus = delta_q ⊗ q_minus
    quaternion_corrected = np.empty(4, dtype=float)
    mujoco.mju_mulQuat(
        quaternion_corrected,
        delta_q,
        quaternion_before,
    )
    mujoco.mju_normalize4(quaternion_corrected)
    self.quaternion[:] = quaternion_corrected

    #####
    # Attitude error thật sau ZUPT correction
    attitude_error_after = self._global_attitude_error(
        q_true,
        self.quaternion,
    )
    # Giá trị attitude error dự kiến sau injection
    # theo xấp xỉ góc nhỏ của global error
    expected_error_after = (
        attitude_error_before - delta_theta
    )

    # Sai lệch giữa quaternion injection thực tế
    # và quan hệ tuyến tính dự kiến
    injection_check_error = (
        attitude_error_after
        - expected_error_after
    )
    alignment = np.dot(
        attitude_error_before,
        delta_theta,
    )
    #####

    # Cập nhật covariance sau ZUPT
    I_KH = np.eye(9, dtype=float) - K @ H
    P_corrected = I_KH @ self.P @ I_KH.T + K @ R @ K.T
    # Giữ P đối xứng do sai số số học
    P_corrected = 0.5 * (P_corrected + P_corrected.T)
    self.P = P_corrected

    # Reset Jacobian
    G_reset = np.eye(9, dtype=float)
    G_theta = (
        np.eye(3, dtype=float)
        + 0.5 * self._skew(delta_theta)
    )
    G_reset[6:9, 6:9] = G_theta
    self.P = (G_reset @ self.P @ G_reset.T)
    self.P = 0.5 * (self.P + self.P.T)


    print("\n===== ATTITUDE INJECTION CHECK =====")

    print(
        "true error before =",
        attitude_error_before,
    )

    print(
        "delta_theta EKF   =",
        delta_theta,
    )

    print(
        "expected after    =",
        expected_error_after,
    )

    print(
        "actual after      =",
        attitude_error_after,
    )

    print(
        "injection check error =",
        injection_check_error,
    )

    print(
        "alignment =",
        alignment,
    )

    print("====================================\n")


    return r.copy()


  def _print_prediction(self) -> None:
    current_time = float(self.data.time)

    # Xử lý khi thời gian mô phỏng được reset
    if current_time < self.last_predict_print_time:
        self.last_predict_print_time = -np.inf

    if (
        current_time - self.last_predict_print_time
        < self.predict_print_period
    ):
        return

    self.last_predict_print_time = current_time

    print(f"\n=== ESEKF: KET QUA DU DOAN t = {current_time:.3f} s ===")
    print(f"Vi tri p_hat [m]          = {self.position}")
    print(f"Van toc v_hat [m/s]       = {self.velocity}")
    print(f"Quaternion q_hat [w,x,y,z] = {self.quaternion}")

    # Kiểm tra covariance P
    P_diag = np.diag(self.P)
    P_std = np.sqrt(np.maximum(P_diag, 0.0))
    symmetry_error = np.max(np.abs(self.P - self.P.T))
    min_eigenvalue = np.min(np.linalg.eigvalsh(self.P))
    print(f"P diag                      = {P_diag}")
    print(f"Std position [m]            = {P_std[0:3]}")
    print(f"Std velocity [m/s]          = {P_std[3:6]}")
    print(f"Std attitude [rad]          = {P_std[6:9]}")
    print(f"P symmetry error            = {symmetry_error:.3e}")
    print(f"Min eigenvalue of P         = {min_eigenvalue:.3e}")
    print("================================================")

  
  def initialize_once(self) -> None:
    """Print the declared initial nominal state once."""
    if self.initialized:
      return

    self.initialized = True
    self._print_initial_state()

    
  def _print_initial_state(self) -> None:
    print("\n=== ESEKF: TRANG THAI DANH DINH BAN DAU ===")
    print(f"x0_hat = {self.x0_hat}")
    print(
      "Vi tri IMU trong world [m]         "
      f"[x, y, z] = {self.position}"
    )
    print(
      "Van toc IMU trong world [m/s]     "
      f"[vx, vy, vz] = {self.velocity}"
    )
    print(
      "Quaternion IMU -> world [w,x,y,z] "
      f"= {self.quaternion}"
    )

    print("Ma tran hiep phuong sai ban dau P0:")
    print(
        np.array2string(
            self.P,
            precision=8,
            suppress_small=True,
        )
    )

    print(f"Delta t = {self.delta_t:.4f} s")
    print("=================================\n")

  def _print_zupt_debug(
        self,
        H: np.ndarray,
        R: np.ndarray,
        S: np.ndarray,
        K: np.ndarray,
    ) -> None:

    print("\n========== ZUPT DEBUG ==========")

    print("H shape =", H.shape)
    print(H)

    print("\nR shape =", R.shape)
    print(R)

    print("\nS shape =", S.shape)
    print(S)

    print("\nK shape =", K.shape)
    print(K)

    print("\nResidual r =", self.zupt_residual)

    print(
        "S symmetry error =",
        np.max(np.abs(S - S.T))
    )

    print(
        "Min eigenvalue of S =",
        np.min(np.linalg.eigvalsh(S))
    )

    print("================================\n")