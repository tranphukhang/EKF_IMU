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
    # The ESEKF origin is chosen at the initial IMU position; therefore p0 = 0.
    self.x0_hat = np.array([
      0.0, 0.0, 0.0,       # p0 [m]
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

    # Chu kỳ lấy mẫu IMU: 200 Hz
    self.delta_t = 0.005  # [s]

    # Vector trọng lực trong hệ tọa độ world
    self.gravity = np.asarray(
        self.model.opt.gravity,
        dtype=float,
    ).copy()

    self.initialized = False

    # In kết quả dự đoán với tần số 5 Hz
    self.predict_print_period = 1.0 / 5.0
    self.last_predict_print_time = -np.inf

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

    # a^w_k = R(q_k-1) a_k + g
    acceleration_world = (
        rotation_imu_to_world @ acceleration
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

    self._print_prediction()

    return self.x_hat.copy()
    

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