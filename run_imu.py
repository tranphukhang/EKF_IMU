#!/usr/bin/env python3
"""Run the G1 walking simulation with live right-foot IMU visualization."""

import sys

import mujoco
import numpy as np

import run
from data_logger import SimulationDataLogger
from esekf import ESEKF
from imu_data_reader import IMUDataReader
from imu_data_visualizer import IMUDataVisualizer
from trajectory_visualizer import SiteTrajectoryVisualizer
from zupt_trigger import ZUPTTrigger


class IMUG1Controller(
    run.G1Controller
):
  """G1 controller with right-foot IMU and ESEKF."""

  def __init__(
      self,
      *args,
      **kwargs,
  ):
    super().__init__(
        *args,
        **kwargs,
    )

    self.lin_vel_x = 0.45
    self.lin_vel_y = 0.0
    self.ang_vel_z = 0.0

    # ==========================================================
    # ESEKF
    # ==========================================================

    self.esekf = ESEKF(
        self.model,
        self.data,
        site_name="imu_right_foot",
    )

    # ESEKF chạy ở 200 Hz
    self.esekf.delta_t = 0.005

    # Giữ Q nhất quán với sampling period
    self.esekf.Q = (
        self.esekf.Qc
        / self.esekf.delta_t
    )

    # ==========================================================
    # ZUPT trigger
    # ==========================================================

    self.zupt_trigger = ZUPTTrigger(
        self.model,
        self.data,
        site_name="imu_right_foot",
        velocity_threshold=0.05,
        min_contact_points=3,
        print_hz=5.0,
    )

    # ==========================================================
    # Trajectory visualizer
    # ==========================================================

    self.imu_trajectory = SiteTrajectoryVisualizer(
        self.model,
        self.data,
        site_name="imu_right_foot",
        plot_fps=10.0,
    )

    # ==========================================================
    # IMU reader
    # ==========================================================

    self.imu_reader = IMUDataReader(
        self.model,
        self.data,
        accelerometer_name="imu_right_acc",
        gyroscope_name="imu_right_gyro",
        print_hz=5.0,
    )

    # ==========================================================
    # IMU visualizer
    # ==========================================================

    self.imu_visualizer = IMUDataVisualizer(
        plot_fps=10.0,
    )

    # ==========================================================
    # CSV logger
    # ==========================================================

    self.data_logger = SimulationDataLogger()

    # Mẫu IMU tại k-1 dùng để dự đoán trạng thái tại k
    self.previous_acceleration = None
    self.previous_angular_velocity = None
    self.previous_imu_time = None

  def initialize_esekf_input(
      self,
  ) -> None:
    """Lưu mẫu IMU u_0 để dự đoán trạng thái x_1."""

    acceleration, angular_velocity = (
        self.imu_reader.update()
    )

    self.previous_acceleration = (
        acceleration.copy()
    )

    self.previous_angular_velocity = (
        angular_velocity.copy()
    )

    self.previous_imu_time = float(
        self.data.time
    )

    self.esekf.initialize_once()

  def step(self):
    target_pos = super().step()

    return target_pos

  def step_esekf(
      self,
  ) -> None:

    # ==========================================================
    # Đọc mẫu IMU hiện tại u_k
    # ==========================================================

    (
        current_acceleration,
        current_angular_velocity,
    ) = self.imu_reader.update()

    # Tín hiệu IMU lý tưởng tại cùng thời điểm,
    # trước khi cộng nhiễu VN-100.
    ground_truth_acceleration = (
        self.imu_reader
        .latest_ground_truth_acceleration
        .copy()
    )

    ground_truth_angular_velocity = (
        self.imu_reader
        .latest_ground_truth_angular_velocity
        .copy()
    )

    current_imu_time = float(
        self.data.time
    )

    # ==========================================================
    # Hiển thị dữ liệu IMU có nhiễu tại thời điểm t_k
    # ==========================================================

    self.imu_visualizer.update(
        sample_time=current_imu_time,
        acceleration=current_acceleration,
        angular_velocity=current_angular_velocity,
    )

    if (
        self.previous_acceleration is None
        or self.previous_angular_velocity is None
    ):
        raise RuntimeError(
            "Mẫu IMU ban đầu chưa được khởi tạo."
        )

    # ==========================================================
    # Prediction:
    # x_k = f(x_{k-1}, u_{k-1})
    # ==========================================================

    self.esekf.predict(
        acceleration=(
            self.previous_acceleration
        ),
        angular_velocity=(
            self.previous_angular_velocity
        ),
    )

    # ==========================================================
    # ZUPT correction tại thời điểm t_k
    # ==========================================================

    zupt_active = (
        self.zupt_trigger.check()
    )

    true_velocity = (
        self.zupt_trigger
        .get_true_linear_velocity()
    )

    if zupt_active:
        self.esekf.correct_zupt(
            true_velocity
        )

    # ==========================================================
    # Trạng thái ESEKF sau predict/correction
    # ==========================================================

    estimated_position = (
        self.esekf.position.copy()
    )

    estimated_velocity = (
        self.esekf.velocity.copy()
    )

    estimated_quaternion = (
        self.esekf.quaternion.copy()
    )

    # ==========================================================
    # Ground-truth position trong world frame
    # ==========================================================

    true_position = (
        self.data.site_xpos[
            self.esekf.site_id
        ].copy()
    )

    # ==========================================================
    # Ground-truth quaternion IMU -> world
    # [qw, qx, qy, qz]
    # ==========================================================

    true_quaternion = np.empty(
        4,
        dtype=float,
    )

    mujoco.mju_mat2Quat(
        true_quaternion,
        self.data.site_xmat[
            self.esekf.site_id
        ],
    )

    # q và -q biểu diễn cùng một rotation.
    # Chọn dấu ground truth gần ESEKF nhất để plot.
    if (
        np.dot(
            true_quaternion,
            estimated_quaternion,
        )
        < 0.0
    ):
        true_quaternion *= -1.0

    # ==========================================================
    # Log toàn bộ dữ liệu
    # ==========================================================

    self.data_logger.log(
        sample_time=current_imu_time,

        # IMU sau khi cộng nhiễu
        acceleration=(
            current_acceleration
        ),

        angular_velocity=(
            current_angular_velocity
        ),

        # IMU ground truth trước khi cộng nhiễu
        ground_truth_acceleration=(
            ground_truth_acceleration
        ),

        ground_truth_angular_velocity=(
            ground_truth_angular_velocity
        ),

        # Ground-truth state
        gt_position=true_position,
        gt_velocity=true_velocity,
        gt_quaternion=true_quaternion,

        # Estimated state
        est_position=estimated_position,
        est_velocity=estimated_velocity,
        est_quaternion=estimated_quaternion,

        correction_applied=zupt_active,

        covariance=(
            self.esekf.P.copy()
        ),
    )

    # ==========================================================
    # Cập nhật trajectory visualizer
    # ==========================================================

    self.imu_trajectory.update(
        estimated_position,
        estimated_velocity,
        estimated_quaternion,
    )

    # ==========================================================
    # Lưu u_k để dự đoán trạng thái tại k+1
    # x_{k+1} = f(x_k, u_k)
    # ==========================================================

    self.previous_acceleration = (
        current_acceleration.copy()
    )

    self.previous_angular_velocity = (
        current_angular_velocity.copy()
    )

    self.previous_imu_time = (
        current_imu_time
    )


if __name__ == "__main__":
  # Cho run.main() sử dụng controller có ESEKF
  run.G1Controller = IMUG1Controller

  # Tắt camera để tăng tốc mô phỏng
  if "--no-cameras" not in sys.argv:
    sys.argv.append(
        "--no-cameras"
    )

  run.main()