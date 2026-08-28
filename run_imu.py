#!/usr/bin/env python3
"""Run the G1 walking simulation with live right-foot IMU visualization."""

import sys
import mujoco
import numpy as np

import run
from imu_data_reader import IMUDataReader
from trajectory_visualizer import SiteTrajectoryVisualizer
from imu_data_visualizer import IMUDataVisualizer
from esekf import ESEKF
from zupt_trigger import ZUPTTrigger
from data_logger import SimulationDataLogger


class IMUG1Controller(run.G1Controller):
  """G1 controller with an initial walking command and IMU trajectory plot."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.lin_vel_x = 0.45
    self.lin_vel_y = 0.0
    self.ang_vel_z = 0.0

    self.zupt_trigger = ZUPTTrigger(
        self.model,
        self.data,
        site_name="imu_right_foot",
        velocity_threshold=0.05,
        min_contact_points=3,
        print_hz=5.0,
    )

    # ESEKF chạy cố định ở 200 Hz,
    # độc lập với physics timestep của MuJoCo
    self.esekf.delta_t = 0.005

    # Giữ Q nhất quán với sampling period của ESEKF
    self.esekf.Q = (
        self.esekf.Qc
        / self.esekf.delta_t
    )

    self.imu_trajectory = SiteTrajectoryVisualizer(
      self.model,
      self.data,
      site_name="imu_right_foot",
      plot_fps=10.0,
    )

    self.imu_reader = IMUDataReader(
      self.model,
      self.data,
      accelerometer_name="imu_right_acc",
      gyroscope_name="imu_right_gyro",
      print_hz=5.0,
    )

    self.imu_visualizer = IMUDataVisualizer(
      plot_fps=10.0,
    )

    self.zupt_trigger = ZUPTTrigger(
        self.model,
        self.data,
        site_name="imu_right_foot",
        print_hz=5.0,
    )

    self.data_logger = SimulationDataLogger()

    # Mẫu IMU tại k-1 dùng để dự đoán trạng thái tại k
    self.previous_acceleration = None
    self.previous_angular_velocity = None
    self.previous_imu_time = None

  def initialize_esekf_input(self) -> None:
    """Lưu mẫu IMU ban đầu u_0 để dự đoán x_1."""

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


  def step_esekf(self) -> None:

    # Đọc mẫu IMU hiện tại u_k
    current_acceleration, current_angular_velocity = (
        self.imu_reader.update()
    )

    current_imu_time = float(
        self.data.time
    )

    # Hiển thị dữ liệu IMU tại đúng thời điểm đo t_k
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

    # Dự đoán x_k từ x_{k-1} bằng u_{k-1}
    self.esekf.predict(
        acceleration=self.previous_acceleration,
        angular_velocity=self.previous_angular_velocity,
    )

    # ZUPT correction tại thời điểm t_k
    zupt_active = self.zupt_trigger.check()

    true_velocity = (
        self.zupt_trigger.get_true_linear_velocity()
    )

    if zupt_active:
        self.esekf.correct_zupt(
            true_velocity
        )

    # Lấy trạng thái danh định sau predict/correction
    estimated_position = self.esekf.position.copy()
    estimated_velocity = self.esekf.velocity.copy()
    estimated_quaternion = self.esekf.quaternion.copy()

    # Ground-truth position của IMU site trong world frame
    true_position = (
        self.data.site_xpos[
            self.esekf.site_id
        ].copy()
    )

    # Ground-truth quaternion IMU -> world [w, x, y, z]
    true_quaternion = np.empty(
        4,
        dtype=float,
    )

    # Lấy quaternion thật từ MuJoCo trước
    mujoco.mju_mat2Quat(
        true_quaternion,
        self.data.site_xmat[
            self.esekf.site_id
        ],
    )

    # q và -q biểu diễn cùng một rotation.
    # Chọn dấu ground truth gần với quaternion ESEKF nhất
    # để thuận tiện khi plot và so sánh.
    if (
        np.dot(
            true_quaternion,
            estimated_quaternion,
        ) < 0.0
    ):
        true_quaternion *= -1.0

    self.data_logger.log(
        sample_time=current_imu_time,

        acceleration=current_acceleration,
        angular_velocity=current_angular_velocity,

        gt_position=true_position,
        gt_velocity=true_velocity,
        gt_quaternion=true_quaternion,

        est_position=estimated_position,
        est_velocity=estimated_velocity,
        est_quaternion=estimated_quaternion,

        correction_applied=zupt_active,

        covariance=self.esekf.P.copy(),
    )

    # Plot trajectory ESEKF
    self.imu_trajectory.update(
        estimated_position,
        estimated_velocity,
        estimated_quaternion,
    )

    # Lưu u_k để sử dụng cho lần dự đoán tiếp theo:
    # x_{k+1} = f(x_k, u_k)
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
  # Make run.main() instantiate the extended controller without editing run.py.
  run.G1Controller = IMUG1Controller

  # The option defined by run.py is plural: --no-cameras.
  if "--no-cameras" not in sys.argv:
    sys.argv.append("--no-cameras")

  run.main()