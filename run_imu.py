#!/usr/bin/env python3
"""Run the G1 walking simulation with live right-foot IMU visualization."""

import sys

import run
from imu_data_reader import IMUDataReader
from trajectory_visualizer import SiteTrajectoryVisualizer
from imu_data_visualizer import IMUDataVisualizer
from esekf import ESEKF
from zupt_trigger import ZUPTTrigger


class IMUG1Controller(run.G1Controller):
  """G1 controller with an initial walking command and IMU trajectory plot."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.lin_vel_x = 0.0
    self.lin_vel_y = 0.0
    self.ang_vel_z = 0.0

    self.esekf = ESEKF(
      self.model,
      self.data,
      site_name="imu_right_foot",
    )

    self.esekf.delta_t = float(self.model.opt.timestep)

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


  def step(self):
    target_pos = super().step()
    return target_pos


  def step_esekf(self) -> None:

    # Luôn đọc IMU để vẫn quan sát được transient ban đầu
    acceleration, angular_velocity = self.imu_reader.update()

    # Luôn plot raw IMU
    self.imu_visualizer.update(
        sample_time=self.data.time,
        acceleration=acceleration,
        angular_velocity=angular_velocity,
    )

    # Chỉ khởi tạo ESEKF sau khi robot đã settle
    self.esekf.initialize_once()

    # Prediction
    self.esekf.predict(
        acceleration=acceleration,
        angular_velocity=angular_velocity,
    )

    # ZUPT correction
    zupt_active = self.zupt_trigger.check()
    if zupt_active:
      true_velocity = (
          self.zupt_trigger.get_true_linear_velocity()
      )

      self.esekf.correct_zupt(true_velocity)

    # Lấy trạng thái danh định sau predict/correction
    estimated_position = self.esekf.position.copy()
    estimated_velocity = self.esekf.velocity.copy()
    estimated_quaternion = self.esekf.quaternion.copy()

    # Plot trajectory ESEKF
    self.imu_trajectory.update(
        estimated_position,
        estimated_velocity,
        estimated_quaternion,
    )


if __name__ == "__main__":
  # Make run.main() instantiate the extended controller without editing run.py.
  run.G1Controller = IMUG1Controller

  # The option defined by run.py is plural: --no-cameras.
  if "--no-cameras" not in sys.argv:
    sys.argv.append("--no-cameras")

  run.main()