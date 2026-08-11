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

    self.data_logger = SimulationDataLogger()


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

    # ============================================================
    # TEST_ONLY BEGIN
    # Prediction-only test:
    # Không cho ESEKF sử dụng bất kỳ measurement correction nào.
    # Logger vẫn lưu ground truth để đối chiếu.
    # Sau test xóa block này.
    # ============================================================
    zupt_active = False
    # TEST_ONLY END

    # Ground-truth velocity của IMU trong world frame
    true_velocity = (
        self.zupt_trigger.get_true_linear_velocity()
    )

    if zupt_active:
        self.esekf.correct_zupt(true_velocity)

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
        sample_time=float(self.data.time),

        acceleration=acceleration,
        angular_velocity=angular_velocity,

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

  # ============================================================
  # TEST_TIMING_SPLIT BEGIN
  # Pha PRE-STEP:
  # Đọc IMU tại state t_k và dùng chính sample này
  # để ESEKF predict từ t_k -> t_k+1.
  # Sau test xóa toàn bộ hàm này.
  # ============================================================
  def test_esekf_pre_step(self) -> None:

    # Đọc IMU tại state hiện tại t_k
    acceleration, angular_velocity = (
        self.imu_reader.update()
    )

    # Lưu lại chính sample đã dùng cho prediction.
    # Pha post-step sẽ dùng lại để log.
    self._test_acceleration = acceleration.copy()
    self._test_angular_velocity = angular_velocity.copy()

    # Plot raw IMU tại thời điểm t_k
    self.imu_visualizer.update(
        sample_time=self.data.time,
        acceleration=acceleration,
        angular_velocity=angular_velocity,
    )

    # ESEKF đã được initialize từ GT bởi TEST_ONLY trước đó.
    # initialize_once() ở đây chủ yếu để giữ logic print hiện tại.
    self.esekf.initialize_once()

    # Prediction:
    # x_hat(t_k) -> x_hat(t_k+1)
    self.esekf.predict(
        acceleration=acceleration,
        angular_velocity=angular_velocity,
    )

  # TEST_TIMING_SPLIT END

  # ============================================================
  # TEST_TIMING_SPLIT BEGIN
  # Pha POST-STEP:
  # MuJoCo đã chuyển sang t_k+1 và mj_forward() đã recompute
  # các đại lượng dẫn xuất. Lúc này mới lấy GT để đối chiếu
  # với ESEKF prediction tại cùng thời điểm t_k+1.
  # Sau test xóa toàn bộ hàm này.
  # ============================================================
  def test_esekf_post_step(self) -> None:

    # Bảo đảm pre-step đã chạy
    assert hasattr(
        self,
        "_test_acceleration",
    )

    assert hasattr(
        self,
        "_test_angular_velocity",
    )

    acceleration = (
        self._test_acceleration.copy()
    )

    angular_velocity = (
        self._test_angular_velocity.copy()
    )

    # Không correction trong prediction-only test
    zupt_active = False

    # Ground-truth velocity tại t_k+1
    true_velocity = (
        self.zupt_trigger.get_true_linear_velocity()
    )

    # ESEKF prediction tại t_k+1
    estimated_position = (
        self.esekf.position.copy()
    )

    estimated_velocity = (
        self.esekf.velocity.copy()
    )

    estimated_quaternion = (
        self.esekf.quaternion.copy()
    )

    # Ground-truth position tại t_k+1
    true_position = (
        self.data.site_xpos[
            self.esekf.site_id
        ].copy()
    )

    # Ground-truth quaternion tại t_k+1
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

    # q và -q biểu diễn cùng orientation
    if (
        np.dot(
            true_quaternion,
            estimated_quaternion,
        ) < 0.0
    ):
        true_quaternion *= -1.0

    # Lưu một transition:
    #
    # IMU_k
    #   -> predict
    # x_hat_(k+1)
    #
    # rồi so với GT_(k+1)
    self.data_logger.log(
        sample_time=float(self.data.time),

        acceleration=acceleration,
        angular_velocity=angular_velocity,

        gt_position=true_position,
        gt_velocity=true_velocity,
        gt_quaternion=true_quaternion,

        est_position=estimated_position,
        est_velocity=estimated_velocity,
        est_quaternion=estimated_quaternion,

        correction_applied=zupt_active,

        covariance=self.esekf.P.copy(),
    )

    # Plot ESEKF prediction vs GT tại t_k+1
    self.imu_trajectory.update(
        estimated_position,
        estimated_velocity,
        estimated_quaternion,
    )

  # TEST_TIMING_SPLIT END


if __name__ == "__main__":
  # Make run.main() instantiate the extended controller without editing run.py.
  run.G1Controller = IMUG1Controller

  # The option defined by run.py is plural: --no-cameras.
  if "--no-cameras" not in sys.argv:
    sys.argv.append("--no-cameras")

  run.main()