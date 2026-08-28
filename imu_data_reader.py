"""Read and buffer accelerometer and gyroscope data from a MuJoCo IMU site."""

from __future__ import annotations

from collections import deque

import mujoco
import numpy as np


class IMUDataReader:
  """Collect 3-axis accelerometer and gyroscope samples from MuJoCo."""

  def __init__(
      self,
      model: mujoco.MjModel,
      data: mujoco.MjData,
      accelerometer_name: str = "imu_right_acc",
      gyroscope_name: str = "imu_right_gyro",
      print_hz: float = 5.0,
      max_samples: int = 20_000,
  ) -> None:
    if print_hz < 0:
      raise ValueError(
        "print_hz must be greater than or equal to zero"
      )

    self.model = model
    self.data = data
    self.accelerometer_name = accelerometer_name
    self.gyroscope_name = gyroscope_name

    self.print_period = (
        1.0 / print_hz
        if print_hz > 0
        else np.inf
    )

    self.last_print_time = -np.inf
    self.last_sample_time: float | None = None

    # VN-100 noise tại tần số lấy mẫu 200 Hz
    self.acc_noise_std = 0.0194162   # [m/s^2]
    self.gyro_noise_std = 0.0008639  # [rad/s]

    # Bộ sinh nhiễu ngẫu nhiên với seed cố định
    self.rng = np.random.default_rng(42)

    self._validate_sensor(
        accelerometer_name
    )

    self._validate_sensor(
        gyroscope_name
    )

    self.times: deque[float] = deque(
        maxlen=max_samples
    )

    self.accelerations: deque[np.ndarray] = deque(
        maxlen=max_samples
    )

    self.angular_velocities: deque[np.ndarray] = deque(
        maxlen=max_samples
    )

    # Tín hiệu IMU sau khi cộng nhiễu
    self.latest_acceleration = np.zeros(
        3,
        dtype=float,
    )

    self.latest_angular_velocity = np.zeros(
        3,
        dtype=float,
    )

    # Tín hiệu lý tưởng từ MuJoCo trước khi cộng nhiễu.
    # Accelerometer và gyro đều biểu diễn trong hệ trục site/IMU.
    self.latest_ground_truth_acceleration = np.zeros(
        3,
        dtype=float,
    )

    self.latest_ground_truth_angular_velocity = np.zeros(
        3,
        dtype=float,
    )

  def update(
      self,
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Đọc một mẫu IMU và cộng nhiễu VN-100.

    Returns
    -------
    acceleration:
        Gia tốc IMU sau khi cộng nhiễu [m/s^2].

    angular_velocity:
        Vận tốc góc IMU sau khi cộng nhiễu [rad/s].
    """

    sample_time = float(
        self.data.time
    )

    # mj_resetData() đưa thời gian về 0.
    # Khi phát hiện reset thì xóa bộ đệm cũ.
    if (
        self.last_sample_time is not None
        and sample_time < self.last_sample_time
    ):
      self.clear()

    # ========================================================
    # Ground truth: đầu ra sensor MuJoCo chưa cộng nhiễu
    # ========================================================

    ground_truth_acceleration = (
        self.data.sensor(
            self.accelerometer_name
        ).data.copy()
    )

    ground_truth_angular_velocity = (
        self.data.sensor(
            self.gyroscope_name
        ).data.copy()
    )

    # ========================================================
    # Cộng nhiễu trắng theo thông số VN-100
    # ========================================================

    acceleration_noise = self.rng.normal(
        loc=0.0,
        scale=self.acc_noise_std,
        size=3,
    )

    angular_velocity_noise = self.rng.normal(
        loc=0.0,
        scale=self.gyro_noise_std,
        size=3,
    )

    acceleration = (
        ground_truth_acceleration
        + acceleration_noise
    )

    angular_velocity = (
        ground_truth_angular_velocity
        + angular_velocity_noise
    )

    # ========================================================
    # Lưu ground truth và dữ liệu có nhiễu
    # ========================================================

    self.latest_ground_truth_acceleration = (
        ground_truth_acceleration.copy()
    )

    self.latest_ground_truth_angular_velocity = (
        ground_truth_angular_velocity.copy()
    )

    self.latest_acceleration = (
        acceleration.copy()
    )

    self.latest_angular_velocity = (
        angular_velocity.copy()
    )

    self.times.append(
        sample_time
    )

    self.accelerations.append(
        acceleration.copy()
    )

    self.angular_velocities.append(
        angular_velocity.copy()
    )

    self.last_sample_time = sample_time

    if (
        sample_time
        - self.last_print_time
        >= self.print_period
    ):
      self.last_print_time = sample_time

      # Bỏ comment nếu muốn in dữ liệu IMU
      # self._print_latest(
      #     sample_time,
      #     acceleration,
      #     angular_velocity,
      # )

    return (
        acceleration.copy(),
        angular_velocity.copy(),
    )

  def clear(self) -> None:
    """Xóa dữ liệu IMU đã lưu trong bộ đệm."""

    self.times.clear()
    self.accelerations.clear()
    self.angular_velocities.clear()

    self.latest_acceleration.fill(
        0.0
    )

    self.latest_angular_velocity.fill(
        0.0
    )

    self.latest_ground_truth_acceleration.fill(
        0.0
    )

    self.latest_ground_truth_angular_velocity.fill(
        0.0
    )

    self.last_print_time = -np.inf
    self.last_sample_time = None

  def as_arrays(
      self,
  ) -> tuple[
      np.ndarray,
      np.ndarray,
      np.ndarray,
  ]:
    """Trả dữ liệu trong bộ đệm dưới dạng các mảng NumPy."""

    times = np.asarray(
        self.times,
        dtype=float,
    )

    accelerations = np.asarray(
        self.accelerations,
        dtype=float,
    ).reshape(-1, 3)

    angular_velocities = np.asarray(
        self.angular_velocities,
        dtype=float,
    ).reshape(-1, 3)

    return (
        times,
        accelerations,
        angular_velocities,
    )

  def _validate_sensor(
      self,
      sensor_name: str,
  ) -> None:
    """Kiểm tra sensor có tồn tại và có đúng 3 trục."""

    sensor_id = mujoco.mj_name2id(
        self.model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        sensor_name,
    )

    if sensor_id < 0:
      raise ValueError(
          f"Không tìm thấy sensor MuJoCo: "
          f"{sensor_name!r}"
      )

    sensor_dimension = int(
        self.model.sensor_dim[
            sensor_id
        ]
    )

    if sensor_dimension != 3:
      raise ValueError(
          f"Sensor {sensor_name!r} phải có 3 trục, "
          f"nhưng hiện có dimension="
          f"{sensor_dimension}"
      )

  @staticmethod
  def _print_latest(
      sample_time: float,
      acceleration: np.ndarray,
      angular_velocity: np.ndarray,
  ) -> None:
    """In mẫu IMU mới nhất."""

    acc_text = ", ".join(
        f"{value: .5f}"
        for value in acceleration
    )

    gyro_text = ", ".join(
        f"{value: .6f}"
        for value in angular_velocity
    )

    print(
        f"[IMU t={sample_time:7.3f} s] "
        f"acc [m/s^2]=[{acc_text}] | "
        f"gyro [rad/s]=[{gyro_text}]"
    )