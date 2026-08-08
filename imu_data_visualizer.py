"""Live plots of 3-axis accelerometer and gyroscope data."""

from __future__ import annotations

import atexit
from collections import deque

import matplotlib.pyplot as plt
import numpy as np


class IMUDataVisualizer:
  """Display acceleration and angular velocity versus simulation time."""

  def __init__(
      self,
      plot_fps: float = 10.0,
      max_points: int = 20_000,
  ) -> None:
    if plot_fps <= 0:
      raise ValueError("plot_fps must be greater than zero")

    self.plot_period = 1.0 / plot_fps
    self.last_plot_time = -np.inf
    self.last_sample_time: float | None = None

    self.times: deque[float] = deque(maxlen=max_points)
    self.accelerations: deque[np.ndarray] = deque(maxlen=max_points)
    self.angular_velocities: deque[np.ndarray] = deque(maxlen=max_points)

    plt.ion()
    self.figure, (self.ax_acc, self.ax_gyro) = plt.subplots(
      2,
      1,
      num="IMU data",
      figsize=(12, 7),
      sharex=True,
      constrained_layout=False,
    )
    self.figure.subplots_adjust(
      left=0.10,
      right=0.97,
      bottom=0.10,
      top=0.93,
      hspace=0.28,
    )

    axis_names = ("x", "y", "z")
    axis_colors = ("red", "green", "blue")

    self.acc_lines = [
      self.ax_acc.plot(
        [], [], label=f"a_{axis}", color=color, linewidth=1.2
      )[0]
      for axis, color in zip(axis_names, axis_colors)
    ]
    self.gyro_lines = [
      self.ax_gyro.plot(
        [], [], label=f"omega_{axis}", color=color, linewidth=1.2
      )[0]
      for axis, color in zip(axis_names, axis_colors)
    ]

    self.ax_acc.set_title("Dữ liệu accelerometer theo thời gian")
    self.ax_acc.set_ylabel("Gia tốc [m/s²]")
    self.ax_acc.grid(True, alpha=0.3)
    self.ax_acc.legend(loc="upper right", ncol=3)

    self.ax_gyro.set_title("Dữ liệu gyroscope theo thời gian")
    self.ax_gyro.set_xlabel("Thời gian [s]")
    self.ax_gyro.set_ylabel("Vận tốc góc [rad/s]")
    self.ax_gyro.grid(True, alpha=0.3)
    self.ax_gyro.legend(loc="upper right", ncol=3)

    self.figure.show()
    atexit.register(self.close)

  def update(
      self,
      sample_time: float,
      acceleration: np.ndarray,
      angular_velocity: np.ndarray,
  ) -> None:
    """Store one IMU sample and refresh the plots at the configured rate."""
    if not plt.fignum_exists(self.figure.number):
      return

    sample_time = float(sample_time)

    # mj_resetData() returns simulation time to zero.
    if self.last_sample_time is not None and sample_time < self.last_sample_time:
      self.clear()

    acceleration = np.asarray(acceleration, dtype=float).reshape(3).copy()
    angular_velocity = np.asarray(
      angular_velocity, dtype=float
    ).reshape(3).copy()

    self.times.append(sample_time)
    self.accelerations.append(acceleration)
    self.angular_velocities.append(angular_velocity)
    self.last_sample_time = sample_time

    if sample_time - self.last_plot_time < self.plot_period:
      return

    self.last_plot_time = sample_time
    self._refresh_plot()

  def clear(self) -> None:
    """Clear all collected samples, for example after resetting the robot."""
    self.times.clear()
    self.accelerations.clear()
    self.angular_velocities.clear()
    self.last_plot_time = -np.inf
    self.last_sample_time = None

  def close(self) -> None:
    """Close the Matplotlib window safely."""
    if hasattr(self, "figure") and plt.fignum_exists(self.figure.number):
      plt.close(self.figure)

  def _refresh_plot(self) -> None:
    if not self.times:
      return

    times = np.asarray(self.times, dtype=float)
    accelerations = np.asarray(self.accelerations, dtype=float)
    angular_velocities = np.asarray(self.angular_velocities, dtype=float)

    for axis_index, line in enumerate(self.acc_lines):
      line.set_data(times, accelerations[:, axis_index])

    for axis_index, line in enumerate(self.gyro_lines):
      line.set_data(times, angular_velocities[:, axis_index])

    time_padding = max(0.1, 0.02 * max(times[-1] - times[0], 1.0))
    self.ax_gyro.set_xlim(
      times[0] - time_padding,
      times[-1] + time_padding,
    )
    self._set_y_limits(self.ax_acc, accelerations)
    self._set_y_limits(self.ax_gyro, angular_velocities)

    self.figure.canvas.draw_idle()
    self.figure.canvas.flush_events()

  @staticmethod
  def _set_y_limits(axis, values: np.ndarray) -> None:
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    value_span = value_max - value_min
    padding = max(1e-6, 0.05 * value_span)

    if value_span < 1e-12:
      padding = max(0.1, 0.05 * abs(value_min))

    axis.set_ylim(value_min - padding, value_max + padding)