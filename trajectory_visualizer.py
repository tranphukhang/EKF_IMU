"""Live visualization of a MuJoCo site's world-frame trajectory."""

from __future__ import annotations

import atexit
from collections import deque

import matplotlib.pyplot as plt
import mujoco
import numpy as np


class SiteTrajectoryVisualizer:
  """Collect and display a site's 3-D path and position versus time."""

  def __init__(
      self,
      model: mujoco.MjModel,
      data: mujoco.MjData,
      site_name: str = "imu_right_foot",
      plot_fps: float = 10.0,
      max_points: int = 20_000,
  ) -> None:
    self.model = model
    self.data = data
    self.site_name = site_name
    self.plot_period = 1.0 / plot_fps
    self.last_plot_time = -np.inf
    self.last_sample_time: float | None = None

    self.site_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_SITE, site_name
    )
    if self.site_id < 0:
      raise ValueError(f"Không tìm thấy site MuJoCo: {site_name!r}")

    self.times: deque[float] = deque(maxlen=max_points)
    self.positions: deque[np.ndarray] = deque(maxlen=max_points)

    plt.ion()
    self.figure = plt.figure(
      "IMU site trajectory", figsize=(14, 7), constrained_layout=True
    )
    self.ax_3d = self.figure.add_subplot(1, 2, 1, projection="3d")
    self.ax_time = self.figure.add_subplot(1, 2, 2)

    (self.path_line,) = self.ax_3d.plot(
      [], [], [], color="tab:red", linewidth=1.5, label=site_name
    )
    (self.current_point,) = self.ax_3d.plot(
      [], [], [], marker="o", color="black", markersize=5
    )

    self.coordinate_lines = [
        self.ax_time.plot([], [], label=axis, color=color, linewidth=1.2)[0]
        for axis, color in zip(
            ("x", "y", "z"),
            ("red", "green", "blue")
        )
    ]

    self.ax_3d.set_title("Quỹ đạo imu trong hệ world")
    self.ax_3d.set_xlabel("X [m]")
    self.ax_3d.set_ylabel("Y [m]")
    self.ax_3d.set_zlabel("Z [m]")
    self.ax_3d.legend(loc="upper right")

    self.ax_time.set_title("Vị trí imu theo thời gian")
    self.ax_time.set_xlabel("Thời gian [s]")
    self.ax_time.set_ylabel("Vị trí [m]")
    self.ax_time.grid(True, alpha=0.3)
    self.ax_time.legend(loc="upper right")

    self.figure.show()
    atexit.register(self.close)

  def update(self) -> None:
    """Record one sample and refresh the plots at the configured plot rate."""
    if not plt.fignum_exists(self.figure.number):
      return

    sample_time = float(self.data.time)

    # mj_resetData() resets data.time to zero. Start a new trajectory afterward.
    if self.last_sample_time is not None and sample_time < self.last_sample_time:
      self.clear()

    self.times.append(sample_time)
    self.positions.append(self.data.site_xpos[self.site_id].copy())
    self.last_sample_time = sample_time

    if sample_time - self.last_plot_time < self.plot_period:
      return

    self.last_plot_time = sample_time
    self._refresh_plot()

  def clear(self) -> None:
    """Clear all collected samples, for example after resetting the robot."""
    self.times.clear()
    self.positions.clear()
    self.last_plot_time = -np.inf
    self.last_sample_time = None

  def close(self) -> None:
    """Close the Matplotlib window safely."""
    if hasattr(self, "figure") and plt.fignum_exists(self.figure.number):
      plt.close(self.figure)

  def _refresh_plot(self) -> None:
    if not self.positions:
      return

    times = np.asarray(self.times, dtype=float)
    positions = np.asarray(self.positions, dtype=float)
    x, y, z = positions.T

    self.path_line.set_data_3d(x, y, z)
    self.current_point.set_data_3d([x[-1]], [y[-1]], [z[-1]])
    self._set_3d_limits(positions)

    for axis_index, line in enumerate(self.coordinate_lines):
      line.set_data(times, positions[:, axis_index])

    time_padding = max(0.1, 0.02 * max(times[-1] - times[0], 1.0))
    self.ax_time.set_xlim(times[0] - time_padding, times[-1] + time_padding)

    position_min = float(np.min(positions))
    position_max = float(np.max(positions))
    position_padding = max(0.05, 0.05 * (position_max - position_min))
    self.ax_time.set_ylim(
      position_min - position_padding, position_max + position_padding
    )

    self.figure.canvas.draw_idle()
    self.figure.canvas.flush_events()

  def _set_3d_limits(self, positions: np.ndarray) -> None:
    lower = positions.min(axis=0)
    upper = positions.max(axis=0)
    center = 0.5 * (lower + upper)
    half_span = max(0.1, 0.55 * float(np.max(upper - lower)))

    self.ax_3d.set_xlim(center[0] - half_span, center[0] + half_span)
    self.ax_3d.set_ylim(center[1] - half_span, center[1] + half_span)
    self.ax_3d.set_zlim(center[2] - half_span, center[2] + half_span)
