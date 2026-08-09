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
    self.velocities: deque[np.ndarray] = deque(maxlen=max_points)
    self.quaternions: deque[np.ndarray] = deque(maxlen=max_points)

    # Trạng thái danh định sau bước dự đoán ESEKF
    self.estimated_positions: deque[np.ndarray] = deque(maxlen=max_points)
    self.estimated_velocities: deque[np.ndarray] = deque(maxlen=max_points)
    self.estimated_quaternions: deque[np.ndarray] = deque(maxlen=max_points)

    plt.ion()
    self.figure = plt.figure(
      "IMU site trajectory", figsize=(12, 7), constrained_layout=True
    )
    self.ax_3d = self.figure.add_subplot(2, 2, 1, projection="3d")
    self.ax_time = self.figure.add_subplot(2, 2, 2)
    self.ax_velocity = self.figure.add_subplot(2, 2, 3)
    self.ax_quaternion = self.figure.add_subplot(2, 2, 4)

    (self.path_line,) = self.ax_3d.plot(
      [], [], [], color="tab:red", linewidth=1.5, label=site_name
    )
    (self.estimated_path_line,) = self.ax_3d.plot(
        [], [], [],
        color="tab:orange",
        linestyle="--",
        linewidth=1.5,
        label="ESEKF estimate",
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
    self.estimated_coordinate_lines = [
        self.ax_time.plot(
            [], [],
            label=f"{axis}",
            color=color,
            linestyle="--",
            linewidth=1.2,
        )[0]
        for axis, color in zip(
            ("x-", "y-", "z-"),
            ("red", "green", "blue"),
        )
    ]

    self.velocity_lines = [
        self.ax_velocity.plot(
            [], [],
            label=axis,
            color=color,
            linewidth=1.2,
        )[0]
        for axis, color in zip(
            ("vx", "vy", "vz"),
            ("red", "green", "blue"),
        )
    ]
    self.estimated_velocity_lines = [
        self.ax_velocity.plot(
            [], [],
            label=f"{axis}",
            color=color,
            linestyle="--",
            linewidth=1.2,
        )[0]
        for axis, color in zip(
            ("vx-", "vy-", "vz-"),
            ("red", "green", "blue"),
        )
    ]

    self.quaternion_lines = [
        self.ax_quaternion.plot(
            [], [],
            label=component,
            color=color,
            linewidth=1.2,
        )[0]
        for component, color in zip(
            ("qw", "qx", "qy", "qz"),
            ("black", "red", "green", "blue"),
        )
    ]
    self.estimated_quaternion_lines = [
        self.ax_quaternion.plot(
            [], [],
            label=f"{component}",
            color=color,
            linestyle="--",
            linewidth=1.2,
        )[0]
        for component, color in zip(
            ("qw-", "qx-", "qy-", "qz-"),
            ("black", "red", "green", "blue"),
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

    self.ax_velocity.set_title("Vận tốc IMU trong hệ world")
    self.ax_velocity.set_xlabel("Thời gian [s]")
    self.ax_velocity.set_ylabel("Vận tốc [m/s]")
    self.ax_velocity.grid(True, alpha=0.3)
    self.ax_velocity.legend(loc="upper right")

    self.ax_quaternion.set_title("Quaternion IMU → world")
    self.ax_quaternion.set_xlabel("Thời gian [s]")
    self.ax_quaternion.set_ylabel("Giá trị quaternion")
    self.ax_quaternion.set_ylim(-1.05, 1.05)
    self.ax_quaternion.grid(True, alpha=0.3)
    self.ax_quaternion.legend(loc="upper right")

    self.figure.show()
    atexit.register(self.close)

  def update(
      self,
      estimated_position: np.ndarray,
      estimated_velocity: np.ndarray,
      estimated_quaternion: np.ndarray,
  ) -> None:
    """Record one sample and refresh the plots at the configured plot rate."""
    if not plt.fignum_exists(self.figure.number):
      return

    sample_time = float(self.data.time)

    # mj_resetData() resets data.time to zero. Start a new trajectory afterward.
    if self.last_sample_time is not None and sample_time < self.last_sample_time:
      self.clear()

    self.times.append(sample_time)
    self.positions.append(self.data.site_xpos[self.site_id].copy())

    # Vận tốc site trong hệ world
    site_velocity_6d = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(
      self.model,
      self.data,
      mujoco.mjtObj.mjOBJ_SITE,
      self.site_id,
      site_velocity_6d,
      0,  # 0: biểu diễn trong hệ world
    )
    # Ba phần tử đầu là vận tốc góc,
    # ba phần tử sau là vận tốc tuyến tính
    self.velocities.append(site_velocity_6d[3:6].copy())

    # Quaternion quay từ frame IMU sang world [w, x, y, z]
    site_quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(
        site_quaternion,
        self.data.site_xmat[self.site_id],
    )
    # Tránh quaternion bị đổi dấu khi vẽ:
    # q và -q biểu diễn cùng một phép quay
    if (
        self.quaternions
        and np.dot(site_quaternion, self.quaternions[-1]) < 0.0
    ):
      site_quaternion *= -1.0
    self.quaternions.append(site_quaternion.copy())

    # Lưu trạng thái danh định sau bước dự đoán ESEKF
    estimated_position = np.asarray(
        estimated_position, dtype=float
    ).reshape(3)

    estimated_velocity = np.asarray(
        estimated_velocity, dtype=float
    ).reshape(3)

    estimated_quaternion = np.asarray(
        estimated_quaternion, dtype=float
    ).reshape(4)

    # Chuẩn hóa quaternion ước lượng
    quaternion_norm = np.linalg.norm(estimated_quaternion)
    if quaternion_norm > 0.0:
      estimated_quaternion = estimated_quaternion / quaternion_norm

    # Tránh quaternion ước lượng bị đổi dấu trên đồ thị
    if (
        self.estimated_quaternions
        and np.dot(
            estimated_quaternion,
            self.estimated_quaternions[-1],
        ) < 0.0
    ):
      estimated_quaternion *= -1.0

    self.estimated_positions.append(estimated_position.copy())
    self.estimated_velocities.append(estimated_velocity.copy())
    self.estimated_quaternions.append(estimated_quaternion.copy())

    self.last_sample_time = sample_time

    if sample_time - self.last_plot_time < self.plot_period:
      return

    self.last_plot_time = sample_time
    self._refresh_plot()

  def clear(self) -> None:
    """Clear all collected samples, for example after resetting the robot."""
    self.times.clear()
    self.positions.clear()
    self.velocities.clear()
    self.quaternions.clear()
    self.estimated_positions.clear()
    self.estimated_velocities.clear()
    self.estimated_quaternions.clear()
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
    velocities = np.asarray(self.velocities, dtype=float)
    quaternions = np.asarray(self.quaternions, dtype=float)
    estimated_positions = np.asarray(self.estimated_positions, dtype=float)
    estimated_velocities = np.asarray(self.estimated_velocities, dtype=float)
    estimated_quaternions = np.asarray(self.estimated_quaternions, dtype=float)
    x, y, z = positions.T
    estimated_x, estimated_y, estimated_z = estimated_positions.T

    self.path_line.set_data_3d(x, y, z)
    self.estimated_path_line.set_data_3d(estimated_x, estimated_y, estimated_z,)
    self.current_point.set_data_3d([x[-1]], [y[-1]], [z[-1]])
    all_positions = np.vstack((positions, estimated_positions))
    self._set_3d_limits(all_positions)

    for axis_index, line in enumerate(self.coordinate_lines):
      line.set_data(times, positions[:, axis_index])

    for axis_index, line in enumerate(
        self.estimated_coordinate_lines
    ):
      line.set_data(
          times,
          estimated_positions[:, axis_index],
      )

    for axis_index, line in enumerate(self.velocity_lines):
      line.set_data(times, velocities[:, axis_index])

    for axis_index, line in enumerate(
        self.estimated_velocity_lines
    ):
      line.set_data(
          times,
          estimated_velocities[:, axis_index],
      )

    for component_index, line in enumerate(self.quaternion_lines):
      line.set_data(times, quaternions[:, component_index])

    for component_index, line in enumerate(
        self.estimated_quaternion_lines
    ):
      line.set_data(
          times,
          estimated_quaternions[:, component_index],
      )

    time_padding = max(0.1, 0.02 * max(times[-1] - times[0], 1.0))
    self.ax_time.set_xlim(times[0] - time_padding, times[-1] + time_padding)

    self.ax_velocity.set_xlim(
        times[0] - time_padding,
        times[-1] + time_padding,
    )

    self.ax_quaternion.set_xlim(
        times[0] - time_padding,
        times[-1] + time_padding,
    )

    position_values = np.vstack((positions, estimated_positions))
    position_min = float(np.min(position_values))
    position_max = float(np.max(position_values))
    position_padding = max(0.05, 0.05 * (position_max - position_min))
    self.ax_time.set_ylim(
      position_min - position_padding, position_max + position_padding
    )

    velocity_values = np.vstack((velocities, estimated_velocities))
    velocity_min = float(np.min(velocity_values))
    velocity_max = float(np.max(velocity_values))
    velocity_padding = max(
        0.05,
        0.05 * (velocity_max - velocity_min),
    )
    self.ax_velocity.set_ylim(
        velocity_min - velocity_padding,
        velocity_max + velocity_padding,
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
