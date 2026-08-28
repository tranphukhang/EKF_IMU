"""Log synchronized IMU, ground-truth, and ESEKF data to CSV."""

from __future__ import annotations

import atexit
from datetime import datetime
from pathlib import Path

import numpy as np


class SimulationDataLogger:

    def __init__(
        self,
        output_path: str | Path | None = None,
    ) -> None:

        if output_path is None:
            run_timestamp = datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )

            output_path = (
                Path(__file__).resolve().parent
                / "logs"
                / run_timestamp
                / "data"
                / "esekf_simulation.csv"
            )

        self.output_path = Path(
            output_path
        )

        self.rows: list[np.ndarray] = []
        self.saved = False

        self.header = [
            # ==================================================
            # Time
            # ==================================================
            "time",

            # ==================================================
            # IMU acceleration có nhiễu [m/s^2]
            # ==================================================
            "imu_ax",
            "imu_ay",
            "imu_az",

            # ==================================================
            # IMU angular velocity có nhiễu [rad/s]
            # ==================================================
            "imu_gx",
            "imu_gy",
            "imu_gz",

            # ==================================================
            # Ground-truth accelerometer output [m/s^2]
            # Noise-free MuJoCo sensor output
            # Hệ tọa độ site/IMU
            # ==================================================
            "gt_imu_ax",
            "gt_imu_ay",
            "gt_imu_az",

            # ==================================================
            # Ground-truth gyroscope output [rad/s]
            # Noise-free MuJoCo sensor output
            # Hệ tọa độ site/IMU
            # ==================================================
            "gt_imu_gx",
            "gt_imu_gy",
            "gt_imu_gz",

            # ==================================================
            # Ground-truth position [m]
            # Hệ tọa độ world
            # ==================================================
            "gt_px",
            "gt_py",
            "gt_pz",

            # ==================================================
            # Ground-truth velocity [m/s]
            # Hệ tọa độ world
            # ==================================================
            "gt_vx",
            "gt_vy",
            "gt_vz",

            # ==================================================
            # Ground-truth quaternion IMU -> world
            # ==================================================
            "gt_qw",
            "gt_qx",
            "gt_qy",
            "gt_qz",

            # ==================================================
            # ESEKF estimated position [m]
            # ==================================================
            "est_px",
            "est_py",
            "est_pz",

            # ==================================================
            # ESEKF estimated velocity [m/s]
            # ==================================================
            "est_vx",
            "est_vy",
            "est_vz",

            # ==================================================
            # ESEKF estimated quaternion IMU -> world
            # ==================================================
            "est_qw",
            "est_qx",
            "est_qy",
            "est_qz",

            # ==================================================
            # Correction flag
            # ==================================================
            "correction_applied",

            # ==================================================
            # Diagonal của ESEKF error-state covariance P
            # ==================================================
            "P_dpx",
            "P_dpy",
            "P_dpz",

            "P_dvx",
            "P_dvy",
            "P_dvz",

            "P_dtheta_x",
            "P_dtheta_y",
            "P_dtheta_z",
        ]

        # Nếu chương trình kết thúc bất thường hoặc user đóng viewer,
        # vẫn cố gắng lưu dữ liệu đã thu được.
        atexit.register(
            self.save
        )

    def log(
        self,
        sample_time: float,
        acceleration: np.ndarray,
        angular_velocity: np.ndarray,
        ground_truth_acceleration: np.ndarray,
        ground_truth_angular_velocity: np.ndarray,
        gt_position: np.ndarray,
        gt_velocity: np.ndarray,
        gt_quaternion: np.ndarray,
        est_position: np.ndarray,
        est_velocity: np.ndarray,
        est_quaternion: np.ndarray,
        correction_applied: bool,
        covariance: np.ndarray,
    ) -> None:

        acceleration = np.asarray(
            acceleration,
            dtype=float,
        ).reshape(3)

        angular_velocity = np.asarray(
            angular_velocity,
            dtype=float,
        ).reshape(3)

        ground_truth_acceleration = np.asarray(
            ground_truth_acceleration,
            dtype=float,
        ).reshape(3)

        ground_truth_angular_velocity = np.asarray(
            ground_truth_angular_velocity,
            dtype=float,
        ).reshape(3)

        gt_position = np.asarray(
            gt_position,
            dtype=float,
        ).reshape(3)

        gt_velocity = np.asarray(
            gt_velocity,
            dtype=float,
        ).reshape(3)

        gt_quaternion = np.asarray(
            gt_quaternion,
            dtype=float,
        ).reshape(4)

        est_position = np.asarray(
            est_position,
            dtype=float,
        ).reshape(3)

        est_velocity = np.asarray(
            est_velocity,
            dtype=float,
        ).reshape(3)

        est_quaternion = np.asarray(
            est_quaternion,
            dtype=float,
        ).reshape(4)

        covariance = np.asarray(
            covariance,
            dtype=float,
        ).reshape(9, 9)

        # Chỉ log diagonal của P
        P_diag = np.diag(
            covariance
        )

        row = np.concatenate([
            np.array([
                sample_time
            ]),

            # IMU có nhiễu
            acceleration,
            angular_velocity,

            # IMU ground truth
            ground_truth_acceleration,
            ground_truth_angular_velocity,

            # Ground-truth state
            gt_position,
            gt_velocity,
            gt_quaternion,

            # ESEKF estimated state
            est_position,
            est_velocity,
            est_quaternion,

            # Correction flag
            np.array([
                1.0
                if correction_applied
                else 0.0
            ]),

            # Covariance diagonal
            P_diag,
        ])

        if row.size != len(
            self.header
        ):
            raise RuntimeError(
                f"Logger expected "
                f"{len(self.header)} values, "
                f"but received {row.size}"
            )

        self.rows.append(
            row
        )

    def save(self) -> None:

        if self.saved:
            return

        if not self.rows:
            return

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = np.vstack(
            self.rows
        )

        np.savetxt(
            self.output_path,
            data,
            delimiter=",",
            header=",".join(
                self.header
            ),
            comments="",
            fmt="%.10e",
        )

        self.saved = True

        print("\n=== DATA LOGGER ===")

        print(
            f"Saved file: "
            f"{self.output_path}"
        )

        print(
            f"Samples   : "
            f"{data.shape[0]}"
        )

        print(
            f"Start time: "
            f"{data[0, 0]:.4f} s"
        )

        print(
            f"End time  : "
            f"{data[-1, 0]:.4f} s"
        )

        print("===================\n")