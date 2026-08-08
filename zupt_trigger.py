import mujoco
import numpy as np


class ZUPTTrigger:
    def __init__(
        self,
        model,
        data,
        site_name="imu_right_foot",
        print_hz=5.0,
    ):
        self.model = model
        self.data = data

        self.site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )

        if self.site_id < 0:
            raise ValueError(
                f"Không tìm thấy site: {site_name}"
            )

        self.print_period = 1.0 / print_hz
        self.last_print_time = -np.inf

    def check(self) -> bool:
        # Lấy vận tốc 6D ground truth của site IMU trong world frame
        site_velocity_6d = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            self.site_id,
            site_velocity_6d,
            0,  # world frame
        )
        # 3 phần tử cuối là vận tốc tuyến tính
        true_linear_velocity = site_velocity_6d[3:6]
        # Độ lớn vận tốc
        speed = np.linalg.norm(
            true_linear_velocity
        )

        current_time = float(self.data.time)

        if current_time < self.last_print_time:
            self.last_print_time = -np.inf

        if ( current_time - self.last_print_time >= self.print_period):
            print(
                f"[ZUPT CHECK] "
                f"t = {current_time:.3f} s | "
                f"|v_true| = {speed:.6f} m/s"
            )

            self.last_print_time = current_time

        return False