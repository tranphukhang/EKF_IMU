import mujoco
import numpy as np


class ZUPTTrigger:
    def __init__(
        self,
        model,
        data,
        site_name="imu_right_foot",
        velocity_threshold=0.05,
        print_hz=5.0,
    ):
        self.velocity_threshold = velocity_threshold
        self.model = model
        self.data = data

        self.site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if self.site_id < 0:
            raise ValueError(f"Không tìm thấy site: {site_name}")

        self.ground_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "ground",
        )

        self.right_foot_geom_ids = {
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"right_foot{i}_collision",
            )
            for i in range(1, 8)
        }

        self.print_period = 1.0 / print_hz
        self.last_print_time = -np.inf


    def right_foot_ground_contact(self) -> bool:
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            if geom1 == self.ground_geom_id and geom2 in self.right_foot_geom_ids:
                return True

            if geom2 == self.ground_geom_id and geom1 in self.right_foot_geom_ids:
                return True

        return False

    def get_true_linear_velocity(self) -> np.ndarray:
        """Ground-truth linear velocity của IMU site trong world frame."""

        site_velocity_6d = np.zeros(6, dtype=float)

        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            self.site_id,
            site_velocity_6d,
            0,  # world frame
        )

        return site_velocity_6d[3:6].copy()

    def check(self) -> bool:

        true_linear_velocity = (
            self.get_true_linear_velocity()
        )

        speed = np.linalg.norm(
            true_linear_velocity
        )

        right_foot_contact = (
            self.right_foot_ground_contact()
        )

        zupt_candidate = (
            right_foot_contact
            and speed < self.velocity_threshold
        )


        current_time = float(self.data.time)

        if current_time < self.last_print_time:
            self.last_print_time = -np.inf

        if current_time - self.last_print_time >= self.print_period:
            # print(
            #     f"[ZUPT CHECK] "
            #     f"t = {current_time:.3f} s | "
            #     f"contact = {right_foot_contact} | "
            #     f"|v_true| = {speed:.6f} m/s | "
            #     f"candidate = {zupt_candidate}"
            # )
            self.last_print_time = current_time

        return zupt_candidate