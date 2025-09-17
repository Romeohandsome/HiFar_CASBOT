import os
import sys
import glob
import yaml
import select
import argparse
import numpy as np
import time
import torch
import mujoco, mujoco.viewer
from utils.model import *


class MujocoRunner:

    def __init__(self):
        self.get_args()
        self.update_cfg_from_args()
        self.model = DenoisingRMA(
            self.cfg["env"]["num_actions"], self.cfg["env"]["num_observations"], self.cfg["runner"]["num_stack"], self.cfg["env"]["num_privileged_obs"], self.cfg["algorithm"]["num_embedding"]
        )
        self.load()
        #self.mj_model = mujoco.MjModel.from_xml_path("resources/T1/T1_serial_collision.xml")
        self.mj_model = mujoco.MjModel.from_xml_path("resources/T1/T1_serial_collision.xml")
        self.mj_model.opt.timestep = self.cfg["sim"]["dt"] / self.cfg["sim"]["substeps"]
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos = np.array([0, 0, 0.25, 0.7071, 0, 0.7071, 0, 0.2, -1.35, 0, -0.5, 0.2, 1.35, 0, 0.5, 0, -0.2, 0, 0, 0.4, -0.25, 0, -0.2, 0, 0, 0.4, -0.25, 0], dtype=np.float32)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--checkpoint", type=str, help="Path of model checkpoint to load. Overrides config file if provided.")
        self.args = parser.parse_args()

    def update_cfg_from_args(self):
        cfg_file = os.path.join("envs", "{}.yaml".format(self.args.task))
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        for arg in vars(self.args):
            if getattr(self.args, arg) is not None:
                self.cfg["basic"][arg] = getattr(self.args, arg)

    def load(self):
        if not self.cfg["basic"]["checkpoint"]:
            raise ValueError("Invalid model checkpoint")
        if (self.cfg["basic"]["checkpoint"] == "-1") or (self.cfg["basic"]["checkpoint"] == -1):
            self.cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
        print("Loading model from {}".format(self.cfg["basic"]["checkpoint"]))
        model_dict = torch.load(self.cfg["basic"]["checkpoint"], map_location="cpu", weights_only=True)
        self.model.load_state_dict(model_dict["model"])

    def play(self):
        default_dof_pos = np.array([0.2, -1.35, 0, -0.5, 0.2, 1.35, 0, 0.5, 0, -0.2, 0, 0, 0.4, -0.25, 0, -0.2, 0, 0, 0.4, -0.25, 0], dtype=np.float32)
        stiffness = np.array([200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 100, 100, 200, 200, 200, 200, 100, 100], dtype=np.float32)
        damping = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1], dtype=np.float32)
        # ctrl_limit = np.array([30, 30, 30, 30, 30, 30, 30, 30, 30, 45, 45, 30, 65, 24, 6, 45, 45, 30, 65, 24, 6], dtype=np.float32)
        ctrl_limit = np.array([24, 24, 24, 24, 24, 24, 24, 24, 30, 45, 45, 30, 65, 24, 6, 45, 45, 30, 65, 24, 6], dtype=np.float32)
        # ctrl_limit = np.array([18, 18, 18, 18, 18, 18, 18, 18, 30, 45, 45, 30, 65, 24, 6, 45, 45, 30, 65, 24, 6], dtype=np.float32)
        actions = np.zeros((self.cfg["env"]["num_actions"]), dtype=np.float32)
        dof_targets = np.zeros(default_dof_pos.shape, dtype=np.float32)
        stacked_obs = np.zeros((1, self.cfg["runner"]["num_stack"], self.cfg["env"]["num_observations"]), dtype=np.float32)
        effective_ids = [0, 3, 4, 7, 9, 12, 13, 15, 18, 19]

        it = 0
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            viewer.cam.elevation = -20
            while viewer.is_running():
                if it % 2500 == 2499:
                    body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")
                    external_force = 30000 * np.random.rand(2)
                    self.mj_data.xfrc_applied[body_id, :2] = external_force
                    print(f"Applied external force: {external_force} at step {it}")

                dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
                dof_vel = self.mj_data.qvel.astype(np.float32)[6:]
                quat = self.mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
                base_ang_vel = self.mj_data.sensor("angular-velocity").data.astype(np.float32)
                projected_gravity = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
                if it % (self.cfg["control"]["decimation"] * self.cfg["sim"]["substeps"]) == 0:
                    obs = np.zeros([1, self.cfg["env"]["num_observations"]], dtype=np.float32)
                    obs[0, 0:3] = projected_gravity * self.cfg["normalization"]["gravity"]
                    obs[0, 3:6] = base_ang_vel * self.cfg["normalization"]["ang_vel"]
                    obs[0, 6:16] = (dof_pos - default_dof_pos)[effective_ids] * self.cfg["normalization"]["dof_pos"]
                    obs[0, 16:26] = dof_vel[effective_ids] * self.cfg["normalization"]["dof_vel"]
                    obs[0, 26:36] = actions
                    if it == 0:
                        stacked_obs[:] = obs[:, np.newaxis, :]
                    stacked_obs[:, 1:, :] = stacked_obs[:, :-1, :]
                    stacked_obs[:, 0, :] = obs
                    mirrored_obs = self.mirror_obs(obs)
                    mirrored_stacked_obs = self.mirror_obs(stacked_obs)
                    dist, _ = self.model.act(torch.tensor(obs), stacked_obs=torch.tensor(stacked_obs))
                    mirrored_dist, _ = self.model.act(torch.tensor(mirrored_obs), stacked_obs=torch.tensor(mirrored_stacked_obs))
                    actions[:] = 0.5 * (dist.loc.detach().numpy() + self.mirror_act(mirrored_dist.loc.detach().numpy()))
                    actions = np.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
                    dof_targets[:] = default_dof_pos
                    dof_targets[effective_ids] += self.cfg["control"]["action_scale"] * actions
                self.mj_data.ctrl = np.clip(stiffness * (dof_targets - dof_pos) - damping * dof_vel, -ctrl_limit, ctrl_limit)
                mujoco.mj_step(self.mj_model, self.mj_data)
                viewer.cam.lookat[:] = self.mj_data.qpos.astype(np.float32)[0:3]
                viewer.sync()
                time.sleep(self.cfg["sim"]["dt"] / self.cfg["sim"]["substeps"])
                it += 1

    @staticmethod
    def quat_rotate_inverse(q, v):
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v) * 2.0)
        return a - b + c

    @staticmethod
    def mirror_obs(obs):
        mat = np.zeros((36, 36), dtype=np.float32)
        mat[ 0: 6,  0: 6] = np.eye(6)
        mat[ 6: 8,  8:10] = np.eye(2)
        mat[ 8:10,  6: 8] = np.eye(2)
        mat[10:13, 13:16] = np.eye(3)
        mat[13:16, 10:13] = np.eye(3)
        mat[16:18, 18:20] = np.eye(2)
        mat[18:20, 16:18] = np.eye(2)
        mat[20:23, 23:26] = np.eye(3)
        mat[23:26, 20:23] = np.eye(3)
        mat[26:28, 28:30] = np.eye(2)
        mat[28:30, 26:28] = np.eye(2)
        mat[30:33, 33:36] = np.eye(3)
        mat[33:36, 30:33] = np.eye(3)
        flip_val = np.ones(36, dtype=np.float32)
        inverse_ids = [  1,  3,  5,
                         7,  9, 
                        17, 19,
                        27, 29]
        flip_val[inverse_ids] = -1
        flip_mat = np.diag(flip_val)
        mirror_transform_mat = np.dot(mat, flip_mat)

        orig_shape = obs.shape
        reshaped_obs = obs.reshape(-1, 36)
        mirrored_obs = np.dot(reshaped_obs, mirror_transform_mat.T)
        mirrored_obs = mirrored_obs.reshape(orig_shape)
        return mirrored_obs

    @staticmethod
    def mirror_act(act):
        mat = np.zeros((10, 10), dtype=np.float32)
        mat[0:2, 2:4] = np.eye(2)
        mat[2:4, 0:2] = np.eye(2)
        mat[4:7, 7:10] = np.eye(3)
        mat[7:10, 4:7] = np.eye(3)
        flip_val = np.ones(10, dtype=np.float32)
        inverse_ids = [1, 3]
        flip_val[inverse_ids] = -1
        flip_mat = np.diag(flip_val)
        mirror_transform_mat = np.dot(mat, flip_mat)
        mirrored_act = np.dot(act, mirror_transform_mat.T)
        return mirrored_act


if __name__ == "__main__":
    runner = MujocoRunner()
    runner.play()
