import os
import sys
from typing import Dict

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch

import torch

import numpy as np
from .base_task import BaseTask

from utils.utils import apply_randomizations


class T1FallRecovery(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self._prepare_reward_function()

    def create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]
        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(asset_cfg["file"])
        asset_file = os.path.basename(asset_cfg["file"])

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg["control"]["drive_mode"]
        asset_options.collapse_fixed_joints = asset_cfg["collapse_fixed_joints"]
        asset_options.replace_cylinder_with_capsule = asset_cfg["replace_cylinder_with_capsule"]
        asset_options.flip_visual_attachments = asset_cfg["flip_visual_attachments"]
        asset_options.fix_base_link = asset_cfg["fix_base_link"]
        asset_options.density = asset_cfg["density"]
        asset_options.angular_damping = asset_cfg["angular_damping"]
        asset_options.linear_damping = asset_cfg["linear_damping"]
        asset_options.max_angular_velocity = asset_cfg["max_angular_velocity"]
        asset_options.max_linear_velocity = asset_cfg["max_linear_velocity"]
        asset_options.armature = asset_cfg["armature"]
        asset_options.thickness = asset_cfg["thickness"]
        asset_options.disable_gravity = asset_cfg["disable_gravity"]

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        print(f"Number of DOFs: {self.num_dofs}")
        print(f"Number of Bodies: {self.num_bodies}")
        print(f"DOF Names: {self.dof_names}")

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            self.dof_pos_limits[i, 0] = dof_props_asset["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props_asset["upper"][i].item()
            self.dof_vel_limits[i] = dof_props_asset["velocity"][i].item()
            self.torque_limits[i] = dof_props_asset["effort"][i].item()

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        penalized_contact_names = []
        for name in self.cfg["rewards"]["penalize_contacts_on"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg["rewards"]["terminate_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["base_name"])
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, penalized_contact_names[i])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, termination_contact_names[i])

        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        self.foot_indices = torch.zeros(len(asset_cfg["foot_names"]), dtype=torch.long, device=self.device, requires_grad=False)
        self.foot_shape_indices = []
        for i in range(len(asset_cfg["foot_names"])):
            indice = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["foot_names"][i])
            self.foot_indices[i] = indice
            self.foot_shape_indices += list(range(rbs_list[indice].start, rbs_list[indice].start + rbs_list[indice].count))

        base_init_state_list = self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"] + self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, asset_cfg["name"], i, asset_cfg["self_collisions"], 0)
            dof_props = self.gym.get_actor_dof_properties(env_handle, actor_handle)
            dof_props = self._process_dof_props(dof_props)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            shape_props = self._process_rigid_shape_props(shape_props)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

    def _process_dof_props(self, props):
        for i in range(self.num_dofs):
            props["driveMode"][i] = self.cfg["control"]["drive_mode"]
            dof_name = self.dof_names[i]
            found = False
            for name in self.cfg["control"]["stiffness"].keys():
                if name in dof_name:
                    props["stiffness"][i] = self.cfg["control"]["stiffness"][name]
                    props["damping"][i] = self.cfg["control"]["damping"][name]
                    found = True
            if not found:
                raise ValueError(f"PD gain of joint {dof_name} were not defined")
        props["stiffness"] = apply_randomizations(props["stiffness"], self.cfg["randomization"].get("stiffness"))
        props["damping"] = apply_randomizations(props["damping"], self.cfg["randomization"].get("damping"))
        return props

    def _process_rigid_body_props(self, props, i):
        for j in range(self.num_bodies):
            if j == self.base_indice:
                props[j].com.x, self.base_mass_scaled[i, 0] = apply_randomizations(props[j].com.x, self.cfg["randomization"].get("base_com"), return_noise=True)
                props[j].com.y, self.base_mass_scaled[i, 1] = apply_randomizations(props[j].com.y, self.cfg["randomization"].get("base_com"), return_noise=True)
                props[j].com.z, self.base_mass_scaled[i, 2] = apply_randomizations(props[j].com.z, self.cfg["randomization"].get("base_com"), return_noise=True)
                props[j].mass, self.base_mass_scaled[i, 3] = apply_randomizations(props[j].mass, self.cfg["randomization"].get("base_mass"), return_noise=True)
            else:
                props[j].com.x = apply_randomizations(props[j].com.x, self.cfg["randomization"].get("other_com"))
                props[j].com.y = apply_randomizations(props[j].com.y, self.cfg["randomization"].get("other_com"))
                props[j].com.z = apply_randomizations(props[j].com.z, self.cfg["randomization"].get("other_com"))
                props[j].mass = apply_randomizations(props[j].mass, self.cfg["randomization"].get("other_mass"))
            props[j].invMass = 1.0 / props[j].mass
        return props

    def _process_rigid_shape_props(self, props):
        for i in self.foot_shape_indices:
            props[i].friction = apply_randomizations(0.0, self.cfg["randomization"].get("friction"))
            props[i].compliance = apply_randomizations(0.0, self.cfg["randomization"].get("compliance"))
            props[i].restitution = apply_randomizations(0.0, self.cfg["randomization"].get("restitution"))
        return props

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if self.cfg["terrain"]["type"] == "plane":
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            spacing = self.cfg["env"]["env_spacing"]
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0
        else:
            num_cols = np.floor(np.sqrt(self.num_envs * self.terrain.env_length / self.terrain.env_width))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            self.env_origins[:, 0] = self.terrain.env_width / (num_rows + 1) * (xx.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 1] = self.terrain.env_length / (num_cols + 1) * (yy.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 2] = self.terrain.terrain_heights(self.env_origins)

    def _init_buffers(self):
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.extras["rew_terms"] = {}

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dofs)
        self.body_states = gymtorch.wrap_tensor(body_state).view(self.num_envs, self.num_bodies, 13)
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.body_states[:, self.foot_indices, 0:3]
        self.feet_quat = self.body_states[:, self.foot_indices, 3:7]

        # initialize some data used later on
        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()
        self.low_height_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.pushing_forces = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.feet_roll = torch.zeros(self.num_envs, len(self.foot_indices), dtype=torch.float, device=self.device)
        self.feet_yaw = torch.zeros(self.num_envs, len(self.foot_indices), dtype=torch.float, device=self.device)
        self.toe_pos = torch.zeros(self.num_envs, len(self.foot_indices), 3, dtype=torch.float, device=self.device)
        self.heel_pos = torch.zeros(self.num_envs, len(self.foot_indices), 3, dtype=torch.float, device=self.device)
        self.last_feet_pos = torch.zeros_like(self.feet_pos)
        self.last_toe_pos = torch.zeros_like(self.toe_pos)
        self.last_heel_pos = torch.zeros_like(self.heel_pos)
        self.toe_contact = torch.zeros(self.num_envs, len(self.foot_indices), dtype=torch.bool, device=self.device)
        self.heel_contact = torch.zeros(self.num_envs, len(self.foot_indices), dtype=torch.bool, device=self.device)
        self.dof_pos_ref = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.default_dof_pos = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["init_state"]["default_joint_angles"].keys():
                if name in self.dof_names[i]:
                    self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"][name]
                    found = True
            if not found:
                self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"]["default"]
        self.effective_ids = self.cfg["control"]["effective_ids"]
        self.num_references = self.cfg["init_state"]["reference_num"]
        self.reference_dof_pos = torch.zeros(self.num_references, self.num_dofs, dtype=torch.float, device=self.device)
        for j in range(self.num_references):
            for i in range(self.num_dofs):
                found = False
                for name in self.cfg["init_state"]["reference_joint_angles_" + str(j + 1)].keys():
                    if name in self.dof_names[i]:
                        self.reference_dof_pos[j, i] = self.cfg["init_state"]["reference_joint_angles_" + str(j + 1)][name]
                        found = True
                if not found:
                    self.reference_dof_pos[j, i] = self.cfg["init_state"]["reference_joint_angles_" + str(j + 1)]["default"]

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        self.reward_scales = self.cfg["rewards"]["scales"]
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

    def reset(self):
        """Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]
        self.episode_length_buf[env_ids] = 0
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.low_height_count[env_ids] = 0.0

        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.

        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs(self, env_ids):
        pos_num = self.cfg["init_state"]["pos_num"]
        dof_pos_mapping = self.cfg["init_state"]["dof_pos_mapping"]

        self.dof_pos[env_ids] = apply_randomizations(self.default_dof_pos, self.cfg["randomization"].get("init_dof_pos"))

        for mod_value, ref_idx in dof_pos_mapping.items():
            mask = (env_ids % pos_num == mod_value)
            idx = env_ids[mask]

            if ref_idx is not None:
                self.dof_pos[idx] = apply_randomizations(self.reference_dof_pos[ref_idx], self.cfg["randomization"].get("init_dof_pos"))

        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        pos_num = self.cfg["init_state"]["pos_num"]
        pos_height_mapping = self.cfg["init_state"]["pos_height_mapping"]

        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :2] += self.env_origins[env_ids, :2]
        self.root_states[env_ids, :2] = apply_randomizations(self.root_states[env_ids, :2], self.cfg["randomization"].get("init_base_pos_xy"))
        self.root_states[env_ids, 2] += self.terrain.terrain_heights(self.root_states[env_ids, :2])

        roll = torch.zeros(len(env_ids), device=self.device)
        pitch = torch.rand(len(env_ids), device=self.device) * 0.2 - 0.1

        for mod_value, params in pos_height_mapping.items():
            if params is None:
                continue

            mask = (env_ids % pos_num == mod_value)
            idx = env_ids[mask]

            if "height" in params:
                self.root_states[idx, 2] = params["height"] + self.terrain.terrain_heights(self.root_states[idx, :2])
            if "roll" in params:
                roll[torch.nonzero(mask, as_tuple=True)[0]] = params["roll"]
            if "pitch" in params:
                pitch[torch.nonzero(mask, as_tuple=True)[0]] = params["pitch"]

        yaw = torch.rand(len(env_ids), device=self.device) * (2 * np.pi)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

        self.root_states[env_ids, 7:9] = apply_randomizations(
            torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device, requires_grad=False),
            self.cfg["randomization"].get("init_base_lin_vel_xy"),
        )

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _teleport_robot(self):
        if self.terrain.type == "plane":
            return
        out_x_min = self.root_states[:, 0] < -0.75 * self.terrain.border_size
        out_x_max = self.root_states[:, 0] > self.terrain.env_width + 0.75 * self.terrain.border_size
        out_y_min = self.root_states[:, 1] < -0.75 * self.terrain.border_size
        out_y_max = self.root_states[:, 1] > self.terrain.env_length + 0.75 * self.terrain.border_size
        self.root_states[out_x_min, 0] += self.terrain.env_width + self.terrain.border_size
        self.root_states[out_x_max, 0] -= self.terrain.env_width + self.terrain.border_size
        self.root_states[out_y_min, 1] += self.terrain.env_length + self.terrain.border_size
        self.root_states[out_y_max, 1] -= self.terrain.env_length + self.terrain.border_size
        self.body_states[out_x_min, :, 0] += self.terrain.env_width + self.terrain.border_size
        self.body_states[out_x_max, :, 0] -= self.terrain.env_width + self.terrain.border_size
        self.body_states[out_y_min, :, 1] += self.terrain.env_length + self.terrain.border_size
        self.body_states[out_y_max, :, 1] -= self.terrain.env_length + self.terrain.border_size
        if out_x_min.any() or out_x_max.any() or out_y_min.any() or out_y_max.any():
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self._refresh_feet_state()

    def step(self, actions):
        self.actions[:] = torch.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        self.render()

        self.dof_pos_ref[:] = self.default_dof_pos
        dof_targets = self.dof_pos_ref.clone()
        dof_targets[:, self.effective_ids] += self.cfg["control"]["action_scale"] * self.actions

        for i in range(self.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.last_dof_targets))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel[:] = self.base_lin_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_lin_vel[:] * (1.0 - self.cfg["normalization"]["filter_weight"])
        self.filtered_ang_vel[:] = self.base_ang_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_ang_vel[:] * (1.0 - self.cfg["normalization"]["filter_weight"])
        self._refresh_feet_state()

        self.check_termination()
        self.compute_reward()
        self.compute_observations()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self._teleport_robot()
        self._kick_robots()
        self._push_robots()

        self.last_last_actions[:] = self.last_actions
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_feet_pos[:] = self.feet_pos
        self.last_toe_pos[:] = self.toe_pos
        self.last_heel_pos[:] = self.heel_pos

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _kick_robots(self):
        """Random kick the robots. Emulates an impulse by setting a randomized base velocity."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_interval_s"] / self.dt) == 0:
            self.root_states[:, 7:10] = apply_randomizations(self.root_states[:, 7:10], self.cfg["randomization"].get("kick_lin_vel"))
            self.root_states[:, 10:13] = apply_randomizations(self.root_states[:, 10:13], self.cfg["randomization"].get("kick_ang_vel"))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots(self):
        """Random push the robots. Emulates an impulse by setting a randomized force."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == 0:
            self.pushing_forces[:, self.base_indice, :] = apply_randomizations(torch.zeros_like(self.pushing_forces[:, 0, :]), self.cfg["randomization"].get("push_force"))
            self.pushing_torques[:, self.base_indice, :] = apply_randomizations(torch.zeros_like(self.pushing_torques[:, 0, :]), self.cfg["randomization"].get("push_torque"))
        elif self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == np.ceil(self.cfg["randomization"]["push_duration_s"] / self.dt):
            self.pushing_forces[:, self.base_indice, :].zero_()
            self.pushing_torques[:, self.base_indice, :].zero_()
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.pushing_forces), gymtorch.unwrap_tensor(self.pushing_torques), gymapi.LOCAL_SPACE)

    def _refresh_feet_state(self):
        self.feet_pos[:] = self.body_states[:, self.foot_indices, 0:3]
        self.feet_quat[:] = self.body_states[:, self.foot_indices, 3:7]
        toe_relative_pos = to_torch(self.cfg["asset"]["toe_pos"], device=self.device, requires_grad=False).unsqueeze(0).expand(self.num_envs, -1)
        heel_relative_pos = to_torch(self.cfg["asset"]["heel_pos"], device=self.device, requires_grad=False).unsqueeze(0).expand(self.num_envs, -1)
        for i in range(self.feet_quat.shape[1]):
            roll, _, yaw = get_euler_xyz(self.feet_quat[:, i, :])
            self.feet_roll[:, i] = (roll + torch.pi) % (2 * torch.pi) - torch.pi
            self.feet_yaw[:, i] = (yaw + torch.pi) % (2 * torch.pi) - torch.pi
            self.toe_pos[:, i, :] = self.feet_pos[:, i, :] + quat_rotate(self.feet_quat[:, i, :], toe_relative_pos)
            self.toe_contact[:, i] = self.toe_pos[:, i, 2] - self.terrain.terrain_heights(self.toe_pos[:, i, :]) < 0.01
            self.heel_pos[:, i, :] = self.feet_pos[:, i, :] + quat_rotate(self.feet_quat[:, i, :], heel_relative_pos)
            self.heel_contact[:, i] = self.heel_pos[:, i, 2] - self.terrain.terrain_heights(self.heel_pos[:, i, :]) < 0.01

    def check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        self.reset_buf |= self.root_states[:, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        self.low_height_count[torch.where(self.root_states[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"])] += 1
        low_height = self.low_height_count > 100
        upside_down = self.projected_gravity[:, 2] > 0.9
        side_roll = torch.abs(self.projected_gravity[:, 1]) > 0.8
        self.reset_buf |= low_height
        self.reset_buf |= upside_down
        if not self.cfg["env"]["allow_side_roll"]:
            self.reset_buf |= side_roll
        self.time_out_buf = self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)
        self.reset_buf |= self.time_out_buf

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew
        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def compute_observations(self):
        """Computes observations"""
        self.obs_buf = torch.cat(
            (
                apply_randomizations(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],
                apply_randomizations(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],
                apply_randomizations(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos"))[..., self.effective_ids] * self.cfg["normalization"]["dof_pos"],
                apply_randomizations(self.dof_vel, self.cfg["noise"].get("dof_vel"))[..., self.effective_ids] * self.cfg["normalization"]["dof_vel"],
                self.actions,
            ),
            dim=-1,
        )
        self.privileged_obs_buf = torch.cat(
            (
                self.base_mass_scaled,
                apply_randomizations(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],
                apply_randomizations(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos), self.cfg["noise"].get("height")).unsqueeze(-1),
                self.pushing_forces[:, 0, :] * self.cfg["normalization"]["push_force"],
                self.pushing_torques[:, 0, :] * self.cfg["normalization"]["push_torque"],
            ),
            dim=-1,
        )
        self.extras["privileged_obs"] = self.privileged_obs_buf

    @staticmethod
    def mirror_obs(obs):
        mat = torch.zeros(36, 36, dtype=torch.float, device=obs.device)
        mat[ 0: 6,  0: 6] = torch.eye(6)
        mat[ 6: 8,  8:10] = torch.eye(2)
        mat[ 8:10,  6: 8] = torch.eye(2)
        mat[10:13, 13:16] = torch.eye(3)
        mat[13:16, 10:13] = torch.eye(3)
        mat[16:18, 18:20] = torch.eye(2)
        mat[18:20, 16:18] = torch.eye(2)
        mat[20:23, 23:26] = torch.eye(3)
        mat[23:26, 20:23] = torch.eye(3)
        mat[26:28, 28:30] = torch.eye(2)
        mat[28:30, 26:28] = torch.eye(2)
        mat[30:33, 33:36] = torch.eye(3)
        mat[33:36, 30:33] = torch.eye(3)
        flip_val = torch.ones(36, dtype=torch.float, device=obs.device)
        inverse_ids = [  1,  3,  5,
                         7,  9, 
                        17, 19,
                        27, 29]
        flip_val[inverse_ids] = -1
        flip_mat = torch.diag(flip_val)
        mirror_transform_mat = torch.matmul(mat, flip_mat)
        mirrored_obs = torch.matmul(mirror_transform_mat, obs.unsqueeze(-1)).squeeze(-1)
        return mirrored_obs

    @staticmethod
    def mirror_priv(privileged):
        flip_val = torch.ones(14, dtype=torch.float, device=privileged.device)
        inverse_ids = [1, 5, 9, 11, 13]
        flip_val[inverse_ids] = -1
        flip_mat = torch.diag(flip_val)
        mirrored_priv = torch.matmul(flip_mat, privileged.unsqueeze(-1)).squeeze(-1)
        return mirrored_priv

    @staticmethod
    def mirror_act(act):
        mat = torch.zeros(10, 10, dtype=torch.float, device=act.device)
        mat[0:2, 2:4] = torch.eye(2)
        mat[2:4, 0:2] = torch.eye(2)
        mat[4:7, 7:10] = torch.eye(3)
        mat[7:10, 4:7] = torch.eye(3)
        flip_val = torch.ones(10, dtype=torch.float, device=act.device)
        inverse_ids = [1, 3]
        flip_val[inverse_ids] = -1
        flip_mat = torch.diag(flip_val)
        mirror_transform_mat = torch.matmul(mat, flip_mat)
        mirrored_act = torch.matmul(mirror_transform_mat, act.unsqueeze(-1)).squeeze(-1)
        return mirrored_act

    # ------------ reward functions----------------
    def _reward_survival(self):
        # Reward survival
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_base_height(self):
        # Tracking of base height
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        return torch.exp(-torch.square(base_height - self.cfg["rewards"]["base_height_target"]) / self.cfg["rewards"]["base_height_sigma"])
    
    def _reward_stand(self):
        return self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) > self.cfg["rewards"]["base_height_target"] - 0.1

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1.0, dim=-1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.filtered_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)

    def _reward_dof_pos_ref(self):
        # Penalize dof positions away from reference
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        standing_mask = torch.abs(base_height - self.cfg["rewards"]["base_height_target"]) < 0.1
        ref_loss = torch.sum(torch.square(self.dof_pos_ref - self.dof_pos), dim=-1)
        return torch.where(standing_mask, ref_loss, torch.zeros_like(ref_loss))

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=-1)

    def _reward_root_acc(self):
        # Penalize root accelerations
        return torch.sum(torch.square((self.last_root_vel - self.root_states[:, 7:13]) / self.dt), dim=-1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square((self.last_actions - self.actions) / self.dt), dim=-1) \
               + torch.sum(torch.square((self.actions - 2 * self.last_actions + self.last_last_actions) / self.dt), dim=-1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0])
        upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0])
        return torch.sum(((self.dof_pos < lower) | (self.dof_pos > upper)).float(), dim=-1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg["rewards"]["soft_dof_vel_limit"]).clip(min=0.0, max=1.0), dim=-1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg["rewards"]["soft_torque_limit"]).clip(min=0.0), dim=-1)

    def _reward_torque_tiredness(self):
        # Penalize torque tiredness
        return torch.sum(torch.square(self.torques / self.torque_limits).clip(max=1.0), dim=-1)

    def _reward_power(self):
        # Penalize power
        return torch.sum((self.torques * self.dof_vel).clip(min=0.0), dim=-1)

    def _reward_waist_pos(self):
        # Penalize waist position
        return torch.square(self.dof_pos[:, 0])

    def _reward_feet_slip(self):
        # Penalize feet velocities when contact
        return (
            torch.square((self.last_toe_pos - self.toe_pos) / self.dt).sum(dim=-1) * self.toe_contact.float()
            + torch.square((self.last_heel_pos - self.heel_pos) / self.dt).sum(dim=-1) * self.heel_contact.float()
        ).sum(dim=-1) * (self.episode_length_buf > 0).float()

    def _reward_feet_vel_z(self):
        return torch.sum(torch.square((self.last_feet_pos - self.feet_pos) / self.dt)[:, :, 2], dim=-1)

    def _reward_feet_roll(self):
        return torch.sum(torch.square(self.feet_roll), dim=-1)

    def _reward_feet_yaw_diff(self):
        return torch.square((self.feet_yaw[:, 1] - self.feet_yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_yaw_mean(self):
        feet_yaw_mean = self.feet_yaw.mean(dim=-1) + torch.pi * (torch.abs(self.feet_yaw[:, 1] - self.feet_yaw[:, 0]) > torch.pi)
        return torch.square((get_euler_xyz(self.base_quat)[2] - feet_yaw_mean + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_distance(self):
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        feet_distance = torch.abs(torch.cos(base_yaw) * (self.feet_pos[:, 1, 1] - self.feet_pos[:, 0, 1]) - torch.sin(base_yaw) * (self.feet_pos[:, 1, 0] - self.feet_pos[:, 0, 0]))
        return torch.clip(self.cfg["rewards"]["feet_distance_ref"] - feet_distance, min=0.0, max=0.1)

    def _reward_feet_contact_number(self):
        contact = self.toe_contact | self.heel_contact
        all_feet_off_ground = torch.all(contact == False, dim=1)
        return torch.where(all_feet_off_ground, 1, 0)
