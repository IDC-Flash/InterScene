# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from enum import Enum

import env.base_envs.humanoid_amp_singleEnv as humanoid_amp_singleEnv
from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

class HumanoidLocationSit(humanoid_amp_singleEnv.HumanoidAMPSingleEnv):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # this env is used for training and evaluating the sit policy

        self._enable_task_obs = cfg["env"]["enableTaskObs"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._tar_speed = cfg["env"]["tarSpeed"]
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float) # 3D XYZ

        # Interaction Early Termination (IET)
        self._success_threshold = cfg["env"]["successThreshold"]
        self._max_IET_steps = cfg["env"]["maxIETSteps"]
        self._IET_step_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._IET_state_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        if (not self._is_train):
            self._setup_evaluation()

        if (not self.headless):
            self._build_marker_state_tensors()

        return

    def _setup_evaluation(self):
        # setup buffers
        self._success_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.long)
        self._executionTime_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)
        self._precision_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)

        # override max_episode_length
        self.max_episode_length = self.cfg["eval"]["episodeLength"]

        # override max_IET_steps
        if self.cfg["eval"]["disableIET"]:
            self._max_IET_steps = self.max_episode_length + 1000000

        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._marker_handles = []
            self._load_marker_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_marker_asset(self):
        asset_root = "InterScene/data/assets/mjcf/"
        asset_file = "location_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        if (not self.headless):
            self._build_marker(env_id, env_ptr)

        return

    def _build_marker(self, env_id, env_ptr):
        col_group = self.num_envs # humanoid's col_group [0, 1, 2, ..., self.num_envs-1]
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()

        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self.gym.set_actor_scale(env_ptr, marker_handle, 0.5)
        self._marker_handles.append(marker_handle)

        return

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 2, :]
        self._marker_pos = self._marker_states[..., :3]

        self._marker_actor_ids = self._object_actor_ids + 1

        return

    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return

    def _draw_task(self):

        self._update_marker()

        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        # self.gym.clear_lines(self.viewer)

        starts = self._humanoid_root_states[..., 0:3]
        ends = self._marker_pos

        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return

    def _update_marker(self):
        self._marker_pos[..., 0:3] = self._tar_pos
        env_ids_int32 = torch.cat([self._marker_actor_ids, self._object_actor_ids.view(-1)], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_task(env_ids) # reset task here
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)

        self._init_amp_obs(env_ids)

        return

    def _reset_actors(self, env_ids):
        if self._is_train:
            super()._reset_actors(env_ids)
        else:
            super()._reset_default(env_ids) # when testing, we only use "reset default"!!!
        return

    def _reset_task(self, env_ids):
        object_ids = self._every_env_object_ids[env_ids]
        target_sit_locations = self._object_lib.get_object_tar_sit_positions(object_ids)

        # transform from object local space to world space
        translation_to_world = self._object_root_states[env_ids, 0:3]
        rotation_to_world = self._object_root_states[env_ids, 3:7]
        target_sit_locations = quat_rotate(rotation_to_world, target_sit_locations) + translation_to_world

        self._tar_pos[env_ids] = target_sit_locations

        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        self._IET_step_buf[env_ids] = 0
        self._IET_state_buf[env_ids] = 0

        if (not self._is_train):
            self._success_buf[env_ids] = 0
            self._executionTime_buf[env_ids] = 0
            self._precision_buf[env_ids] = float('Inf')

    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if (self._enable_task_obs):
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size += 3 # xyz target location
        return obs_size
    
    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        object_obs = self._compute_object_obs(env_ids)
        policy_obs = torch.cat([humanoid_obs, object_obs], dim=-1)
        
        if (self._enable_task_obs):
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([policy_obs, task_obs], dim=-1)
        else:
            obs = policy_obs

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_root_pos = self._tar_pos
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_root_pos = self._tar_pos[env_ids]

        obs = compute_location_observations(root_states, tar_root_pos)
        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        object_root_pos = self._object_root_states[..., 0:3]
        object_root_rot = self._object_root_states[..., 3:7]

        self.rew_buf[:] = compute_location_reward(root_pos, self._prev_root_pos, root_rot, 
                                                    object_root_pos,
                                                    self._tar_pos, self._tar_speed,
                                                    self.dt,)

        mask = compute_finish_state(root_pos, self._tar_pos, self._success_threshold)
        self._IET_step_buf[mask] += 1

        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:], self._IET_state_buf[:] = compute_humanoid_reset_location(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights,
                                                   self._max_IET_steps, self._IET_step_buf)
        return
    
    def _fetch_humanoid_full_states(self, env_ids):
        n = len(env_ids)
        root_state = self._humanoid_root_states[env_ids] # 13
        dof_pos = self._dof_pos[env_ids] # 28
        dof_vel = self._dof_vel[env_ids] # 28
        rigid_body_state = torch.cat((self._rigid_body_pos[env_ids], self._rigid_body_rot[env_ids], self._rigid_body_vel[env_ids], self._rigid_body_ang_vel[env_ids]), dim=-1)
        rigid_body_state = rigid_body_state.view(n, -1)
        return torch.cat((root_state, dof_pos, dof_vel, rigid_body_state), dim=-1)

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return

    def post_physics_step(self):
        super().post_physics_step()

        if (not self._is_train):
            self._compute_metrics_evaluation()
            self.extras["success"] = self._success_buf
            self.extras["executionTime"] = self._executionTime_buf
            self.extras["precision"] = self._precision_buf
            
        return

    def _compute_metrics_evaluation(self):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        tar_pos = self._tar_pos

        pos_diff = tar_pos - root_pos
        pos_err = torch.norm(pos_diff, p=2, dim=-1)
        dist_mask = pos_err <= self._success_threshold
        self._success_buf[dist_mask] += 1

        self._precision_buf[dist_mask] = torch.where(pos_err[dist_mask] < self._precision_buf[dist_mask], pos_err[dist_mask], self._precision_buf[dist_mask])

        first_time_success_mask = (self._success_buf == 1)
        self._executionTime_buf[first_time_success_mask] = self.progress_buf[first_time_success_mask] * self.dt

        return

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_location_observations(root_states, tar_root_pos):
    # type: (Tensor, Tensor) -> Tensor

    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    # xyz
    obs = quat_rotate(heading_rot, tar_root_pos - root_pos)
    return obs

@torch.jit.script
def compute_location_reward(root_pos, prev_root_pos, root_rot, object_root_pos, tar_pos, tar_speed, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor

    dist_threshold = 0.5

    # when humanoid is far away from the object
    pos_diff = object_root_pos - root_pos # xyz
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)

    tar_dir = object_root_pos[..., 0:2] - root_pos[..., 0:2] # d* is a horizontal unit vector pointing from the root to the object's location
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)

    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    reward_far_pos = torch.exp(-0.5 * pos_err)
    reward_far_vel = torch.exp(-2.0 * tar_vel_err * tar_vel_err)
    speed_mask = tar_dir_speed <= 0
    reward_far_vel[speed_mask] = 0

    reward_far_final = 0.5 * reward_far_pos + 0.4 * reward_far_vel + 0.1 * facing_reward
    dist_mask = pos_err < dist_threshold
    reward_far_final[dist_mask] = 1.0

    # when humanoid is close to the object
    pos_diff = tar_pos - root_pos # xyz
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    reward_near = torch.exp(-10.0 * pos_err)

    reward = 0.7 * reward_near + 0.3 * reward_far_final

    return reward

@torch.jit.script
def compute_finish_state(root_pos, tar_pos, success_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    pos_diff = tar_pos - root_pos
    pos_err = torch.norm(pos_diff, p=2, dim=-1)
    dist_mask = pos_err <= success_threshold
    return dist_mask

@torch.jit.script
def compute_humanoid_reset_location(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights,
                           max_finish_length, finish_length_buf):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, int, Tensor) -> Tuple[Tensor, Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    finished = torch.zeros_like(reset_buf)

    if (enable_early_termination):

        # fall down
        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)
        has_fallen = torch.logical_and(torch.ones_like(fall_height), fall_height)

        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    finished = torch.where(finish_length_buf >= max_finish_length - 1, torch.ones_like(reset_buf), finished)
    reset = torch.logical_or(finished, terminated)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reset, terminated, finished
