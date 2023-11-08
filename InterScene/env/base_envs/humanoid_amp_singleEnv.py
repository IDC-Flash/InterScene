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

from enum import Enum
import numpy as np
import torch

from isaacgym import gymapi
from isaacgym import gymtorch

from env.base_envs.humanoid_amp import HumanoidAMP
from utils import gym_util
from utils.object_lib import ObjectLib
from isaacgym.torch_utils import *

from utils import torch_utils

class HumanoidAMPSingleEnv(HumanoidAMP):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # this env only contains 1 object

        self._is_train = cfg["args"].train # determine to load training or testing set of objects
        
        self._randomRot_obj = cfg["env"]["randomRotObj"]
        self._randomRot_humanoid = cfg["env"]["randomRotHumanoid"]
        self._min_dist = cfg["env"]["minDist"]
        self._max_dist = cfg["env"]["maxDist"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._build_object_state_tensors()

        return

    def get_obs_size(self):
        return super().get_obs_size() + 8 * 3 + 2 # eight 3D points on bbox + 2D facing vector

    def _compute_object_obs(self, env_ids=None):
        if (env_ids is None):
            root_pos = self._humanoid_root_states[..., 0:3]
            root_rot = self._humanoid_root_states[..., 3:7]
            obj_root_pos = self._object_root_states[..., 0:3]
            obj_root_rot = self._object_root_states[..., 3:7]
            obj_bps = self._every_env_object_bps
            obj_facings = self._every_env_object_facings
        else:
            root_pos = self._humanoid_root_states[env_ids, 0:3]
            root_rot = self._humanoid_root_states[env_ids, 3:7]
            obj_root_pos = self._object_root_states[env_ids, 0:3]
            obj_root_rot = self._object_root_states[env_ids, 3:7]
            obj_bps = self._every_env_object_bps[env_ids]
            obj_facings = self._every_env_object_facings[env_ids]

        obs = build_object_observations(root_pos, root_rot, obj_root_pos, obj_root_rot, obj_bps, obj_facings)
        return obs

    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        object_obs = self._compute_object_obs(env_ids)
        obs = torch.cat([humanoid_obs, object_obs], dim=-1)

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._object_assets = [] # size: [num_object_assets]
        self._object_handles = [] # size: [num_envs]

        self._object_lib = ObjectLib(is_train=self._is_train, motion_file=self.cfg['env']['motion_file'], device=self.device)

        # Load assets
        self._load_object_assets()

        # random sample a fixed object for each simulation env, due to the limitation of IsaacGym
        self._every_env_object_ids = self._object_lib.sample_objects(num_envs)
        self._every_env_object_bps, self._every_env_object_facings = self._object_lib.get_object_infos(self._every_env_object_ids)

        super()._create_envs(num_envs, spacing, num_per_row)

        return
    
    def _load_object_assets(self):
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        # Load materials from meshes
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

        # use default convex decomposition params
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 100000
        asset_options.vhacd_params.max_convex_hulls = 50
        asset_options.vhacd_params.max_num_vertices_per_ch = 64

        # load all object assets
        asset_root = "./"
        for asset_file in self._object_lib._all_object_urdfs:
            asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            self._object_assets.append(asset)

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_object(env_id, env_ptr)
        return

    def _get_object_collision_filter(self):
        return self._get_humanoid_collision_filter() + 1 # enable collision detection between humanoid and object

    def _build_object(self, env_id, env_ptr):
        col_group = env_id
        col_filter = self._get_object_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, 5.0) # avoid overlapping with the humanoid
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # create actor
        asset_id = self._every_env_object_ids[env_id]
        asset = self._object_assets[asset_id]
        handle = self.gym.create_actor(env_ptr, asset, start_pose, "object", col_group, col_filter, segmentation_id)
        self._object_handles.append(handle)
    
        return
    
    def _build_object_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._object_root_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        self._object_actor_ids = self._humanoid_actor_ids + 1

        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        env_ids_int32 = self._object_actor_ids[env_ids].view(-1)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return
    
    def _reset_default(self, env_ids):
        # acquire initial state of humanoid
        root_pos = self._initial_humanoid_root_states[env_ids, 0:3]
        root_rot = self._initial_humanoid_root_states[env_ids, 3:7]
        root_vel = self._initial_humanoid_root_states[env_ids, 7:10]
        root_ang_vel = self._initial_humanoid_root_states[env_ids, 10:13]
        dof_pos = self._initial_dof_pos[env_ids]
        dof_vel = self._initial_dof_vel[env_ids]

        # acquire reference state of object
        object_ids = self._every_env_object_ids[env_ids]
        root_pos_obj, root_rot_obj = self._object_lib.get_object_states(object_ids)

        num_samples = len(env_ids)
        rot_axis = torch.zeros((num_samples, 3), device=self.device, dtype=torch.float32)
        rot_axis[..., 2] = 1 # z

        ########### randomize humanoid state (pos/rot) ###########
        a = torch.rand(num_samples, device=self.device, dtype=torch.float32) * (self._max_dist ** 2 - self._min_dist) + self._min_dist # r=sqrt(a) r:1-5 a:1-25
        b = torch.rand(num_samples, device=self.device, dtype=torch.float32)
        
        x_noises = torch.sqrt(a) * torch.cos(2 * np.pi * b)
        y_noises = torch.sqrt(a) * torch.sin(2 * np.pi * b)

        # import matplotlib.pyplot as plt
        # plt.scatter(x_noises.cpu().numpy(), y_noises.cpu().numpy())
        # plt.show()

        z_noises = torch.zeros((num_samples, 1), device=self.device, dtype=torch.float32)
        root_pos_noises = torch.cat((x_noises.unsqueeze(-1), y_noises.unsqueeze(-1), z_noises), dim=-1)
        root_pos_noises += root_pos_obj
        root_pos_noises -= root_pos
        root_pos_noises[..., 2] = 0

        # root_pos (in global frame)
        root_pos += root_pos_noises

        if not self._randomRot_humanoid:
            pass
        else:
            random_rot_angle = torch.rand(num_samples, device=self.device, dtype=torch.float32) * np.pi * 2
            random_rot_quat = quat_from_angle_axis(random_rot_angle, rot_axis)
            root_rot_noises = random_rot_quat
            root_rot = quat_mul(root_rot, root_rot_noises)

        ########### randomize object state (rot) ###########
        if not self._randomRot_obj:
            # rotation
            root_rot_obj_noises = torch.zeros((num_samples, 4), device=self.device, dtype=torch.float32)
            root_rot_obj_noises[..., -1] = 1 # quat [0, 0, 0, 1]
        else:
            # rotation
            random_rot_angle = torch.rand(num_samples, device=self.device, dtype=torch.float32) * np.pi * 2
            random_rot_quat = quat_from_angle_axis(random_rot_angle, rot_axis)
            root_rot_obj_noises = random_rot_quat
        
        root_rot_obj = quat_mul(root_rot_obj, root_rot_obj_noises)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)
        
        self._set_object_state(env_ids=env_ids,
                               root_pos=root_pos_obj,
                               root_rot=root_rot_obj)
        
        self._reset_default_env_ids = env_ids

        if (len(env_ids) > 0):
            self._kinematic_humanoid_rigid_body_states[env_ids] = \
                compute_kinematic_rigid_body_states(root_pos, root_rot, self._initial_humanoid_rigid_body_states[env_ids])

        self._every_env_init_dof_pos[env_ids] = self._initial_dof_pos[env_ids] # for "enableTrackInitState"

        return

    def _reset_ref_state_init(self, env_ids):
        super()._reset_ref_state_init(env_ids)

        # reset objects
        object_ids = self._every_env_object_ids[env_ids]
        root_pos_obj, root_rot_obj = self._object_lib.get_object_states(object_ids)   

        self._set_object_state(env_ids=env_ids,
                               root_pos=root_pos_obj,
                               root_rot=root_rot_obj)
        
        return
    
    def _set_object_state(self, env_ids, root_pos, root_rot):
        self._object_root_states[env_ids, 0:3] = root_pos
        self._object_root_states[env_ids, 3:7] = root_rot
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights)
        return
    
    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self.gym.clear_lines(self.viewer)
            self._draw_obj_bbox()
        return
    
    def _draw_obj_bbox(self):
        # draw lines of the bbox
        cols = np.zeros((12, 3), dtype=np.float32) # 12 lines
        cols[:] = [1.0, 0.0, 0.0] # red

        # transform bps from object local space to world space
        bps = self._every_env_object_bps.clone().reshape(self.num_envs, -1, 3)
        obj_pos = self._object_root_states[:, 0:3]
        obj_rot = self._object_root_states[:, 3:7]
        obj_pos_exp = torch.broadcast_to(obj_pos.unsqueeze(-2), (obj_pos.shape[0], bps.shape[1], obj_pos.shape[1])) # (num_envs, 3) >> (num_envs, 8, 3)
        obj_rot_exp = torch.broadcast_to(obj_rot.unsqueeze(-2), (obj_rot.shape[0], bps.shape[1], obj_rot.shape[1])) # (num_envs, 4) >> (num_envs, 8, 4)
        box_bps_world_space = (quat_rotate(obj_rot_exp.reshape(-1, 4), bps.reshape(-1, 3)) + obj_pos_exp.reshape(-1, 3)).reshape(self.num_envs, 8, 3) # (num_envs, 8, 3)

        verts = torch.cat([
            box_bps_world_space[:, 0, :], box_bps_world_space[:, 1, :],
            box_bps_world_space[:, 1, :], box_bps_world_space[:, 2, :],
            box_bps_world_space[:, 2, :], box_bps_world_space[:, 3, :],
            box_bps_world_space[:, 3, :], box_bps_world_space[:, 0, :],

            box_bps_world_space[:, 4, :], box_bps_world_space[:, 5, :],
            box_bps_world_space[:, 5, :], box_bps_world_space[:, 6, :],
            box_bps_world_space[:, 6, :], box_bps_world_space[:, 7, :],
            box_bps_world_space[:, 7, :], box_bps_world_space[:, 4, :],

            box_bps_world_space[:, 0, :], box_bps_world_space[:, 4, :],
            box_bps_world_space[:, 1, :], box_bps_world_space[:, 5, :],
            box_bps_world_space[:, 2, :], box_bps_world_space[:, 6, :],
            box_bps_world_space[:, 3, :], box_bps_world_space[:, 7, :],
        ], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([12, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def build_object_observations(root_pos, root_rot, obj_root_pos, obj_root_rot, obj_bps, obj_facings):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    obj_root_pos_exp = torch.broadcast_to(obj_root_pos.unsqueeze(1), (obj_root_pos.shape[0], 8, obj_root_pos.shape[1])).reshape(-1, 3) # [4096, 3] >> [4096, 8, 3] >> [4096*8, 3]
    obj_root_rot_exp = torch.broadcast_to(obj_root_rot.unsqueeze(1), (obj_root_rot.shape[0], 8, obj_root_rot.shape[1])).reshape(-1, 4) # [4096, 4] >> [4096, 8, 4] >> [4096*8, 4]
    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(1), (root_pos.shape[0], 8, root_pos.shape[1])).reshape(-1, 3) # [4096, 3] >> [4096, 8, 3] >> [4096*8, 3]
    heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(1), (heading_rot.shape[0], 8, heading_rot.shape[1])).reshape(-1, 4) # [4096, 4] >> [4096, 8, 4] >> [4096*8, 4]

    obj_bps_world_space = quat_rotate(obj_root_rot_exp, obj_bps.reshape(-1, 3)) + obj_root_pos_exp
    obj_bps_local_space = quat_rotate(heading_rot_exp, obj_bps_world_space - root_pos_exp).reshape(-1, 24)

    face_vec_world_space = quat_rotate(obj_root_rot, obj_facings)
    face_vec_local_space = quat_rotate(heading_rot, face_vec_world_space)

    obs = torch.cat([obj_bps_local_space, face_vec_local_space[..., 0:2]], dim=-1)
    return obs

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        # masked_contact_buf = contact_buf.clone()
        # masked_contact_buf[:, contact_body_ids, :] = 0
        # fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        # fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(torch.ones_like(fall_height), fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

@torch.jit.script
def compute_kinematic_rigid_body_states(root_pos, root_rot, initial_humanoid_rigid_body_states):
    # type: (Tensor, Tensor, Tensor) -> Tensor

    num_bodies = initial_humanoid_rigid_body_states.shape[1] # 15

    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(1), (root_pos.shape[0], num_bodies, root_pos.shape[1])).reshape(-1, 3) # (num_envs, 3) >> (num_envs, 15, 3) >> (num_envs*15, 3)
    root_rot_exp = torch.broadcast_to(root_rot.unsqueeze(1), (root_rot.shape[0], num_bodies, root_rot.shape[1])).reshape(-1, 4) # (num_envs, 4) >> (num_envs, 15, 4) >> (num_envs*15, 4)

    init_body_pos = initial_humanoid_rigid_body_states[..., 0:3] # (num_envs, 15, 3)
    init_body_rot = initial_humanoid_rigid_body_states[..., 3:7] # (num_envs, 15, 4)
    init_body_vel = initial_humanoid_rigid_body_states[..., 7:10] # (num_envs, 15, 3)
    init_body_ang_vel = initial_humanoid_rigid_body_states[..., 10:13] # (num_envs, 15, 3)

    init_root_pos = init_body_pos[:, 0:1, :] # (num_envs, 1, 3)
    init_body_pos_canonical = (init_body_pos - init_root_pos).reshape(-1, 3) # (num_envs, 15, 3) >> (num_envs*15, 3)
    init_body_rot = init_body_rot.reshape(-1, 4)
    init_body_vel = init_body_vel.reshape(-1, 3)
    init_body_ang_vel = init_body_ang_vel.reshape(-1, 3)
    
    curr_body_pos = (quat_rotate(root_rot_exp, init_body_pos_canonical) + root_pos_exp).reshape(-1, num_bodies, 3)
    curr_body_rot = (quat_mul(root_rot_exp, init_body_rot)).reshape(-1, num_bodies, 4)
    curr_body_vel = (quat_rotate(root_rot_exp, init_body_vel)).reshape(-1, num_bodies, 3)
    curr_body_ang_vel = (quat_rotate(root_rot_exp, init_body_ang_vel)).reshape(-1, num_bodies, 3)
    curr_humanoid_rigid_body_states = torch.cat((curr_body_pos, curr_body_rot, curr_body_vel, curr_body_ang_vel), dim=-1)
    
    return curr_humanoid_rigid_body_states
