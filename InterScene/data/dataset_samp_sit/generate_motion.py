import sys
sys.path.append("./")

import os
import os.path as osp
import trimesh
import numpy as np
import argparse
from tqdm import tqdm
import pickle
import torch
from smplx import SMPLX
import torchgeometry as tgm
import yaml

from InterScene.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from InterScene.poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from InterScene.poselib.core.rotation3d import quat_mul, quat_from_angle_axis, quat_mul_norm, quat_rotate, quat_identity

from InterScene.utils.sp_animation import sp_animation
from InterScene.utils.build_humanoid_mesh import build_complete_body

def project_joints(motion):
    right_upper_arm_id = motion.skeleton_tree._node_indices["right_upper_arm"]
    right_lower_arm_id = motion.skeleton_tree._node_indices["right_lower_arm"]
    right_hand_id = motion.skeleton_tree._node_indices["right_hand"]
    left_upper_arm_id = motion.skeleton_tree._node_indices["left_upper_arm"]
    left_lower_arm_id = motion.skeleton_tree._node_indices["left_lower_arm"]
    left_hand_id = motion.skeleton_tree._node_indices["left_hand"]
    
    right_thigh_id = motion.skeleton_tree._node_indices["right_thigh"]
    right_shin_id = motion.skeleton_tree._node_indices["right_shin"]
    right_foot_id = motion.skeleton_tree._node_indices["right_foot"]
    left_thigh_id = motion.skeleton_tree._node_indices["left_thigh"]
    left_shin_id = motion.skeleton_tree._node_indices["left_shin"]
    left_foot_id = motion.skeleton_tree._node_indices["left_foot"]
    
    device = motion.global_translation.device

    # right arm
    right_upper_arm_pos = motion.global_translation[..., right_upper_arm_id, :]
    right_lower_arm_pos = motion.global_translation[..., right_lower_arm_id, :]
    right_hand_pos = motion.global_translation[..., right_hand_id, :]
    right_shoulder_rot = motion.local_rotation[..., right_upper_arm_id, :]
    right_elbow_rot = motion.local_rotation[..., right_lower_arm_id, :]
    
    right_arm_delta0 = right_upper_arm_pos - right_lower_arm_pos
    right_arm_delta1 = right_hand_pos - right_lower_arm_pos
    right_arm_delta0 = right_arm_delta0 / torch.norm(right_arm_delta0, dim=-1, keepdim=True)
    right_arm_delta1 = right_arm_delta1 / torch.norm(right_arm_delta1, dim=-1, keepdim=True)
    right_elbow_dot = torch.sum(-right_arm_delta0 * right_arm_delta1, dim=-1)
    right_elbow_dot = torch.clamp(right_elbow_dot, -1.0, 1.0)
    right_elbow_theta = torch.acos(right_elbow_dot)
    right_elbow_q = quat_from_angle_axis(-torch.abs(right_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                            device=device, dtype=torch.float32))
    
    right_elbow_local_dir = motion.skeleton_tree.local_translation[right_hand_id]
    right_elbow_local_dir = right_elbow_local_dir / torch.norm(right_elbow_local_dir)
    right_elbow_local_dir_tile = torch.tile(right_elbow_local_dir.unsqueeze(0), [right_elbow_rot.shape[0], 1])
    right_elbow_local_dir0 = quat_rotate(right_elbow_rot, right_elbow_local_dir_tile)
    right_elbow_local_dir1 = quat_rotate(right_elbow_q, right_elbow_local_dir_tile)
    right_arm_dot = torch.sum(right_elbow_local_dir0 * right_elbow_local_dir1, dim=-1)
    right_arm_dot = torch.clamp(right_arm_dot, -1.0, 1.0)
    right_arm_theta = torch.acos(right_arm_dot)
    right_arm_theta = torch.where(right_elbow_local_dir0[..., 1] <= 0, right_arm_theta, -right_arm_theta)
    right_arm_q = quat_from_angle_axis(right_arm_theta, right_elbow_local_dir.unsqueeze(0))
    right_shoulder_rot = quat_mul(right_shoulder_rot, right_arm_q)
    
    # left arm
    left_upper_arm_pos = motion.global_translation[..., left_upper_arm_id, :]
    left_lower_arm_pos = motion.global_translation[..., left_lower_arm_id, :]
    left_hand_pos = motion.global_translation[..., left_hand_id, :]
    left_shoulder_rot = motion.local_rotation[..., left_upper_arm_id, :]
    left_elbow_rot = motion.local_rotation[..., left_lower_arm_id, :]
    
    left_arm_delta0 = left_upper_arm_pos - left_lower_arm_pos
    left_arm_delta1 = left_hand_pos - left_lower_arm_pos
    left_arm_delta0 = left_arm_delta0 / torch.norm(left_arm_delta0, dim=-1, keepdim=True)
    left_arm_delta1 = left_arm_delta1 / torch.norm(left_arm_delta1, dim=-1, keepdim=True)
    left_elbow_dot = torch.sum(-left_arm_delta0 * left_arm_delta1, dim=-1)
    left_elbow_dot = torch.clamp(left_elbow_dot, -1.0, 1.0)
    left_elbow_theta = torch.acos(left_elbow_dot)
    left_elbow_q = quat_from_angle_axis(-torch.abs(left_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))

    left_elbow_local_dir = motion.skeleton_tree.local_translation[left_hand_id]
    left_elbow_local_dir = left_elbow_local_dir / torch.norm(left_elbow_local_dir)
    left_elbow_local_dir_tile = torch.tile(left_elbow_local_dir.unsqueeze(0), [left_elbow_rot.shape[0], 1])
    left_elbow_local_dir0 = quat_rotate(left_elbow_rot, left_elbow_local_dir_tile)
    left_elbow_local_dir1 = quat_rotate(left_elbow_q, left_elbow_local_dir_tile)
    left_arm_dot = torch.sum(left_elbow_local_dir0 * left_elbow_local_dir1, dim=-1)
    left_arm_dot = torch.clamp(left_arm_dot, -1.0, 1.0)
    left_arm_theta = torch.acos(left_arm_dot)
    left_arm_theta = torch.where(left_elbow_local_dir0[..., 1] <= 0, left_arm_theta, -left_arm_theta)
    left_arm_q = quat_from_angle_axis(left_arm_theta, left_elbow_local_dir.unsqueeze(0))
    left_shoulder_rot = quat_mul(left_shoulder_rot, left_arm_q)
    
    # right leg
    right_thigh_pos = motion.global_translation[..., right_thigh_id, :]
    right_shin_pos = motion.global_translation[..., right_shin_id, :]
    right_foot_pos = motion.global_translation[..., right_foot_id, :]
    right_hip_rot = motion.local_rotation[..., right_thigh_id, :]
    right_knee_rot = motion.local_rotation[..., right_shin_id, :]
    
    right_leg_delta0 = right_thigh_pos - right_shin_pos
    right_leg_delta1 = right_foot_pos - right_shin_pos
    right_leg_delta0 = right_leg_delta0 / torch.norm(right_leg_delta0, dim=-1, keepdim=True)
    right_leg_delta1 = right_leg_delta1 / torch.norm(right_leg_delta1, dim=-1, keepdim=True)
    right_knee_dot = torch.sum(-right_leg_delta0 * right_leg_delta1, dim=-1)
    right_knee_dot = torch.clamp(right_knee_dot, -1.0, 1.0)
    right_knee_theta = torch.acos(right_knee_dot)
    right_knee_q = quat_from_angle_axis(torch.abs(right_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))
    
    right_knee_local_dir = motion.skeleton_tree.local_translation[right_foot_id]
    right_knee_local_dir = right_knee_local_dir / torch.norm(right_knee_local_dir)
    right_knee_local_dir_tile = torch.tile(right_knee_local_dir.unsqueeze(0), [right_knee_rot.shape[0], 1])
    right_knee_local_dir0 = quat_rotate(right_knee_rot, right_knee_local_dir_tile)
    right_knee_local_dir1 = quat_rotate(right_knee_q, right_knee_local_dir_tile)
    right_leg_dot = torch.sum(right_knee_local_dir0 * right_knee_local_dir1, dim=-1)
    right_leg_dot = torch.clamp(right_leg_dot, -1.0, 1.0)
    right_leg_theta = torch.acos(right_leg_dot)
    right_leg_theta = torch.where(right_knee_local_dir0[..., 1] >= 0, right_leg_theta, -right_leg_theta)
    right_leg_q = quat_from_angle_axis(right_leg_theta, right_knee_local_dir.unsqueeze(0))
    right_hip_rot = quat_mul(right_hip_rot, right_leg_q)
    
    # left leg
    left_thigh_pos = motion.global_translation[..., left_thigh_id, :]
    left_shin_pos = motion.global_translation[..., left_shin_id, :]
    left_foot_pos = motion.global_translation[..., left_foot_id, :]
    left_hip_rot = motion.local_rotation[..., left_thigh_id, :]
    left_knee_rot = motion.local_rotation[..., left_shin_id, :]
    
    left_leg_delta0 = left_thigh_pos - left_shin_pos
    left_leg_delta1 = left_foot_pos - left_shin_pos
    left_leg_delta0 = left_leg_delta0 / torch.norm(left_leg_delta0, dim=-1, keepdim=True)
    left_leg_delta1 = left_leg_delta1 / torch.norm(left_leg_delta1, dim=-1, keepdim=True)
    left_knee_dot = torch.sum(-left_leg_delta0 * left_leg_delta1, dim=-1)
    left_knee_dot = torch.clamp(left_knee_dot, -1.0, 1.0)
    left_knee_theta = torch.acos(left_knee_dot)
    left_knee_q = quat_from_angle_axis(torch.abs(left_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))
    
    left_knee_local_dir = motion.skeleton_tree.local_translation[left_foot_id]
    left_knee_local_dir = left_knee_local_dir / torch.norm(left_knee_local_dir)
    left_knee_local_dir_tile = torch.tile(left_knee_local_dir.unsqueeze(0), [left_knee_rot.shape[0], 1])
    left_knee_local_dir0 = quat_rotate(left_knee_rot, left_knee_local_dir_tile)
    left_knee_local_dir1 = quat_rotate(left_knee_q, left_knee_local_dir_tile)
    left_leg_dot = torch.sum(left_knee_local_dir0 * left_knee_local_dir1, dim=-1)
    left_leg_dot = torch.clamp(left_leg_dot, -1.0, 1.0)
    left_leg_theta = torch.acos(left_leg_dot)
    left_leg_theta = torch.where(left_knee_local_dir0[..., 1] >= 0, left_leg_theta, -left_leg_theta)
    left_leg_q = quat_from_angle_axis(left_leg_theta, left_knee_local_dir.unsqueeze(0))
    left_hip_rot = quat_mul(left_hip_rot, left_leg_q)
    

    new_local_rotation = motion.local_rotation.clone()
    new_local_rotation[..., right_upper_arm_id, :] = right_shoulder_rot
    new_local_rotation[..., right_lower_arm_id, :] = right_elbow_q
    new_local_rotation[..., left_upper_arm_id, :] = left_shoulder_rot
    new_local_rotation[..., left_lower_arm_id, :] = left_elbow_q
    
    new_local_rotation[..., right_thigh_id, :] = right_hip_rot
    new_local_rotation[..., right_shin_id, :] = right_knee_q
    new_local_rotation[..., left_thigh_id, :] = left_hip_rot
    new_local_rotation[..., left_shin_id, :] = left_knee_q
    
    new_local_rotation[..., left_hand_id, :] = quat_identity([1])
    new_local_rotation[..., right_hand_id, :] = quat_identity([1])

    new_sk_state = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree, new_local_rotation, motion.root_translation, is_local=True)
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)
    
    return new_motion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--samp_pkl_dir", type=str, required=True)
    parser.add_argument("--smplx_dir", type=str, required=True)
    args = parser.parse_args()

    samp_pkl_dir = args.samp_pkl_dir
    smplx_dir = args.smplx_dir

    # hyperparameters
    target_fps = 30
    motion_cfg = {
        "chair_mo_stageII": {"start": 60, "end": 480, "stateInitRange": 0.7142},
        "chair_mo001_stageII": {"start": 60, "end": 480, "stateInitRange": 0.6714},
        "chair_mo002_stageII": {"start": 60, "end": 420, "stateInitRange": 0.5000},
        "chair_mo003_stageII": {"start": 60, "end": 480, "stateInitRange": 0.7000},
        "chair_mo004_stageII": {"start": 60, "end": 480, "stateInitRange": 0.7142},
        "chair_mo005_stageII": {"start": 42, "end": 480, "stateInitRange": 0.8082},
        "chair_mo006_stageII": {"start": 48, "end": 480, "stateInitRange": 0.7808},
        "chair_mo007_stageII": {"start": 48, "end": 480, "stateInitRange": 0.7083},
        "chair_mo008_stageII": {"start": 120, "end": 360, "stateInitRange": 0.525},
        "chair_mo009_stageII": {"start": 60, "end": 420, "stateInitRange": 0.5500},
        "chair_mo010_stageII": {"start": 90, "end": 480, "stateInitRange": 0.5077},
        "chair_mo011_stageII": {"start": 60, "end": 600, "stateInitRange": 0.7777},
        "chair_mo012_stageII": {"start": 30, "end": 360, "stateInitRange": 0.6181},
        "chair_mo013_stageII": {"start": 90, "end": 540, "stateInitRange": 0.6266},
        "chair_mo014_stageII": {"start": 78, "end": 420, "stateInitRange": 0.4736},
        "chair_mo015_stageII": {"start": 60, "end": 480, "stateInitRange": 0.7428},
        "chair_mo016_stageII": {"start": 60, "end": 420, "stateInitRange": 0.7000},
        "chair_mo017_stageII": {"start": 60, "end": 420, "stateInitRange": 0.5833},
        "chair_mo018_stageII": {"start": 60, "end": 540, "stateInitRange": 0.8125},
        "chair_mo019_stageII": {"start": 60, "end": 420, "stateInitRange": 0.7500},
    }

    with open(osp.join(osp.dirname(__file__), "dataset_samp_sit.yaml"), "r") as f:
        dataset_split = yaml.load(f, Loader=yaml.SafeLoader)["objects"]

    object_meshes = {}
    object_dir = osp.join(osp.dirname(__file__), "objects")
    for split, obj_ids in dataset_split.items():
        object_meshes[split] = {}
        for obj_name in obj_ids:
            textured_mesh = trimesh.load(osp.join(object_dir, obj_name, "geom/mesh.obj"), process=False)
            del_texture_mesh = trimesh.Trimesh(vertices=textured_mesh.vertices, faces=textured_mesh.faces)
            if split == "train":
                del_texture_mesh.visual.vertex_colors[:, :3] = np.array([1.00, 1.00, 0.88]) * 255
            else:
                del_texture_mesh.visual.vertex_colors[:, :3] = np.array([1.00, 0.88, 1.00]) * 255
            
            cfg_file_path = osp.join(object_dir, obj_name, "config.yaml")
            with open(cfg_file_path, "r") as f:
                obj_cfg = yaml.load(f, Loader=yaml.SafeLoader)
                from scipy.spatial.transform import Rotation as R
                r = R.from_euler('XYZ', obj_cfg["pose"]["rotAngle"] * np.asarray(obj_cfg["pose"]["rotAxis"]), degrees=False)
                rotation_matrix = r.as_matrix()
                transform_matrix = np.identity(4)
                transform_matrix[:3, :3] = rotation_matrix
                transform_matrix[:3, -1] = obj_cfg["pose"]["trans"]
                del_texture_mesh.apply_transform(transform_matrix)

            object_meshes[split][obj_name] = del_texture_mesh

    pbar = tqdm(list(motion_cfg.keys()))
    for seq in pbar:
        pbar.set_description(seq)

        save_dir = osp.join(osp.dirname(__file__), "motions", seq)
        os.makedirs(save_dir, exist_ok=True)

        # read smplx parameters from SAMP dataset
        with open(osp.join(samp_pkl_dir, seq + ".pkl"), "rb") as f:
            data = pickle.load(f, encoding="latin1")
            source_fps = data["mocap_framerate"]
            full_poses = torch.tensor(data["pose_est_fullposes"], dtype=torch.float32)
            full_trans = torch.tensor(data["pose_est_trans"], dtype=torch.float32)

        # downsample from source_fps (120Hz) to target_fps (30Hz)
        skip = int(source_fps // target_fps)
        full_poses = full_poses[::skip]
        full_trans = full_trans[::skip]

        # crop
        full_poses = full_poses[motion_cfg[seq]["start"]:motion_cfg[seq]["end"]]
        full_trans = full_trans[motion_cfg[seq]["start"]:motion_cfg[seq]["end"]]

        batch_size = full_poses.shape[0]
        bm = SMPLX(
            model_path=smplx_dir,
            gender="male",
            batch_size=batch_size,
            num_betas=10,
            use_pca=False,
            flat_hand_mean=True
        )
        full_trans = bm(
            global_orient=full_poses[:, 0:3],
            body_pose=full_poses[:, 3:66],
            transl=full_trans[:, :],
        ).joints[:, 0, :].cpu().detach() # get absolute position of root joint

        # extract useful joints
        joints_to_use = [0, 1, 4, 7, 2, 5, 8, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]
        joints_to_use = np.arange(0, 165).reshape((-1, 3))[joints_to_use].reshape(-1)
        full_poses = full_poses[:, joints_to_use]
    
        # angle axis ---> quaternion
        full_poses_quat = tgm.angle_axis_to_quaternion(full_poses.reshape(-1, 3)).reshape(full_poses.shape[0], -1, 4)

        # switch quaternion order
        # wxyz -> xyzw
        full_poses_quat = full_poses_quat[:, :, [1, 2, 3, 0]]

        ################ retarget ################

        # load tposes
        smplx_tpose = SkeletonState.from_file(osp.join(osp.dirname(__file__), "../tposes/smplx_tpose.npy"))
        amp_humanoid_tpose = SkeletonState.from_file(osp.join(osp.dirname(__file__), "../tposes/amp_humanoid_tpose.npy"))

        # plot_skeleton_state(amp_humanoid_tpose)
        # plot_skeleton_state(smplx_tpose)

        smplx_state = SkeletonState.from_rotation_and_root_translation(smplx_tpose.skeleton_tree, full_poses_quat, full_trans, is_local=True)

        # amp_humanoid (ID name)  smplx (ID name)
        #  0 "pelvis"              0 "pelvis"    
        #  1 "torso"               8 "spine"
        #  2 "head"               10 "neck"
        #  3 "right_upper_arm"    17 "right_upper_arm"
        #  4 "right_lower_arm"    18 "right_lower_arm"
        #  5 "right_hand"         19 "right_hand"
        #  6 "left_upper_arm"     13 "left_upper_arm"
        #  7 "left_lower_arm"     14 "left_lower_arm"
        #  8 "left_hand"          15 "left_hand"
        #  9 "right_thigh"         4 "right_thigh"
        # 10 "right_shin"          5 "right_thin"
        # 11 "right_foot"          6 "right_foot"
        # 12 "left_thigh"          1 "left_thigh"
        # 13 "left_shin"           2 "left_shin"
        # 14 "left_foot"           3 "left_foot"
        joint_mapping = [0, 8, 10, 17, 18, 19, 13, 14, 15, 4, 5, 6, 1, 2, 3]

        gr = amp_humanoid_tpose.global_rotation.clone()
        gr = quat_mul_norm(torch.tensor([-0.5, -0.5, -0.5, 0.5]), gr.clone())
        gr = quat_mul_norm(smplx_state.global_rotation.clone()[..., joint_mapping, :], gr.clone())

        amp_humanoid_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree=amp_humanoid_tpose.skeleton_tree,
            r=gr,
            t=smplx_state.root_translation.clone(),
            is_local=False,
        ).local_repr()

        # move the root so that the feet are on the ground
        amp_humanoid_lr = amp_humanoid_state.local_rotation.clone()
        amp_humanoid_rt = amp_humanoid_state.root_translation.clone()

        # check foot-ground penetration for every frame
        for i in range(amp_humanoid_rt.shape[0]):
            min_h = torch.min(amp_humanoid_state.global_translation[i, :, 2])
            amp_humanoid_rt[i, 2] += -min_h

        amp_humanoid_rt[:, 2] += 0.05
        amp_humanoid_rt[:, 1] += 0.1

        new_sk_state = SkeletonState.from_rotation_and_root_translation(amp_humanoid_tpose.skeleton_tree, amp_humanoid_lr, amp_humanoid_rt, is_local=True)
        res = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_fps)

        # need to convert some joints from 3D to 1D (e.g. elbows and knees)
        res = project_joints(res)
        res.to_file(osp.join(save_dir, "ref_motion.npy"))

        # make scenepic animation
        sp_anim = sp_animation(framerate=target_fps)
        plane = trimesh.load(osp.join(osp.dirname(__file__), "../meshes/plane.obj"), process=False)
        sp_anim.add_static_mesh(plane, 'plane')
        for obj_name, obj_mesh in object_meshes["train"].items():
            sp_anim.add_static_mesh(obj_mesh, obj_name)
        for obj_name, obj_mesh in object_meshes["test"].items():
            sp_anim.add_static_mesh(obj_mesh, obj_name)

        part_meshes = []
        for partName in amp_humanoid_tpose.skeleton_tree.node_names:
            part_meshes.append(trimesh.load(osp.join(osp.dirname(__file__), "../meshes/amp_humanoid/{}.obj".format(partName))))
      
        global_rotation_of_bodies = res.global_rotation
        global_translation_of_bodies = res.global_translation
        num_frames = global_rotation_of_bodies.shape[0]

        for i in range(num_frames):
            human_mesh = build_complete_body(global_translation_of_bodies[i], global_rotation_of_bodies[i], part_meshes)
            if i <= num_frames * motion_cfg[seq]["stateInitRange"]:
                human_mesh.visual.vertex_colors[:, :3] = np.array([0.94, 0.97, 1.00]) * 255
            else:
                human_mesh.visual.vertex_colors[:, :3] = np.array([0.93, 0.07, 0.54]) * 255
            sp_anim.add_frame([human_mesh], ['humanoid'])
        
        sp_anim.save_animation(osp.join(save_dir, "ref_motion_vis.html"))
