import os
import yaml
import torch
import numpy as np

from poselib.core.rotation3d import *
from isaacgym.torch_utils import *

class ObjectLib():
    def __init__(self, is_train, motion_file, device):
        self._is_train = is_train
        self._device = device
        self._load_objects(motion_file)

    def _load_valid_seated_poses(self, data):
        # all variables are in world space
        self._seated_humanoid_root_states = data[..., 0:13].reshape(self.num_objects(), -1, 13)
        self._seated_humanoid_rigid_body_states = data[..., 69:].reshape(self.num_objects(), -1, 15, 13)
        self._seated_humanoid_dof_poss = data[..., 13:41].reshape(self.num_objects(), -1, 28)
        self._seated_humanoid_dof_vels = data[..., 41:69].reshape(self.num_objects(), -1, 28)
        return

    def _load_objects(self, motion_file):
        self._centers = []
        self._bboxes = []
        self._bboxes_bps = [] # bps: bounding points
        self._facings = []
        self._tar_sit_positions = []

        self._pose_scales = []
        self._pose_transls = []
        self._pose_rotangles = []
        self._pose_rotaxiss = []
        self._pose_rotquats = []

        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            dir_name = os.path.dirname(motion_file)

            with open(os.path.join(os.getcwd(), motion_file), "r") as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            if self._is_train:
                self._all_object_names = motion_config["objects"]["train"]
            else:
                self._all_object_names = motion_config["objects"]["test"]

            self._all_object_urdfs = [os.path.join(dir_name, "objects/{}/asset.urdf".format(k)) for k in self._all_object_names]
            for obj in self._all_object_names:
                with open(os.path.join(dir_name, "objects/{}/config.yaml".format(obj)), "r") as f:
                    state = yaml.load(f, Loader=yaml.SafeLoader)
                    self._centers.append(state["center"])
                    self._bboxes.append(state["bbox"])
                    self._facings.append(state["facing"])
                    self._tar_sit_positions.append(state["tarSitPos"])

                    self._pose_scales.append(state["pose"]["scale"])
                    self._pose_transls.append(state["pose"]["trans"])
                    self._pose_rotangles.append(state["pose"]["rotAngle"])
                    self._pose_rotaxiss.append(state["pose"]["rotAxis"])

                    # conver angle axis to quaternion
                    angle = torch.tensor(state["pose"]["rotAngle"])
                    axis = torch.tensor(state["pose"]["rotAxis"], dtype=torch.float32)
                    quat = quat_from_angle_axis(angle, axis).numpy()
                    self._pose_rotquats.append(quat)

                    # compute bps for the bbox
                    object_bbox = np.array(state["bbox"], dtype=np.float32)
                    object_P1 = np.array([     object_bbox[0]/2,      object_bbox[1]/2, -1 * object_bbox[2]/2])
                    object_P2 = np.array([-1 * object_bbox[0]/2,      object_bbox[1]/2, -1 * object_bbox[2]/2])
                    object_P3 = np.array([-1 * object_bbox[0]/2, -1 * object_bbox[1]/2, -1 * object_bbox[2]/2])
                    object_P4 = np.array([     object_bbox[0]/2, -1 * object_bbox[1]/2, -1 * object_bbox[2]/2])
                    object_P5 = np.array([     object_bbox[0]/2,      object_bbox[1]/2,      object_bbox[2]/2])
                    object_P6 = np.array([-1 * object_bbox[0]/2,      object_bbox[1]/2,      object_bbox[2]/2])
                    object_P7 = np.array([-1 * object_bbox[0]/2, -1 * object_bbox[1]/2,      object_bbox[2]/2])
                    object_P8 = np.array([     object_bbox[0]/2, -1 * object_bbox[1]/2,      object_bbox[2]/2])
                    self._bboxes_bps.append(
                        np.concatenate([
                            object_P1, object_P2, object_P3, object_P4, 
                            object_P5, object_P6, object_P7, object_P8,
                        ])
                    )

        else:
            raise Exception("Currently only support motion file using .yaml input")

        # convert to tensor
        convert = lambda data: torch.tensor(data, device=self._device, dtype=torch.float32)
        self._pose_scales = convert(self._pose_scales)
        self._pose_transls = convert(self._pose_transls)
        self._pose_rotangles = convert(self._pose_rotangles)
        self._pose_rotaxiss = convert(self._pose_rotaxiss)
        self._pose_rotquats = convert(self._pose_rotquats)

        self._centers = convert(self._centers)
        self._bboxes = convert(self._bboxes)
        self._bboxes_bps = convert(self._bboxes_bps)
        self._facings = convert(self._facings)
        self._tar_sit_positions = convert(self._tar_sit_positions)

        print("===== Object Lib Infor =====")
        print("Loaded {:d} objects.".format(self.num_objects()))

        return
    
    def num_objects(self):
        return len(self._all_object_names)
    
    def get_object_infos(self, object_ids):
        return self._bboxes_bps[object_ids], self._facings[object_ids]

    def get_object_tar_sit_positions(self, object_ids):
        return self._tar_sit_positions[object_ids]

    def get_object_states(self, object_ids):
        pos = (self._pose_transls - self._centers)[object_ids]
        rot = self._pose_rotquats[object_ids]
        return pos, rot
    
    def get_seated_humanoid_root_states(self, object_ids, pose_ids):
        return self._seated_humanoid_root_states[object_ids, pose_ids]
        
    def get_seated_humanoid_rigid_body_states(self, object_ids, pose_ids):
        return self._seated_humanoid_rigid_body_states[object_ids, pose_ids]
    
    def get_seated_humanoid_dof_poss(self, object_ids, pose_ids):
        return self._seated_humanoid_dof_poss[object_ids, pose_ids]
    
    def get_seated_humanoid_dof_vels(self, object_ids, pose_ids):
        return self._seated_humanoid_dof_vels[object_ids, pose_ids]

    def sample_objects(self, n):
        num_objects = self.num_objects()
        weights = torch.full(size=[num_objects], fill_value=1.0/num_objects, device=self._device)
        object_ids = torch.multinomial(weights, num_samples=n, replacement=True).squeeze(-1)
        return object_ids
    
    def sample_poses(self, n):
        num_poses_per_object = self._seated_humanoid_root_states.shape[1] # 7
        weights = torch.full(size=[n, num_poses_per_object], fill_value=1.0/num_poses_per_object, device=self._device)
        pose_ids = torch.multinomial(weights, num_samples=1, replacement=True).squeeze(-1)
        return pose_ids
