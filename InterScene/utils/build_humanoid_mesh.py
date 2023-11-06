import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

def state2mat(pos, rot):
    Rm = R.from_quat(rot)
    matrix_l = np.hstack((Rm.as_matrix(), np.mat(pos).T))
    matrix_l = np.vstack((matrix_l, np.mat([0, 0, 0, 1])))
    return matrix_l.A

def build_complete_body(sub_body_transes, sub_body_rots, sub_body_meshes):
    full_body_mesh = []
    for i in range(len(sub_body_meshes)):
        mesh = sub_body_meshes[i].copy()
        matrix = state2mat(sub_body_transes[i], sub_body_rots[i])
        mesh.apply_transform(matrix)
        full_body_mesh.append(mesh)
    full_body_mesh = trimesh.util.concatenate(full_body_mesh)
    return full_body_mesh
