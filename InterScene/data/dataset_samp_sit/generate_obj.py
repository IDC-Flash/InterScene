import os
import os.path as osp
import yaml
import trimesh
import numpy as np

if __name__ == "__main__":
    obj_dir = osp.join(osp.dirname(__file__), "objects")
    obj_files = os.listdir(obj_dir)
    for id in obj_files:
        components = []

        mesh = trimesh.load(osp.join(obj_dir, id, "geom/mesh.obj"))
        
        with open(osp.join(obj_dir, id, "config.yaml"), "rb") as f:
            cfg = yaml.safe_load(f)

        # load target sitting position
        marker = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
        marker.apply_translation(cfg["tarSitPos"])
        marker.visual.vertex_colors[:, :3] = [255, 0, 0]

        components.append(mesh)
        components.append(marker)

        # load bbox
        bbox = cfg["bbox"]
        line_x = trimesh.creation.box([bbox[0], 0.02, 0.02])
        line_y = trimesh.creation.box([0.02, bbox[1], 0.02])
        line_z = trimesh.creation.box([0.02, 0.02, bbox[2]])

        line_x.visual.vertex_colors[:, :3] = [0, 255, 0]
        line_y.visual.vertex_colors[:, :3] = [0, 255, 0]
        line_z.visual.vertex_colors[:, :3] = [0, 255, 0]

        components.append(line_x.copy().apply_translation([0,      bbox[1]/2, -1 * bbox[2]/2]))
        components.append(line_x.copy().apply_translation([0,      bbox[1]/2,      bbox[2]/2]))
        components.append(line_x.copy().apply_translation([0, -1 * bbox[1]/2,      bbox[2]/2]))
        components.append(line_x.copy().apply_translation([0, -1 * bbox[1]/2, -1 * bbox[2]/2]))
        components.append(line_y.copy().apply_translation([     bbox[0]/2, 0, -1 * bbox[2]/2]))
        components.append(line_y.copy().apply_translation([-1 * bbox[0]/2, 0, -1 * bbox[2]/2]))
        components.append(line_y.copy().apply_translation([-1 * bbox[0]/2, 0,      bbox[2]/2]))
        components.append(line_y.copy().apply_translation([     bbox[0]/2, 0,      bbox[2]/2]))
        components.append(line_z.copy().apply_translation([     bbox[0]/2,      bbox[1]/2, 0]))
        components.append(line_z.copy().apply_translation([     bbox[0]/2, -1 * bbox[1]/2, 0]))
        components.append(line_z.copy().apply_translation([-1 * bbox[0]/2, -1 * bbox[1]/2, 0]))
        components.append(line_z.copy().apply_translation([-1 * bbox[0]/2,      bbox[1]/2, 0]))

        merged_mesh = trimesh.util.concatenate(components)

        # transform from canonical space to world space
        merged_mesh.apply_scale([cfg["pose"]["scale"], cfg["pose"]["scale"], cfg["pose"]["scale"]])
        from scipy.spatial.transform import Rotation as R
        r = R.from_euler('XYZ', cfg["pose"]["rotAngle"] * np.asarray(cfg["pose"]["rotAxis"]), degrees=False)
        rotation_matrix = r.as_matrix()
        transform_matrix = np.identity(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, -1] = cfg["pose"]["trans"]
        merged_mesh.apply_transform(transform_matrix)

        # save
        save_dir = osp.join(obj_dir, id, "vis")
        os.makedirs(save_dir, exist_ok=True)
        merged_mesh.export(osp.join(save_dir, "mesh.obj"))

        print("Saving processed object mesh at {}".format(save_dir))
