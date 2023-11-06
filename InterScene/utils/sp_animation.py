import scenepic as sp
import numpy as np

class sp_animation():
    def __init__(self,
                 width = 1600,
                 height = 1600,
                 framerate = 30.0,
                 ):
        super(sp_animation, self).__init__()

        self.scene = sp.Scene()
        self.main = self.scene.create_canvas_3d(width=width, height=height)
        self.colors = sp.Colors

        self.scene.framerate = framerate
        self.static_meshes = []

    def trimeshes_to_sp(self, trimeshes_list, layer_names):

        sp_meshes = []

        for i, m in enumerate(trimeshes_list):
            params = {
                'vertices': m.vertices.astype(np.float32),
                'normals': m.vertex_normals.astype(np.float32),
                'triangles': m.faces,
                'colors': m.visual.vertex_colors[:, :3].astype(np.float32) / 255
            }
            sp_m = self.scene.create_mesh(layer_id=layer_names[i])
            sp_m.add_mesh_with_normals(**params)
            if layer_names[i] == 'ground_mesh':
                sp_m.double_sided = True
            sp_meshes.append(sp_m)

        return sp_meshes

    def add_static_mesh(self, trimesh, layer_name):
        sp_mesh = self.trimeshes_to_sp([trimesh], [layer_name])
        self.static_meshes += sp_mesh

    def add_frame(self, meshes_list_tm, layer_names):

        meshes_list = self.trimeshes_to_sp(meshes_list_tm, layer_names)
        if not hasattr(self, 'focus_point'):
            look_at = self.static_meshes[0].center_of_mass[np.newaxis, ]
            center = look_at + [8, -8, 5]
            self.camera = sp.Camera(center=center, look_at=look_at, up_dir=[0, 0, 1], fov_y_degrees=30.0, far_crop_distance=40.0)

        main_frame = self.main.create_frame(meshes=self.static_meshes, camera=self.camera)
        for i, m in enumerate(meshes_list):
            # self.main.set_layer_settings({layer_names[i]:{}})
            main_frame.add_mesh(m)

    def save_animation(self, sp_anim_name):
        self.scene.link_canvas_events(self.main)
        self.scene.save_as_html(sp_anim_name, title=sp_anim_name.split('/')[-1])
