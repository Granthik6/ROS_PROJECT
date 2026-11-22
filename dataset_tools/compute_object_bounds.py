import os
import trimesh
import numpy as np

# (model_dir, mesh_rel_path, visual_pose_xyz, class_name, unit_meter_scale)
OBJECTS = [
    ("plastic_cup", "meshes/plastic_cup.dae", (0, 0, 0.065), "cup", 1.0),
    ("coke_can", "meshes/coke_can.dae", (0.003937, 0.0047244, -0.18), "coke_can", 0.001),
    ("cordless_drill", "meshes/cordless_drill.dae", (0, 0, -0.09), "cordless_drill", 0.001),
    ("hammer", "meshes/hammer.dae", (0, 0, 0), "hammer", 0.0254),
    ("bowl", "meshes/bowl.dae", (0, 0, 0), "bowl", 1.0),
]

GAZEBO_MODELS = os.path.expanduser("~/gazebo_models")

results = {}

for model_dir, mesh_rel, pose_xyz, class_name, unit_scale in OBJECTS:
    mesh_path = os.path.join(GAZEBO_MODELS, model_dir, mesh_rel)
    scene_or_mesh = trimesh.load(mesh_path, force='scene')

    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = scene_or_mesh.to_geometry()
    else:
        mesh = scene_or_mesh

    bounds = mesh.bounds * unit_scale  # (2,3) min/max in meters, mesh local frame

    size = bounds[1] - bounds[0]
    center_local = (bounds[0] + bounds[1]) / 2.0

    # apply visual pose translation to get position relative to model/link origin
    center_model = center_local + np.array(pose_xyz)

    half = size / 2.0

    print(f"{class_name:15s} size_m(x,y,z)={size.round(4).tolist()} "
          f"center_in_model_frame={center_model.round(4).tolist()}")

    results[class_name] = {
        "size": size.round(4).tolist(),
        "center_offset": center_model.round(4).tolist(),
    }

import json
print()
print(json.dumps(results, indent=2))
