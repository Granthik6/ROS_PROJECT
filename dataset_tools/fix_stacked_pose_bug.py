import os
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
from compute_furniture_bounds import combined_bounds_from_sdf, GAZEBO_MODELS

WORLD = "/root/ros2_ws/src/robot_tasks/worlds/realistic_room.world"

AFFECTED = {
    "asus_xtion_pro_camera_object": "asus_xtion_pro_camera",
    "cardboard_box_object": "cardboard_box",
    "fire_hose_long_object": "fire_hose_long",
    "hokuyo_object": "hokuyo",
    "kinect_object": "kinect",
    "stop_light_object": "stop_light",
}

world_text = open(WORLD).read()

for entity, model_dir in AFFECTED.items():
    sdf_path = os.path.join(GAZEBO_MODELS, model_dir, "model.sdf")
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    model_el = root.find("model")
    mpose = model_el.find("pose")
    model_root_z = float(mpose.text.split()[2]) if mpose is not None else 0.0

    bounds = combined_bounds_from_sdf(sdf_path, model_dir)
    bmin, bmax = bounds
    true_bottom = bmin[2] + model_root_z  # mesh bottom in the model's own frame, plus its baked-in root pose
    true_top = bmax[2] + model_root_z

    # current world pose
    m = re.search(
        rf'<name>{re.escape(entity)}</name>\s*<pose>([^<]+)</pose>', world_text
    )
    cur_pose = m.group(1).split()
    cur_world_z = float(cur_pose[2])

    # to make the true bottom sit at world z=0: world_z = -true_bottom
    new_world_z = round(-true_bottom, 4)

    print(f"{entity:35s} model_root_z={model_root_z:7.4f} mesh_bottom(local)={bmin[2]:8.4f} "
          f"true_bottom(world-frame-if-z0)={true_bottom:8.4f} true_top_height={true_top - true_bottom:7.4f} "
          f"cur_world_z={cur_world_z:8.4f} -> new_world_z={new_world_z:8.4f}")
