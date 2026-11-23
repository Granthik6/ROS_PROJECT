import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
from compute_furniture_bounds import combined_bounds_from_sdf, GAZEBO_MODELS

TARGETS = {
    "asus_xtion_pro_camera": "asus_xtion_pro_camera_object",
    "cardboard_box": "cardboard_box_object",
    "fire_hose_long": "fire_hose_long_object",
    "hokuyo": "hokuyo_object",
    "kinect": "kinect_object",
    "stop_light": "stop_light_object",
}

for model_dir, entity in TARGETS.items():
    sdf_path = os.path.join(GAZEBO_MODELS, model_dir, "model.sdf")
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    model_el = root.find("model")
    mpose = model_el.find("pose")
    model_root_z = float(mpose.text.split()[2]) if mpose is not None else 0.0

    bmin, bmax = combined_bounds_from_sdf(sdf_path, model_dir)
    bmin = bmin.copy()
    bmax = bmax.copy()
    bmin[2] += model_root_z
    bmax[2] += model_root_z

    size = bmax - bmin
    half = size / 2.0
    offset_xy = (bmin[:2] + bmax[:2]) / 2.0
    offset_z = half[2]

    print(f"{entity:32s} half=({half[0]:.4f}, {half[1]:.4f}, {half[2]:.4f})  "
          f"offset=({offset_xy[0]:.4f}, {offset_xy[1]:.4f}, {offset_z:.4f})")
