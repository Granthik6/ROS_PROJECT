import os
import sys

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
from compute_furniture_bounds import combined_bounds_from_sdf, GAZEBO_MODELS

TARGETS = ["stop_sign", "table_marble"]

MODEL_ROOT_Z = {
    "stop_sign": 0.0,
    "table_marble": 0.648,
}

for model_dir in TARGETS:
    sdf_path = os.path.join(GAZEBO_MODELS, model_dir, "model.sdf")
    bounds = combined_bounds_from_sdf(sdf_path, model_dir)
    if bounds is None:
        print(f"{model_dir}: NO GEOMETRY FOUND")
        continue
    bmin, bmax = bounds
    bmin = bmin.copy()
    bmax = bmax.copy()
    bmin[2] += MODEL_ROOT_Z[model_dir]
    bmax[2] += MODEL_ROOT_Z[model_dir]
    size = bmax - bmin
    center = (bmin + bmax) / 2.0
    half = size / 2.0
    print(f"{model_dir:15s} half={half.round(4).tolist()} offset={center.round(4).tolist()} "
          f"(z range {bmin[2]:.3f} to {bmax[2]:.3f})")
