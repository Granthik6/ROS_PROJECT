import os
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, "/root/ros2_ws/dataset_tools")
from compute_furniture_bounds import combined_bounds_from_sdf, GAZEBO_MODELS

WORLD = "/root/ros2_ws/src/robot_tasks/worlds/realistic_room.world"
TOLERANCE = 0.02  # 2cm - below this we don't care

world_text = open(WORLD).read()

entries = []
for block in re.findall(r"<include>(.*?)</include>", world_text, re.S):
    name_m = re.search(r"<name>([^<]+)</name>", block)
    uri_m = re.search(r"<uri>model://([^<]+)</uri>", block)
    pose_m = re.search(r"<pose>([^<]+)</pose>", block)
    if not (name_m and uri_m):
        continue
    entity = name_m.group(1)
    model_dir = uri_m.group(1)
    pose = pose_m.group(1).split() if pose_m else ["0"] * 6
    entries.append((entity, model_dir, float(pose[2])))

print(f"{'entity':35s} {'model':25s} {'joints':7s} {'world_z':>9s} {'true_bottom':>12s} {'status':>10s}")
print("-" * 100)

flagged = []
skipped_jointed = []

for entity, model_dir, world_z in entries:
    sdf_path = os.path.join(GAZEBO_MODELS, model_dir, "model.sdf")
    if not os.path.exists(sdf_path):
        candidates = [f for f in os.listdir(os.path.join(GAZEBO_MODELS, model_dir))
                      if f.endswith(".sdf")] if os.path.isdir(os.path.join(GAZEBO_MODELS, model_dir)) else []
        if not candidates:
            print(f"{entity:35s} {model_dir:25s}  NO MODEL DIR FOUND (skip - not a mesh prop, e.g. wall/ground)")
            continue
        sdf_path = os.path.join(GAZEBO_MODELS, model_dir, candidates[0])

    n_joints = 0
    try:
        tree = ET.parse(sdf_path)
        root = tree.getroot()
        n_joints = len(root.findall(".//joint"))
    except Exception as e:
        print(f"{entity:35s} {model_dir:25s}  PARSE ERROR: {e}")
        continue

    if n_joints > 0:
        print(f"{entity:35s} {model_dir:25s} {n_joints:7d} {world_z:9.4f} {'--jointed--':>12s} {'SKIP':>10s}")
        skipped_jointed.append(entity)
        continue

    model_el = root.find("model")
    mpose = model_el.find("pose") if model_el is not None else None
    model_root_z = float(mpose.text.split()[2]) if mpose is not None else 0.0

    try:
        bounds = combined_bounds_from_sdf(sdf_path, model_dir)
    except Exception as e:
        print(f"{entity:35s} {model_dir:25s} {0:7d} {world_z:9.4f}  BOUNDS ERROR: {e}")
        continue

    if bounds is None:
        print(f"{entity:35s} {model_dir:25s} {0:7d} {world_z:9.4f} {'no geometry':>12s} {'SKIP':>10s}")
        continue

    bmin, _ = bounds
    true_bottom_local = bmin[2] + model_root_z
    true_bottom_world = true_bottom_local + world_z

    status = "OK"
    if true_bottom_world > TOLERANCE:
        status = "FLOATING"
    elif true_bottom_world < -TOLERANCE:
        status = "SUNK"

    print(f"{entity:35s} {model_dir:25s} {n_joints:7d} {world_z:9.4f} {true_bottom_world:12.4f} {status:>10s}")

    if status != "OK":
        flagged.append((entity, true_bottom_world, world_z, true_bottom_local))

print()
print("=== SUMMARY ===")
if flagged:
    print(f"{len(flagged)} object(s) still floating/sunk:")
    for entity, tb, wz, tbl in flagged:
        correction = -tbl
        print(f"  {entity}: true_bottom={tb:.4f}m  (currently world_z={wz:.4f} -> should be {correction:.4f})")
else:
    print("All non-jointed objects sit correctly on the ground (within 2cm tolerance).")
print(f"{len(skipped_jointed)} jointed/dynamic model(s) skipped (need manual visual check, not bounds math): {skipped_jointed}")
