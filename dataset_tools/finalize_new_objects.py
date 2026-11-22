import csv

IN_CSV = "/root/ros2_ws/dataset_tools/new_object_bounds.csv"
OUT_PY = "/root/ros2_ws/dataset_tools/new_objects_entries.py"

FALLBACK_HALF = (0.25, 0.25, 0.3)
FALLBACK_OFFSET_Z = 0.3

ALL_NEW = [
    ("beer_object", "beer", (-5.5, -4.5)),
    ("fire_hydrant_object", "fire_hydrant", (-4.15, -4.5)),
    ("construction_cone_object", "construction_cone", (-2.8, -4.5)),
    ("construction_barrel_object", "construction_barrel", (-1.45, -4.5)),
    ("cardboard_box_object", "cardboard_box", (-0.1, -4.5)),
    ("cricket_ball_object", "cricket_ball", (1.25, -4.5)),
    ("monkey_wrench_object", "monkey_wrench", (2.6, -4.5)),
    ("mailbox_object", "mailbox", (3.95, -4.5)),
    ("postbox_object", "postbox", (5.3, -4.5)),
    ("stop_sign_object", "stop_sign", (-4.83, -3.15)),
    ("speed_limit_sign_object", "speed_limit_sign", (-2.12, -3.15)),
    ("lamp_post_object", "lamp_post", (-0.77, -3.15)),
    ("first_2015_trash_can_object", "first_2015_trash_can", (0.58, -3.15)),
    ("utility_cart_object", "utility_cart", (1.93, -3.15)),
    ("grey_tote_object", "grey_tote", (3.28, -3.15)),
    ("car_wheel_object", "car_wheel", (-5.5, -1.8)),
    ("door_handle_object", "door_handle", (-1.45, -1.8)),
    ("kinect_object", "kinect", (-0.1, -1.8)),
    ("asus_xtion_pro_camera_object", "asus_xtion_pro_camera", (1.25, -1.8)),
    ("hokuyo_object", "hokuyo", (2.6, -1.8)),
    ("table_marble_object", "table_marble", (3.95, -1.8)),
    ("wooden_case_object", "wooden_case", (5.3, -1.8)),
    ("saucepan_object", "saucepan", (-4.83, -0.45)),
    ("jersey_barrier_object", "jersey_barrier", (-2.12, -0.45)),
    ("stop_light_object", "stop_light", (-0.77, -0.45)),
    ("oak_tree_object", "oak_tree", (0.58, -0.45)),
    ("ball_bearing_object", "ball_bearing", (3.28, -0.45)),
    ("gear_part_object", "gear_part", (4.62, -0.45)),
    ("fire_hose_long_object", "fire_hose_long", (-5.5, 0.9)),
]

calibrated = {}
with open(IN_CSV) as f:
    for row in csv.DictReader(f):
        calibrated[row["cls"]] = row


def sane(row):
    ox, oy = float(row["offset_x"]), float(row["offset_y"])
    hx, hy, hz = float(row["half_x"]), float(row["half_y"]), float(row["half_z"])
    if abs(ox) > 0.8 or abs(oy) > 0.8:
        return False
    if max(hx, hy, hz) > 1.0 or min(hx, hy, hz) < 0.01:
        return False
    return True


lines = []
n_calibrated = 0
n_fallback = 0

for entity, cls, pos in ALL_NEW:
    row = calibrated.get(cls)
    if row is not None and sane(row):
        half = (round(float(row["half_x"]), 4), round(float(row["half_y"]), 4),
                round(float(row["half_z"]), 4))
        offset = (round(float(row["offset_x"]), 4), round(float(row["offset_y"]), 4),
                  round(float(row["offset_z"]), 4))
        n_calibrated += 1
        src = "calibrated"
    else:
        half = FALLBACK_HALF
        offset = (0.0, 0.0, FALLBACK_OFFSET_Z)
        n_fallback += 1
        src = "fallback"

    lines.append(
        f'    dict(entity="{entity}", cls="{cls}", pos=({pos[0]}, {pos[1]}), yaw=0.0,\n'
        f'         half={half}, offset={offset},\n'
        f'         distances=[1.3, 1.8, 2.4, 3.2]),  # {src}\n'
    )

with open(OUT_PY, "w") as f:
    f.writelines(lines)

print(f"calibrated={n_calibrated} fallback={n_fallback} total={len(ALL_NEW)}")
print(f"wrote {OUT_PY}")
