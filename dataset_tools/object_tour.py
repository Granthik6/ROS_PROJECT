import math
import subprocess
import time

TARGETS = [
    ("cup", (-1.8, -2.5)),
    ("bowl", (-0.4, -2.8)),
    ("coke_can", (1.2, -2.5)),
    ("cordless_drill", (-1.0, 0.0)),
    ("hammer", (1.8, 0.0)),
]

DISTANCES = [1.3, 1.8, 2.3]


def set_pose(x, y, yaw):
    subprocess.run(
        ["gz", "model", "-m", "burger", "-x", str(x), "-y", str(y),
         "-z", "0.02", "-R", "0", "-P", "0", "-Y", str(yaw)],
        capture_output=True, timeout=8.0
    )


def main():
    for name, (ox, oy) in TARGETS:
        for dist in DISTANCES:
            for angle_deg in (-15, 0, 15):
                theta = math.radians(angle_deg)
                robot_x = ox - dist * math.cos(theta)
                robot_y = oy - dist * math.sin(theta)
                yaw = math.atan2(oy - robot_y, ox - robot_x)
                set_pose(robot_x, robot_y, yaw)
                print(f"{name}: at dist={dist} angle={angle_deg}")
                time.sleep(1.2)


if __name__ == "__main__":
    main()
