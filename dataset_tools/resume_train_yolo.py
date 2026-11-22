from ultralytics import YOLO

LAST_WEIGHTS = "/root/ros2_ws/realistic_sim_yolo_runs/realistic_sim_yolo11n/weights/last.pt"


def main():
    model = YOLO(LAST_WEIGHTS)
    model.train(resume=True)


if __name__ == "__main__":
    main()
