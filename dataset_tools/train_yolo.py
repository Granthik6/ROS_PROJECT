import os
from ultralytics import YOLO

DATA_YAML = os.path.expanduser("~/ros2_ws/realistic_sim_yolo.yaml")
BASE_MODEL = os.path.expanduser("~/ros2_ws/yolo11n.pt")
PROJECT = os.path.expanduser("~/ros2_ws/realistic_sim_yolo_runs")


def main():
    model = YOLO(BASE_MODEL)

    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=16,
        patience=25,
        project=PROJECT,
        name="realistic_sim_yolo11n",
        seed=0,
        deterministic=True,
        plots=True,
        val=True,
    )


if __name__ == "__main__":
    main()
