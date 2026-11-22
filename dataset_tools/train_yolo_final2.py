import os
from ultralytics import YOLO

DATA_YAML = os.path.expanduser("~/ros2_ws/realistic_sim_yolo.yaml")
WARM_START = os.path.expanduser(
    "~/ros2_ws/realistic_sim_yolo_runs/realistic_sim_yolo11n_final/weights/best.pt"
)
PROJECT = os.path.expanduser("~/ros2_ws/realistic_sim_yolo_runs")


def main():
    model = YOLO(WARM_START)

    model.train(
        data=DATA_YAML,
        epochs=40,
        imgsz=640,
        batch=16,
        patience=15,
        project=PROJECT,
        name="realistic_sim_yolo11n_final2",
        seed=0,
        deterministic=True,
        plots=True,
        val=True,
    )


if __name__ == "__main__":
    main()
