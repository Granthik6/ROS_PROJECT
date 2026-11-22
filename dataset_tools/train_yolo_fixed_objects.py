import os
from ultralytics import YOLO

DATA_YAML = os.path.expanduser("~/ros2_ws/realistic_sim_yolo.yaml")
WARM_START = os.path.expanduser(
    "~/ros2_ws/realistic_sim_yolo_runs/realistic_sim_yolo11n_39class_best_backup.pt"
)
PROJECT = os.path.expanduser("~/ros2_ws/realistic_sim_yolo_runs")


def main():
    model = YOLO(WARM_START)

    model.train(
        data=DATA_YAML,
        epochs=30,
        imgsz=640,
        batch=16,
        patience=10,
        project=PROJECT,
        name="realistic_sim_yolo11n_fixed_objects",
        seed=0,
        deterministic=True,
        plots=True,
        val=True,
    )


if __name__ == "__main__":
    main()
