import os
from ultralytics import YOLO

DATA_YAML = os.path.expanduser("~/ros2_ws/realistic_sim_yolo.yaml")
WEIGHTS = os.path.expanduser(
    "~/ros2_ws/realistic_sim_yolo_runs/realistic_sim_yolo11n_39class_best_backup.pt"
)


def main():
    model = YOLO(WEIGHTS)
    metrics = model.val(data=DATA_YAML, split="test")
    per_class = {
        metrics.names[cls_id]: round(ap50, 3)
        for cls_id, ap50 in zip(metrics.box.ap_class_index, metrics.box.ap50)
    }
    print("per-class mAP50 (backup):", per_class)


if __name__ == "__main__":
    main()
