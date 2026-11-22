import os
from ultralytics import YOLO

DATA_YAML = os.path.expanduser("~/ros2_ws/realistic_sim_yolo.yaml")
WEIGHTS = os.path.expanduser(
    "~/ros2_ws/realistic_sim_yolo_runs/realistic_sim_yolo11n_fixed_objects/weights/best.pt"
)


def main():
    model = YOLO(WEIGHTS)
    metrics = model.val(data=DATA_YAML, split="test")
    print()
    print("=== TEST SET METRICS (fixed_objects) ===")
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)
    per_class = {
        metrics.names[cls_id]: round(ap50, 3)
        for cls_id, ap50 in zip(metrics.box.ap_class_index, metrics.box.ap50)
    }
    print("per-class mAP50:", per_class)


if __name__ == "__main__":
    main()
