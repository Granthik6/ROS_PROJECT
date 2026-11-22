import csv
import os
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

YOLO_MODEL_PATH = (
    '/root/ros2_ws/realistic_sim_yolo_runs/'
    'realistic_sim_yolo11n/weights/best.pt'
)
HOG_MODEL_PATH = os.path.expanduser('~/ros2_ws/realistic_sim_hog_svm.yml')
CSV_PATH = os.path.expanduser('~/ros2_ws/vision_comparison.csv')

HOG_CLASSES = {
    0: 'cup', 1: 'bowl', 2: 'coke_can',
    3: 'cordless_drill', 4: 'hammer', 5: 'background',
}
BACKGROUND_ID = 5


class CompareVisionNode(Node):

    def __init__(self):
        super().__init__('compare_vision')
        self.bridge = CvBridge()

        self.svm = cv2.ml.SVM_load(HOG_MODEL_PATH)
        self.hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)

        self.yolo = YOLO(YOLO_MODEL_PATH)

        self.subscription = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10
        )

        new_file = not os.path.exists(CSV_PATH)
        self.csv_file = open(CSV_PATH, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if new_file:
            self.csv_writer.writerow([
                'stamp', 'hog_ms', 'hog_fps', 'hog_prediction',
                'yolo_ms', 'yolo_fps', 'yolo_num_detections',
                'yolo_mean_confidence',
            ])

        self.get_logger().info(f'HOG model: {HOG_MODEL_PATH}')
        self.get_logger().info(f'YOLO model: {YOLO_MODEL_PATH}')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        t1 = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128))
        features = self.hog.compute(resized).reshape(1, -1)
        _, result = self.svm.predict(features)
        hog_ms = (time.time() - t1) * 1000
        hog_pred_id = int(result[0][0])
        hog_pred = HOG_CLASSES.get(hog_pred_id, 'unknown')

        t2 = time.time()
        yolo_results = self.yolo(frame, conf=0.25, imgsz=640, verbose=False)
        yolo_ms = (time.time() - t2) * 1000

        boxes = yolo_results[0].boxes
        num_dets = 0 if boxes is None else len(boxes)
        mean_conf = (
            float(boxes.conf.mean().item()) if num_dets > 0 else 0.0
        )

        hog_fps = 1000.0 / max(hog_ms, 0.001)
        yolo_fps = 1000.0 / max(yolo_ms, 0.001)

        self.csv_writer.writerow([
            time.time(), round(hog_ms, 3), round(hog_fps, 2), hog_pred,
            round(yolo_ms, 3), round(yolo_fps, 2), num_dets, round(mean_conf, 4),
        ])
        self.csv_file.flush()

        os.system('clear')
        print(f'| Metric           | HOG + SVM        | YOLO (realistic_sim) |')
        print(f'|------------------|------------------|-----------------------|')
        print(f'| Inference Time   | {hog_ms:6.2f} ms       | {yolo_ms:6.2f} ms             |')
        print(f'| Implied FPS      | {hog_fps:6.1f}           | {yolo_fps:6.1f}                 |')
        print(f'| Prediction       | {hog_pred:16s} | {num_dets} detection(s)'
              f'{" @ conf=" + format(mean_conf, ".2f") if num_dets else ""}')
        print()
        print(f'Logging to {CSV_PATH}')

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CompareVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
