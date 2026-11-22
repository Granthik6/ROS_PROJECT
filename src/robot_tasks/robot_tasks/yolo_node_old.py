import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import time

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        
        yolo_dir = os.path.expanduser('~/ros2_ws/src/robot_tasks/robot_tasks/yolo_files')
        cfg_path = os.path.join(yolo_dir, 'yolov4-tiny.cfg')
        weights_path = os.path.join(yolo_dir, 'yolov4-tiny.weights')
        names_path = os.path.join(yolo_dir, 'coco.names')

        self.net = cv2.dnn.readNet(weights_path, cfg_path)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.confidence_publisher = self.create_publisher(
            Float32,
            '/detection_confidence',
            10
        )

        self.frame_count = 0
        self.start_time = time.time()

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        class_ids, scores, boxes = self.model.detect(frame, confThreshold=0.2, nmsThreshold=0.4)
        
        if len(class_ids) > 0:
            for class_id, score, box in zip(class_ids, scores, boxes):
        
                c_id = class_id[0] if isinstance(
                    class_id, (list, tuple, np.ndarray)
                ) else class_id
        
                confidence = float(
                    score[0] if isinstance(score, (list, tuple, np.ndarray)) else score
                )
        
                label = f"{self.classes[c_id]}: {confidence:.2f}"
        
                cv2.rectangle(frame, box, (0, 255, 255), 2)
        
                cv2.putText(
                    frame,
                    label,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )
        
                msg = Float32()
                msg.data = confidence
                self.confidence_publisher.publish(msg)
        
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            self.get_logger().info(f'YOLOv4-tiny FPS: {fps:.2f}')
            self.frame_count = 0
            self.start_time = time.time()

        cv2.imshow('YOLO-tiny Detection', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
