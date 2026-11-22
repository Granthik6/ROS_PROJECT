import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time


class HogSvmDetector(Node):

    def __init__(self):
        super().__init__('hog_svm_detector')

        model_path = os.path.expanduser(
            '~/ros2_ws/realistic_sim_hog_svm.yml'
        )

        self.svm = cv2.ml.SVM_load(model_path)

        self.hog = cv2.HOGDescriptor(
            (128, 128),
            (16, 16),
            (8, 8),
            (8, 8),
            9
        )

        self.bridge = CvBridge()

        self.classes = {
            0: 'cup',
            1: 'bowl',
            2: 'coke_can',
            3: 'cordless_drill',
            4: 'hammer',
            5: 'background'
        }
        self.background_id = 5

        self.subscription = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.frame_count = 0
        self.start_time = time.time()

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        frame_for_model = cv2.resize(
            gray,
            (128, 128)
        )

        features = self.hog.compute(
            frame_for_model
        )

        features = features.reshape(
            1,
            -1
        )

        _, result = self.svm.predict(
            features
        )

        prediction = int(
            result[0][0]
        )

        label = self.classes.get(
            prediction,
            'Unknown'
        )

        if prediction != self.background_id:

            cv2.putText(
                frame,
                'Detected: ' + label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                'Background',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        self.frame_count += 1

        elapsed = time.time() - self.start_time

        if elapsed >= 1.0:

            fps = self.frame_count / elapsed

            self.get_logger().info(
                f'HOG + SVM FPS: {fps:.2f}'
            )

            self.frame_count = 0
            self.start_time = time.time()

        cv2.imshow(
            'HOG + SVM Gazebo Detection',
            frame
        )

        cv2.waitKey(1)

    def destroy_node(self):

        cv2.destroyAllWindows()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = HogSvmDetector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
