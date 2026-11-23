import rclpy

from rclpy.node import Node

from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np


class CameraNode(Node):

    def __init__(self):

        super().__init__('camera_node')

        self.bridge = CvBridge()


        self.rgb_subscription = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.rgb_callback,
            10
        )


        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        self.rgb_frame = None

        self.depth_frame = None

        self.get_logger().info(
            'RGB-D camera node started.'
        )

        self.get_logger().info(
            'RGB topic: /camera/rgb/image_raw'
        )

        self.get_logger().info(
            'Depth topic: /camera/depth/image_raw'
        )



    def rgb_callback(self, msg):

        try:

            self.rgb_frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            self.display_images()

        except CvBridgeError as error:

            self.get_logger().error(
                f'RGB CvBridge error: {error}'
            )



    def depth_callback(self, msg):

        try:

            depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='32FC1'
            )

            self.depth_frame = depth

            self.display_images()

        except CvBridgeError as error:

            self.get_logger().error(
                f'Depth CvBridge error: {error}'
            )



    def display_images(self):


        if self.rgb_frame is not None:

            rgb_display = self.rgb_frame.copy()

            cv2.imshow(
                'RGB Camera',
                rgb_display
            )



        if self.depth_frame is not None:

            depth = self.depth_frame.copy()

            valid = np.isfinite(depth)

            if np.any(valid):

                min_depth = np.percentile(
                    depth[valid],
                    2
                )

                max_depth = np.percentile(
                    depth[valid],
                    98
                )

                if max_depth > min_depth:

                    depth_normalized = np.clip(
                        (depth - min_depth)
                        / (max_depth - min_depth),
                        0.0,
                        1.0
                    )

                    depth_display = (
                        depth_normalized * 255
                    ).astype(np.uint8)

                else:

                    depth_display = np.zeros(
                        depth.shape,
                        dtype=np.uint8
                    )

            else:

                depth_display = np.zeros(
                    depth.shape,
                    dtype=np.uint8
                )


            depth_display = cv2.applyColorMap(
                depth_display,
                cv2.COLORMAP_JET
            )

            cv2.imshow(
                'Depth Camera',
                depth_display
            )


        cv2.waitKey(1)



    def destroy_node(self):

        cv2.destroyAllWindows()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = CameraNode()

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
