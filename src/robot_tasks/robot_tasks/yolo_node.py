import json
import math
import os
import struct
import time
from collections import deque

import cv2
import message_filters
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32, String
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

from ultralytics import YOLO


DEFAULT_MODEL_PATH = (
    '/root/ros2_ws/realistic_sim_yolo_runs/'
    'realistic_sim_yolo11n/weights/best.pt'
)
DEFAULT_PCD_MODEL_PATH = (
    '/root/ros2_ws/pcd_yolo_runs/pcd_bottle_doll_v3/weights/best.pt'
)


class YoloNode(Node):

    def __init__(self):

        super().__init__('yolo_node')

        self.declare_parameter('model_path', DEFAULT_MODEL_PATH)
        self.declare_parameter('person_fallback_model_path', '')
        self.declare_parameter('person_fallback_interval_frames', 5)
        self.declare_parameter('hose_fallback_model_path', '')
        self.declare_parameter('hose_fallback_interval_frames', 2)
        self.declare_parameter('pcd_model_path', DEFAULT_PCD_MODEL_PATH)
        self.declare_parameter('pcd_interval_frames', 1)
        self.declare_parameter('pcd_conf_threshold', 0.20)
        self.declare_parameter('show_window', True)
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('inference_imgsz', 640)
        self.declare_parameter('max_image_age_s', 0.75)
        self.declare_parameter('log_detections', True)

        model_path = self.get_parameter('model_path').value
        self.show_window = self.get_parameter('show_window').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.inference_imgsz = max(
            320, int(self.get_parameter('inference_imgsz').value)
        )
        self.max_image_age_s = max(
            0.0, float(self.get_parameter('max_image_age_s').value)
        )
        self.log_detections = self.get_parameter('log_detections').value

        self.model = YOLO(model_path)
        person_model_path = str(
            self.get_parameter('person_fallback_model_path').value
        ).strip()
        self.person_model = YOLO(person_model_path) if person_model_path else None
        self.person_fallback_interval_frames = max(
            1, int(self.get_parameter('person_fallback_interval_frames').value)
        )
        hose_model_path = str(
            self.get_parameter('hose_fallback_model_path').value
        ).strip()
        self.hose_model = (
            YOLO(hose_model_path)
            if hose_model_path and os.path.isfile(hose_model_path)
            else None
        )
        self.hose_fallback_interval_frames = max(
            1, int(self.get_parameter('hose_fallback_interval_frames').value)
        )
        self.hose_class_id = (
            next((int(index) for index, name in self.hose_model.names.items()
                  if name == 'fire_hose_long'), None)
            if self.hose_model is not None else None
        )
        if self.hose_model is not None and self.hose_class_id is None:
            raise ValueError('Hose fallback model lacks fire_hose_long class')
        pcd_model_path = str(self.get_parameter('pcd_model_path').value).strip()
        self.pcd_model = (
            YOLO(pcd_model_path)
            if pcd_model_path and os.path.isfile(pcd_model_path)
            else None
        )
        self.pcd_interval_frames = max(
            1, int(self.get_parameter('pcd_interval_frames').value)
        )
        self.pcd_conf_threshold = float(
            self.get_parameter('pcd_conf_threshold').value
        )

        self.get_logger().info(
            f'Loaded YOLO model: {model_path}'
        )
        if self.person_model is not None:
            self.get_logger().info(
                f'Using COCO person detector every '
                f'{self.person_fallback_interval_frames} frames: '
                f'{person_model_path}'
            )
        if self.hose_model is not None:
            self.get_logger().info(
                f'Using fire hose detector every '
                f'{self.hose_fallback_interval_frames} frames: '
                f'{hose_model_path}'
            )
        if self.pcd_model is not None:
            self.get_logger().info(
                f'Using PCD bottle/doll detector: {pcd_model_path}; '
                f'classes={self.pcd_model.names}'
            )
        elif pcd_model_path:
            self.get_logger().warn(
                f'PCD bottle/doll weights absent: {pcd_model_path}'
            )

        self.bridge = CvBridge()

        self.odom_history = deque(maxlen=200)
        self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile_sensor_data
        )

        self.active_pointcloud = None

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.image_subscription = message_filters.Subscriber(
            self,
            Image,
            '/camera/rgb/image_raw',
            qos_profile=qos_profile_sensor_data,
        )

        self.pointcloud_subscription = message_filters.Subscriber(
            self,
            PointCloud2,
            '/camera/depth/points',
            qos_profile=qos_profile_sensor_data,
        )
        self.camera_sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_subscription, self.pointcloud_subscription],
            queue_size=5,
            slop=0.15,
        )
        self.camera_sync.registerCallback(self.image_callback)

        self.detections_publisher = self.create_publisher(
            String,
            '/yolo/detections',
            10
        )

        self.confidence_publisher = self.create_publisher(
            Float32,
            '/yolo/confidence',
            10
        )

        self.distance_publisher = self.create_publisher(
            Float32,
            '/yolo/distance',
            10
        )

        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0.0

    def odom_callback(self, message):
        stamp = (message.header.stamp.sec +
                 message.header.stamp.nanosec * 1e-9)
        self.odom_history.append((stamp, message.pose.pose))

    def camera_to_odom(self, camera_xyz, image_stamp_s):
        """Use camera geometry and odometry sampled at image capture time.

        Gazebo TF can retain an old camera rotation across SetEntityState
        teleports. The odom message itself tracks the current robot pose.
        The camera mount in turtlebot3_burger_camera/model.sdf is 0.105 m
        forward and 0.45 m high; optical X points right and Y points down.
        """
        if not self.odom_history:
            return None
        stamp, pose = min(self.odom_history,
                          key=lambda row: abs(row[0] - image_stamp_s))
        if abs(stamp - image_stamp_s) > 0.75:
            return None
        q = pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        right, down, forward = camera_xyz
        base_forward = 0.105 + forward
        base_left = -right
        c, s = math.cos(yaw), math.sin(yaw)
        return (
            pose.position.x + c * base_forward - s * base_left,
            pose.position.y + s * base_forward + c * base_left,
            pose.position.z + 0.45 - down,
        )

    def odom_to_map(self, odom_xyz):
        if self.active_pointcloud is None:
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'odom',
                rclpy.time.Time.from_msg(self.active_pointcloud.header.stamp),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            point = PointStamped()
            point.header.frame_id = 'odom'
            point.header.stamp = self.active_pointcloud.header.stamp
            point.point.x, point.point.y, point.point.z = odom_xyz
            mapped = do_transform_point(point, transform)
            return mapped.point.x, mapped.point.y, mapped.point.z
        except Exception:
            return None

    def get_xyz(self, u, v):

        msg = self.active_pointcloud

        if msg is None:
            return None

        if u < 0 or u >= msg.width:
            return None

        if v < 0 or v >= msg.height:
            return None

        try:

            offset = (
                v * msg.row_step +
                u * msg.point_step
            )

            x = struct.unpack_from(
                '<f',
                msg.data,
                offset
            )[0]

            y = struct.unpack_from(
                '<f',
                msg.data,
                offset + 4
            )[0]

            z = struct.unpack_from(
                '<f',
                msg.data,
                offset + 8
            )[0]

            if not math.isfinite(x):
                return None

            if not math.isfinite(y):
                return None

            if not math.isfinite(z):
                return None

            if z <= 0.1 or z > 10.0:
                return None

            return x, y, z

        except Exception:

            return None

    def get_box_xyz(self, x1, y1, x2, y2):
        """Robust depth/XYZ estimate for a bounding box: sample a grid of
        points inside the box, drop NaN/inf/zero/out-of-range readings,
        and return the per-axis median of what's left."""

        width = x2 - x1
        height = y2 - y1

        sample_points = []

        for fx in (0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90):
            for fy in (0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90):
                sample_points.append((
                    int(x1 + width * fx),
                    int(y1 + height * fy),
                ))

        points = []

        for u, v in sample_points:

            xyz = self.get_xyz(u, v)

            if xyz is not None:
                points.append(xyz)

        if len(points) < 3:
            return None

        nearest = sorted(points, key=lambda point: point[2])[
            :max(3, len(points) // 3)
        ]
        xs = sorted(point[0] for point in nearest)
        ys = sorted(point[1] for point in nearest)
        zs = sorted(point[2] for point in nearest)

        middle = len(nearest) // 2

        if len(nearest) % 2 == 1:

            median_x = xs[middle]
            median_y = ys[middle]
            median_z = zs[middle]

        else:

            median_x = (
                xs[middle - 1] +
                xs[middle]
            ) / 2.0

            median_y = (
                ys[middle - 1] +
                ys[middle]
            ) / 2.0

            median_z = (
                zs[middle - 1] +
                zs[middle]
            ) / 2.0

        return (
            median_x,
            median_y,
            median_z
        )

    def transform_point(self, x, y, z, target_frame):

        if self.active_pointcloud is None:
            return None

        source_frame = self.active_pointcloud.header.frame_id

        try:
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time.from_msg(
                        self.active_pointcloud.header.stamp
                    ),
                    timeout=rclpy.duration.Duration(seconds=0.1),
                )
            except Exception:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2),
                )

            point_camera = PointStamped()

            point_camera.header.frame_id = source_frame
            point_camera.header.stamp = self.active_pointcloud.header.stamp

            point_camera.point.x = x
            point_camera.point.y = y
            point_camera.point.z = z

            point_target = do_transform_point(
                point_camera,
                transform
            )

            return (
                point_target.point.x,
                point_target.point.y,
                point_target.point.z
            )

        except Exception:

            return None

    def image_callback(self, msg, pointcloud):

        image_stamp_s = (msg.header.stamp.sec +
                         msg.header.stamp.nanosec * 1e-9)
        current_stamp = self.get_clock().now().to_msg()
        current_s = current_stamp.sec + current_stamp.nanosec * 1e-9
        if (self.max_image_age_s and image_stamp_s > 0 and current_s > 0 and
                current_s - image_stamp_s > self.max_image_age_s):
            return

        self.active_pointcloud = pointcloud

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

        except Exception as error:

            self.get_logger().error(
                f'Image conversion failed: {error}'
            )

            return

        run_person_fallback = (
            self.person_model is not None and
            self.frame_count % self.person_fallback_interval_frames == 0
        )
        run_pcd_detector = (
            self.pcd_model is not None and
            self.frame_count % self.pcd_interval_frames == 0
        )
        run_hose_fallback = (
            self.hose_model is not None and
            self.frame_count % self.hose_fallback_interval_frames == 0
        )
        try:

            results = ([] if run_hose_fallback else self.model(
                frame,
                conf=self.conf_threshold,
                imgsz=self.inference_imgsz,
                verbose=False
            ))
            person_results = (
                self.person_model(
                    frame, conf=0.15, classes=[0], imgsz=640,
                    verbose=False,
                ) if run_person_fallback else []
            )
            hose_results = (
                self.hose_model(
                    frame, conf=self.conf_threshold,
                    imgsz=self.inference_imgsz, verbose=False,
                ) if run_hose_fallback else []
            )
            pcd_results = (
                self.pcd_model(
                    frame, conf=self.pcd_conf_threshold, imgsz=640,
                    verbose=False,
                ) if run_pcd_detector else []
            )

        except Exception as error:

            self.get_logger().error(
                f'YOLO inference failed: {error}'
            )

            return

        stamp = msg.header.stamp
        boxes = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_name = self.model.names[int(box.cls[0].item())]
                    if class_name != 'person' or not run_person_fallback:
                        if class_name != 'fire_hose_long' or self.hose_model is None:
                            boxes.append((box, class_name))
        for result in person_results:
            if result.boxes is not None:
                boxes.extend((box, 'person') for box in result.boxes)
        for result in hose_results:
            if result.boxes is not None:
                boxes.extend(
                    (box, 'fire_hose_long') for box in result.boxes
                    if int(box.cls[0].item()) == self.hose_class_id
                )
        for result in pcd_results:
            if result.boxes is not None:
                boxes.extend(
                    (box, self.pcd_model.names[int(box.cls[0].item())])
                    for box in result.boxes
                )

        for box, class_name in boxes:

                confidence = float(
                    box.conf[0].item()
                )

                self.confidence_publisher.publish(
                    Float32(data=confidence)
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                center_u = int(
                    (x1 + x2) / 2
                )

                center_v = int(
                    (y1 + y2) / 2
                )

                camera_xyz = self.get_box_xyz(
                    x1,
                    y1,
                    x2,
                    y2
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (center_u, center_v),
                    5,
                    (0, 0, 255),
                    -1
                )

                detection = {
                    'stamp': stamp.sec + stamp.nanosec * 1e-9,
                    'class': class_name,
                    'confidence': round(confidence, 4),
                    'bbox': [x1, y1, x2, y2],
                }

                if camera_xyz is None:

                    cv2.putText(
                        frame,
                        (
                            f'{class_name} '
                            f'{confidence:.2f} '
                            'Depth unavailable'
                        ),
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )

                    self.detections_publisher.publish(
                        String(data=json.dumps(detection))
                    )

                    continue

                camera_x, camera_y, camera_z = camera_xyz

                distance = math.sqrt(
                    camera_x ** 2 +
                    camera_y ** 2 +
                    camera_z ** 2
                )

                detection['camera_xyz'] = [
                    round(camera_x, 4),
                    round(camera_y, 4),
                    round(camera_z, 4),
                ]
                detection['distance_m'] = round(distance, 4)

                self.distance_publisher.publish(
                    Float32(data=distance)
                )

                odom_xyz = self.camera_to_odom(
                    camera_xyz, image_stamp_s
                )

                map_xyz = (self.odom_to_map(odom_xyz)
                           if odom_xyz is not None else None)

                if odom_xyz is not None:
                    detection['odom_xyz'] = [round(v, 4) for v in odom_xyz]

                if map_xyz is not None:
                    detection['map_xyz'] = [round(v, 4) for v in map_xyz]

                label = (
                    f'{class_name} '
                    f'{confidence:.2f}'
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 55, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                camera_text = (
                    f'C: '
                    f'{camera_x:.2f},'
                    f'{camera_y:.2f},'
                    f'{camera_z:.2f}m'
                )

                cv2.putText(
                    frame,
                    camera_text,
                    (x1, max(y1 - 35, 40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    2
                )

                if odom_xyz is not None:

                    odom_x, odom_y, odom_z = odom_xyz

                    odom_text = (
                        f'O: '
                        f'{odom_x:.2f},'
                        f'{odom_y:.2f},'
                        f'{odom_z:.2f}m'
                    )

                    cv2.putText(
                        frame,
                        odom_text,
                        (x1, max(y1 - 12, 60)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 0),
                        2
                    )

                self.detections_publisher.publish(
                    String(data=json.dumps(detection))
                )

                if self.log_detections:
                    self.get_logger().info(
                        f'{class_name} '
                        f'confidence={confidence:.3f} '
                        f'distance={distance:.3f}m '
                        f'camera_xyz=('
                        f'{camera_x:.3f}, '
                        f'{camera_y:.3f}, '
                        f'{camera_z:.3f}) '
                        f'odom_xyz={detection.get("odom_xyz", "unavailable")} '
                        f'map_xyz={detection.get("map_xyz", "unavailable")}'
                    )

        self.frame_count += 1

        elapsed = (
            time.time() -
            self.start_time
        )

        if elapsed >= 1.0:

            self.current_fps = self.frame_count / elapsed

            if self.log_detections:
                self.get_logger().info(
                    f'YOLO FPS: {self.current_fps:.2f}'
                )

            self.frame_count = 0
            self.start_time = time.time()

        if self.show_window:

            cv2.imshow(
                'YOLO RGB + Camera XYZ + Odom XYZ',
                frame
            )

            cv2.waitKey(1)

    def destroy_node(self):

        if self.show_window:
            cv2.destroyAllWindows()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = YoloNode()

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
