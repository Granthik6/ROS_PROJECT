"""Try clear, size-aware diagnostic viewpoints for missed Gazebo objects.

Known object locations select camera poses only in this detector audit.
The output must not be presented as autonomous exploration performance.
"""

import argparse
import csv
import json
import math
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import rclpy
from rclpy.parameter import Parameter
from cv_bridge import CvBridge
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).parent))
import generate_yolo_dataset as gyd
from audit_object_visibility import move_robot, spin_for
from robot_tasks.world_coverage_report import read_world
from robot_tasks.world_frame import detection_in_world


def world_specs(path):
    instances = read_world(path)
    poses = {}
    root = ET.parse(path).getroot().find('world')
    for include in root.findall('include'):
        name = include.findtext('name')
        if name:
            poses[name] = [float(value) for value in
                           include.findtext('pose').split()]
    templates = {obj['entity']: obj for obj in gyd.OBJECTS}
    specs = []
    for instance in instances:
        template = templates.get(instance['name'])
        if template is None and instance['class'] in ('bottle', 'doll'):
            template = {'entity': instance['name'],
                        'cls': instance['class'],
                        'half': (0.10, 0.10, 0.18),
                        'offset': (0.0, 0.0, 0.18)}
        if template is None:
            raise ValueError(f'Missing object geometry for {instance["name"]}')
        spec = dict(template)
        spec['pos'] = (instance['x'], instance['y'])
        spec['yaw'] = poses[instance['name']][5]
        # Template geometry is relative to the model origin. Some worlds
        # raise models above the floor, so include that height in projection.
        ox, oy, oz = spec['offset']
        spec['offset'] = (ox, oy, oz + poses[instance['name']][2])
        if instance['class'] == 'ball_bearing':
            # The Gazebo sphere radius is 3 mm, not the 3 cm used by the
            # older approximate template. Its display stand permits closer
            # camera views that keep the tiny sphere in frame.
            spec['half'] = (0.003, 0.003, 0.003)
            spec['offset'] = (0.0, 0.0, 0.003 + poses[instance['name']][2])
            spec['distances'] = [0.30, 0.40, 0.55, 0.75]
        if (instance['name'] == 'cricket_ball_object' and
                poses[instance['name']][2] >= 0.30):
            spec['half'] = (0.0375, 0.0375, 0.0375)
            spec['offset'] = (0.0, 0.0,
                              poses[instance['name']][2] + 0.0375)
            spec['distances'] = [0.45, 0.60, 0.80, 1.10]
        if poses[instance['name']][2] >= 0.30:
            raised_distances = {
                'monkey_wrench_object': [0.50, 0.70, 1.00, 1.30],
                'hokuyo_object': [0.35, 0.50, 0.70, 0.90],
                'saucepan_object': [0.80, 1.10, 1.50, 2.00],
                'gear_part_object': [0.40, 0.55, 0.75, 1.00],
            }
            spec['distances'] = raised_distances.get(
                instance['name'], spec.get('distances'))
            if instance['name'] == 'hokuyo_object':
                spec['offset'] = (ox, oy, oz + poses[instance['name']][2] -
                                  0.0351)
            elif instance['name'] == 'saucepan_object':
                spec['offset'] = (ox, oy, oz + poses[instance['name']][2] -
                                  0.0290)
        specs.append(spec)
    return specs, instances


def candidate_views(spec, attempts):
    random.seed(42)
    viewpoints = gyd.sample_object_viewpoints(spec, n_angles=24)
    scored = []
    for x, y, yaw, _ in viewpoints:
        if not gyd.is_free(x, y):
            continue
        camera_pos, rotation = gyd.compute_camera_pose(x, y, yaw)
        boxes = gyd.compute_visible_boxes(camera_pos, rotation)
        target = next((box for box in boxes
                       if box[0]['entity'] == spec['entity']), None)
        if target is None:
            continue
        _, x1, y1, x2, y2, _ = target
        width, height = x2 - x1, y2 - y1
        center_penalty = abs((x1 + x2) / 2 - gyd.CX) / gyd.IMG_W
        area = width * height / (gyd.IMG_W * gyd.IMG_H)
        clipped = (x1 <= 5 or y1 <= 5 or
                   x2 >= gyd.IMG_W - 6 or y2 >= gyd.IMG_H - 6)
        # A close view that cuts off the furniture is a poor detector test.
        # Prefer an object occupying up to about 15% of the frame; for tiny
        # objects, this still chooses the largest available image.
        score = (area - 3.0 * max(0.0, area - 0.15) -
                 0.15 * center_penalty - 0.30 * clipped)
        angle = math.atan2(y - spec['pos'][1], x - spec['pos'][0])
        scored.append({
            'row': (score, x, y, yaw, [round(x1), round(y1),
                                      round(x2), round(y2)]),
            'score': score,
            'range': math.hypot(x - spec['pos'][0], y - spec['pos'][1]),
            'angle': angle,
            'clipped': clipped,
            'area': area,
        })
    scored.sort(key=lambda item: item['score'], reverse=True)
    if not scored:
        return []
    selected = [scored.pop(0)]
    if attempts > 1:
        # A close, complete view often works when a distant view has too few
        # pixels. Keep one of each rather than three nearly identical ranges.
        complete = [item for item in scored if
                    not item['clipped'] and item['area'] <= 0.35]
        pool = complete or scored
        distinct = [item for item in pool if
                    abs(math.atan2(math.sin(item['angle'] - selected[0]['angle']),
                                   math.cos(item['angle'] - selected[0]['angle']))) > 0.45
                    or abs(item['range'] - selected[0]['range']) > 0.35]
        pool = distinct or pool
        near = min(pool, key=lambda item: (item['range'], -item['score']))
        selected.append(near)
        scored.remove(near)
    while scored and len(selected) < attempts:
        def diversity(item):
            angular = min(abs(math.atan2(
                math.sin(item['angle'] - other['angle']),
                math.cos(item['angle'] - other['angle'])))
                          for other in selected)
            distance = min(abs(item['range'] - other['range'])
                           for other in selected)
            return (item['score'] + 0.06 * angular +
                    0.03 * min(distance, 2.0))
        next_item = max(scored, key=diversity)
        selected.append(next_item)
        scored.remove(next_item)
    return [item['row'] for item in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', required=True)
    parser.add_argument('--missing-list', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--view-seconds', type=float, default=4.0)
    parser.add_argument('--views-per-object', type=int, default=3)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save-images', action='store_true')
    args = parser.parse_args()
    specs, instances = world_specs(args.world)
    by_name = {obj['name']: obj for obj in instances}
    missing = json.loads(args.missing_list.read_text())['missing_objects']
    targets = [spec for spec in specs if spec['entity'] in missing]
    if len(targets) != len(missing):
        raise ValueError('Missing-list names do not match world objects')
    gyd.OBJECTS = specs
    world_models = ET.parse(args.world).getroot().find('world').findall('model')
    if any(model.get('name', '').startswith('search_boundary_')
           for model in world_models):
        gyd.ROOM_X_MIN, gyd.ROOM_X_MAX = -8.75, 8.75
        gyd.ROOM_Y_MIN, gyd.ROOM_Y_MAX = -10.75, 6.75
    else:
        gyd.ROOM_X_MIN, gyd.ROOM_X_MAX = -15.0, 15.0
        gyd.ROOM_Y_MIN, gyd.ROOM_Y_MAX = -15.0, 10.0
    gyd.OBSTACLES = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state = {'odom': None, 'robot': None, 'image': None,
             'target': None, 'hits': [], 'raw_hits': 0,
             'min_stamp': 0.0}
    rclpy.init()
    node = Node('planned_view_detector_audit')
    node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
    bridge = CvBridge()
    publisher = node.create_publisher(Twist, '/cmd_vel', 10)
    client = node.create_client(SetEntityState, '/gazebo/set_entity_state')

    def on_odom(message):
        state['odom'] = message.pose.pose

    def on_models(message):
        try:
            state['robot'] = message.pose[message.name.index('burger')]
        except ValueError:
            pass

    def on_detection(message):
        target = state['target']
        if target is None:
            return
        try:
            detection = json.loads(message.data)
            if detection['class'] != target['class']:
                return
            confidence = float(detection['confidence'])
            stamp_s = float(detection.get('stamp', 0.0))
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            return
        if args.debug:
            print('detection', target['name'], confidence,
                  'image_stamp', stamp_s,
                  'min_stamp', state['min_stamp'],
                  'has_odom_xyz', 'odom_xyz' in detection, flush=True)
        threshold = (0.20 if target['class'] in ('bottle', 'doll') else
                     0.18 if target['class'] == 'fire_hose_long' else
                     0.12 if target['class'] == 'ball_bearing' else
                     0.15 if target['class'] in ('cup', 'gear_part', 'coke_can')
                     else 0.25)
        if confidence < threshold:
            return
        if stamp_s < state['min_stamp']:
            return
        state['raw_hits'] += 1
        xyz = detection.get('odom_xyz')
        if not isinstance(xyz, list) or len(xyz) != 3:
            return
        if state['odom'] is None or state['robot'] is None:
            return
        try:
            point = detection_in_world(xyz, state['odom'], state['robot'])
        except (TypeError, ValueError):
            return
        if math.hypot(point[0] - target['x'],
                      point[1] - target['y']) <= 1.25:
            state['hits'].append(str(detection.get('stamp', '')))
        elif args.debug:
            print('position_error_m',
                  math.hypot(point[0] - target['x'],
                             point[1] - target['y']),
                  'point', point[:2],
                  'odom_xyz', xyz[:2],
                  'odom_robot', (state['odom'].position.x,
                                 state['odom'].position.y),
                  'world_robot', (state['robot'].position.x,
                                  state['robot'].position.y),
                  'camera_xyz', detection.get('camera_xyz'), flush=True)

    node.create_subscription(Odometry, '/odom', on_odom, 10)
    node.create_subscription(ModelStates, '/gazebo/model_states', on_models, 1)
    node.create_subscription(String, '/yolo/detections', on_detection, 50)
    if args.save_images:
        node.create_subscription(
            Image, '/camera/rgb/image_raw',
            lambda message: state.__setitem__('image', message),
            qos_profile_sensor_data,
        )
    results = []
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if (state['odom'] is not None and state['robot'] is not None and
                    node.count_publishers('/yolo/detections') and
                    client.service_is_ready()):
                break
        else:
            raise RuntimeError('Gazebo or YOLO did not become ready')

        for number, spec in enumerate(targets, 1):
            target = by_name[spec['entity']]
            rows = []
            candidates = candidate_views(spec, args.views_per_object)
            for index, (_, x, y, yaw, box) in enumerate(candidates, 1):
                publisher.publish(Twist())
                if not move_robot(node, client, x, y, yaw):
                    continue
                state['target'] = None
                state['min_stamp'] = node.get_clock().now().nanoseconds * 1e-9
                spin_for(node, 0.8)
                state['hits'], state['raw_hits'] = [], 0
                state['target'] = target
                deadline = time.monotonic() + args.view_seconds
                while time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.05)
                    if len(set(state['hits'])) >= 2:
                        break
                state['target'] = None
                frame_path = None
                if state['image'] is not None:
                    frame_path = args.output_dir / f'{spec["entity"]}_{index}.png'
                    cv2.imwrite(str(frame_path),
                                bridge.imgmsg_to_cv2(state['image'], 'bgr8'))
                rows.append({'view': index, 'pose': [x, y, yaw],
                             'expected_box': box,
                             'raw_class_hits': state['raw_hits'],
                             'localized_frames': len(set(state['hits'])),
                             'image': str(frame_path) if frame_path else None})
                if args.debug:
                    print('view_result', spec['entity'], index,
                          state['raw_hits'], len(set(state['hits'])), flush=True)
                if len(set(state['hits'])) >= 2:
                    break
            confirmed = any(row['localized_frames'] >= 2 for row in rows)
            results.append({'name': spec['entity'], 'class': target['class'],
                            'confirmed': confirmed, 'views': rows})
            print(f'{number}/{len(targets)} {spec["entity"]}: '
                  f'{"visible" if confirmed else "MISS"} '
                  f'({len(rows)} views)', flush=True)
    finally:
        publisher.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

    report = {'diagnostic': 'known_position_planned_views_not_autonomous',
              'world_file': str(Path(args.world).resolve()),
              'target_objects': len(targets),
              'confirmed_objects': sum(row['confirmed'] for row in results),
              'missing_objects': [row['name'] for row in results
                                  if not row['confirmed']],
              'objects': results}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.output_dir / 'report.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('name', 'class', 'confirmed', 'views', 'raw_hits',
                         'localized_frames'))
        for row in results:
            writer.writerow((row['name'], row['class'], row['confirmed'],
                             len(row['views']),
                             sum(v['raw_class_hits'] for v in row['views']),
                             sum(v['localized_frames'] for v in row['views'])))
    print(json.dumps({key: value for key, value in report.items()
                      if key != 'objects'}, indent=2))


if __name__ == '__main__':
    main()
