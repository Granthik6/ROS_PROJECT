"""Build an open Gazebo layout with all 36 objects in camera-visible arcs.

This is an alternate evaluation world. It preserves the original models and
their floor offsets, and never supplies model positions to the DDQN policy.
"""

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path('/root/ros2_ws')
SOURCE = ROOT / 'src/robot_tasks/worlds/realistic_room.world'
OUTPUT = ROOT / 'src/robot_tasks/worlds/ddqn_showcase.world'
CENTER = (0.0, -2.0)

LARGE = {
    'bookshelf', 'cabinet', 'table', 'cafe_table', 'person_standing',
    'person_walking', 'construction_barrel', 'mailbox', 'postbox',
    'stop_sign', 'lamp_post', 'first_2015_trash_can', 'grey_tote',
    'jersey_barrier', 'stop_light',
    'fire_hose_long',
}
SMALL = {
    'plastic_cup', 'bowl', 'coke_can', 'cordless_drill', 'hammer',
    'beer', 'cricket_ball', 'monkey_wrench', 'hokuyo', 'saucepan',
    'ball_bearing', 'gear_part', 'pcd_bottle',
    'pcd_doll',
}


def build():
    source_world = ET.parse(SOURCE).getroot().find('world')
    if source_world is None:
        raise ValueError(f'World missing from {SOURCE}')
    original = []
    for include in source_world.findall('include'):
        model_type = include.findtext('uri', '').rsplit('/', 1)[-1]
        if model_type not in ('sun', 'ground_plane'):
            original.append((model_type, include))
    if len(original) != 36:
        raise ValueError(f'Expected 36 placed models; found {len(original)}')

    large = [entry for entry in original if entry[0] in LARGE]
    other = [entry for entry in original if entry[0] not in LARGE]
    slots = [None] * len(original)
    # Spread bulky models across the outside ring to preserve sight lines.
    for index, entry in enumerate(large):
        slots[index * 2] = entry
    others = iter(other)
    for index, entry in enumerate(slots):
        if entry is None:
            slots[index] = next(others)

    root = ET.Element('sdf', version='1.6')
    world = ET.SubElement(root, 'world', name='ddqn_showcase')
    for name in ('plugin', 'include', 'include', 'physics'):
        element = next((item for item in source_world if item.tag == name), None)
        if element is None:
            raise ValueError(f'Missing {name} in {SOURCE}')
        world.append(copy.deepcopy(element))
        source_world.remove(element)
    for index, (model_type, source_include) in enumerate(slots):
        include = copy.deepcopy(source_include)
        pose = include.find('pose')
        values = [float(value) for value in pose.text.split()]
        angle = -math.pi / 2 + 2 * math.pi * index / len(slots)
        radius = 5.0 if model_type == 'fire_hose_long' else (
            3.1 if model_type in LARGE else (
            1.8 if model_type in SMALL else 2.4)
        )
        x = CENTER[0] + radius * math.cos(angle)
        y = CENTER[1] + radius * math.sin(angle)
        yaw = (angle + math.pi / 2 if model_type == 'fire_hose_long'
               else math.atan2(CENTER[1] - y, CENTER[0] - x))
        pose.text = f'{x:.4f} {y:.4f} {values[2]:.4f} 0 0 {yaw:.4f}'
        world.append(include)
    ET.indent(root, space='  ')
    tree = ET.ElementTree(root)
    tree.write(OUTPUT, encoding='unicode', xml_declaration=True)
    print(f'Wrote {len(slots)} objects to {OUTPUT}')


if __name__ == '__main__':
    build()
