#!/usr/bin/env python3
"""Arrange 26 selected objects in two spaced display rings.

The layout leaves the robot's start at (0, -2) open. The ring coordinates
are fixed world design only; the detector and DDQN navigation receive no
target poses while running.
"""

import math
from pathlib import Path
import xml.etree.ElementTree as ET


root = Path('/root/ros2_ws/src/robot_tasks/worlds')
source = root / 'spaced_objects_easiest_26.world'
target = root / 'spaced_objects_showcase_26.world'
outer = [
    'cabinet_main', 'person_standing_main', 'person_walking_main',
    'construction_barrel_object', 'mailbox_object', 'postbox_object',
    'stop_sign_object', 'lamp_post_object', 'grey_tote_object',
    'table_marble_object', 'jersey_barrier_object', 'stop_light_object',
    'wooden_case_object',
]
inner = [
    'beer_object', 'bottle_pcd_object', 'bowl_object',
    'car_wheel_object', 'cardboard_box_object',
    'construction_cone_object', 'cricket_ball_object',
    'doll_pcd_object', 'fire_hydrant_object', 'gear_part_object',
    'hammer_object', 'hokuyo_object', 'saucepan_object',
]
assert len(outer) == len(inner) == 13

tree = ET.parse(source)
world = tree.getroot().find('world')
world.set('name', 'spaced_objects_showcase_26')
includes = {item.findtext('name'): item for item in world.findall('include')}
if set(includes) - {None} != set(outer + inner):
    raise SystemExit('The selected source world has changed')

for ring, radius, offset in ((outer, 5.8, 0.0),
                             (inner, 3.0, math.pi / 13)):
    for index, name in enumerate(ring):
        angle = 2 * math.pi * index / 13 + offset
        x, y = radius * math.cos(angle), -2.0 + radius * math.sin(angle)
        pose = includes[name].find('pose')
        fields = pose.text.split()
        fields[0], fields[1] = f'{x:.4f}', f'{y:.4f}'
        pose.text = ' '.join(fields)
        stand = world.find(f"./model[@name='display_stand_{name}']/pose")
        if stand is not None:
            stand_fields = stand.text.split()
            stand_fields[0], stand_fields[1] = f'{x:.4f}', f'{y:.4f}'
            stand.text = ' '.join(stand_fields)

ET.indent(tree, space='  ')
tree.write(target, encoding='utf-8', xml_declaration=True)
print(target)

# Put small props where the camera can inspect them from the open western
# aisle. Preserve the rest of the two-ring layout, including both PCD props.
variant = root / 'spaced_objects_showcase_26_v2.world'
tree = ET.parse(target)
world = tree.getroot().find('world')
world.set('name', 'spaced_objects_showcase_26_v2')
reposition = {
    'hammer_object': (-3.8, 2.8),
    'hokuyo_object': (1.2, 2.3),
    'saucepan_object': (-0.5, -0.8),
    'gear_part_object': (-4.5, -1.8),
}
for name, (x, y) in reposition.items():
    include = next(item for item in world.findall('include')
                   if item.findtext('name') == name)
    pose = include.find('pose')
    fields = pose.text.split()
    fields[0], fields[1] = f'{x:.4f}', f'{y:.4f}'
    pose.text = ' '.join(fields)
    stand = world.find(f"./model[@name='display_stand_{name}']/pose")
    if stand is not None:
        stand_fields = stand.text.split()
        stand_fields[0], stand_fields[1] = f'{x:.4f}', f'{y:.4f}'
        stand.text = ' '.join(stand_fields)
ET.indent(tree, space='  ')
tree.write(variant, encoding='utf-8', xml_declaration=True)
print(variant)
