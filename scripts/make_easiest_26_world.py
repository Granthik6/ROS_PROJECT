#!/usr/bin/env python3
"""Build a 26-target showcase from classes with reliable saved detections."""

from pathlib import Path
import xml.etree.ElementTree as ET


root = Path('/root/ros2_ws/src/robot_tasks/worlds')
source = root / 'spaced_objects_accessible_v17.world'
target = root / 'spaced_objects_easiest_26.world'
remove = {
    'ball_bearing_object', 'first_2015_trash_can_object',
    'cafe_table_main', 'bookshelf_main', 'fire_hose_long_object',
    'table_main', 'can_object', 'cup_object',
    'monkey_wrench_object', 'drill_object',
}
required = {'bottle_pcd_object', 'doll_pcd_object'}

tree = ET.parse(source)
world = tree.getroot().find('world')
world.set('name', 'spaced_objects_easiest_26')
removed = set()
for include in list(world.findall('include')):
    name = include.findtext('name')
    if name in remove:
        world.remove(include)
        removed.add(name)
for model in list(world.findall('model')):
    name = model.get('name', '')
    if name.startswith('display_stand_') and name.removeprefix('display_stand_') in remove:
        world.remove(model)
names = {include.findtext('name') for include in world.findall('include')}
if removed != remove or not required.issubset(names):
    raise SystemExit('Unexpected target set')
ET.indent(tree, space='  ')
tree.write(target, encoding='utf-8', xml_declaration=True)
print(target)

second = root / 'spaced_objects_easiest_26_v2.world'
tree = ET.parse(target)
world = tree.getroot().find('world')
world.set('name', 'spaced_objects_easiest_26_v2')
for name, pose in {
    'jersey_barrier_object': '-4.0000 -3.5000 0.0000 0 0 0.0000',
    'stop_light_object': '6.5000 -2.0000 0.0000 0 0 0.0000',
}.items():
    include = next((item for item in world.findall('include')
                    if item.findtext('name') == name), None)
    if include is None:
        raise SystemExit(f'Missing target: {name}')
    include.find('pose').text = pose
ET.indent(tree, space='  ')
tree.write(second, encoding='utf-8', xml_declaration=True)
print(second)
