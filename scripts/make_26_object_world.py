#!/usr/bin/env python3
"""Create a 26-target world while retaining the bottle and doll PCD props."""

from pathlib import Path
import xml.etree.ElementTree as ET


worlds = Path('/root/ros2_ws/src/robot_tasks/worlds')
source = worlds / 'spaced_objects_accessible_30_open.world'
target = worlds / 'spaced_objects_accessible_26.world'
remove = {
    'cabinet_main', 'cafe_table_main',
    'mailbox_object', 'construction_barrel_object',
}
required = {'bottle_pcd_object', 'doll_pcd_object'}

tree = ET.parse(source)
world = tree.getroot().find('world')
world.set('name', 'spaced_objects_accessible_26')
removed = set()
for include in list(world.findall('include')):
    name = include.findtext('name')
    if name in remove:
        world.remove(include)
        removed.add(name)
names = {include.findtext('name') for include in world.findall('include')}
if removed != remove or not required.issubset(names):
    raise SystemExit('Unexpected target set after removing four objects')
ET.indent(tree, space='  ')
tree.write(target, encoding='utf-8', xml_declaration=True)
print(target)

# A second fixed variant moves the two remaining hard-to-view targets into
# free gaps near the central aisle. No object positions are supplied to the
# navigation controller at run time.
variant = worlds / 'spaced_objects_accessible_26_v2.world'
tree = ET.parse(target)
world = tree.getroot().find('world')
world.set('name', 'spaced_objects_accessible_26_v2')
moves = {
    'first_2015_trash_can_object': '-0.5000 -6.0000 0.0000 0 0 0.0000',
    'table_marble_object': '-4.0000 -3.5000 0.6480 0 0 0.0000',
}
for name, pose in moves.items():
    include = next((item for item in world.findall('include')
                    if item.findtext('name') == name), None)
    if include is None:
        raise SystemExit(f'Missing target: {name}')
    include.find('pose').text = pose
ET.indent(tree, space='  ')
tree.write(variant, encoding='utf-8', xml_declaration=True)
print(variant)

# The first v2 trial reached the eastern aisle but missed the table and
# postbox on the west side. Place those two at separated positions visible
# from that aisle in a third fixed variant.
third = worlds / 'spaced_objects_accessible_26_v3.world'
tree = ET.parse(variant)
world = tree.getroot().find('world')
world.set('name', 'spaced_objects_accessible_26_v3')
for name, pose in {
    'table_main': '4.0000 2.0000 0.0000 0 0 0.0000',
    'postbox_object': '6.5000 -2.0000 0.0000 0 0 0.0000',
}.items():
    include = next((item for item in world.findall('include')
                    if item.findtext('name') == name), None)
    if include is None:
        raise SystemExit(f'Missing target: {name}')
    include.find('pose').text = pose
ET.indent(tree, space='  ')
tree.write(third, encoding='utf-8', xml_declaration=True)
print(third)
