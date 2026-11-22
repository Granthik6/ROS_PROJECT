#!/usr/bin/env python3
"""Create fixed 30-object evaluation worlds from the v17 layout.

Each variant removes six objects missed in saved trials. Bottle and doll PCD
objects are required and remain in both worlds.
"""

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path('/root/ros2_ws/src/robot_tasks/worlds')
source = ROOT / 'spaced_objects_accessible_v17.world'
variants = {
    'spaced_objects_accessible_30.world': {
        'bookshelf_main', 'stop_light_object', 'fire_hose_long_object',
        'cup_object', 'can_object', 'ball_bearing_object',
    },
    'spaced_objects_accessible_30_v2.world': {
        'cup_object', 'can_object', 'ball_bearing_object',
        'cafe_table_main', 'first_2015_trash_can_object',
        'monkey_wrench_object',
    },
    'spaced_objects_accessible_30_stands.world': {
        'bookshelf_main', 'stop_light_object', 'fire_hose_long_object',
        'cup_object', 'can_object', 'ball_bearing_object',
    },
}
required = {'bottle_pcd_object', 'doll_pcd_object'}

for filename, remove in variants.items():
    tree = ET.parse(source)
    world = tree.getroot().find('world')
    world.set('name', Path(filename).stem)
    removed = set()
    for include in list(world.findall('include')):
        name = include.findtext('name')
        if name in remove:
            world.remove(include)
            removed.add(name)
    if filename != 'spaced_objects_accessible_30_stands.world':
        for model in list(world.findall('model')):
            name = model.get('name', '')
            if (name.startswith('display_stand_') and
                    name.removeprefix('display_stand_') in remove):
                world.remove(model)
    if removed != remove:
        raise SystemExit(f'Expected six target removals, got {sorted(removed)}')
    names = {include.findtext('name') for include in world.findall('include')}
    if not required.issubset(names):
        raise SystemExit('Bottle or doll PCD object is missing')
    target = ROOT / filename
    ET.indent(tree, space='  ')
    tree.write(target, encoding='utf-8', xml_declaration=True)
    print(target)

# Keep the 30-target set and the empty stands, while giving four difficult
# targets clear, separated views. These coordinates are only in the world;
# neither the policy nor coverage assist receives them for navigation.
tree = ET.parse(ROOT / 'spaced_objects_accessible_30_stands.world')
world = tree.getroot().find('world')
world.set('name', 'spaced_objects_accessible_30_open')
moves = {
    'table_main': ('-2.5000 -5.6000 0.0000 0 0 0.0000', None),
    'first_2015_trash_can_object': ('5.2000 2.0000 0.0000 0 0 0.0000', None),
    'grey_tote_object': ('3.5000 4.6000 0.0250 0 0 0.0000', None),
    'cricket_ball_object': (
        '0.5000 0.5000 0.3975 0.0000 0.0000 0.0000',
        '0.5000 0.5000 0 0 0 0'),
}
for name, (pose, stand_pose) in moves.items():
    include = next((item for item in world.findall('include')
                    if item.findtext('name') == name), None)
    if include is None:
        raise SystemExit(f'Missing target: {name}')
    include.find('pose').text = pose
    if stand_pose is not None:
        stand = world.find(f"./model[@name='display_stand_{name}']/pose")
        if stand is None:
            raise SystemExit(f'Missing stand: {name}')
        stand.text = stand_pose
target = ROOT / 'spaced_objects_accessible_30_open.world'
ET.indent(tree, space='  ')
tree.write(target, encoding='utf-8', xml_declaration=True)
print(target)
