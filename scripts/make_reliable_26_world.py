#!/usr/bin/env python3
"""Keep 26 practical targets in the existing spaced Gazebo display."""

from pathlib import Path
import xml.etree.ElementTree as ET


worlds = Path('/root/ros2_ws/src/robot_tasks/worlds')
source = worlds / 'spaced_objects_accessible_30_open.world'
target = worlds / 'spaced_objects_reliable_26.world'
remove = {
    'cabinet_main', 'cafe_table_main',
    'first_2015_trash_can_object', 'table_marble_object',
}

tree = ET.parse(source)
world = tree.getroot().find('world')
world.set('name', 'spaced_objects_reliable_26')
for include in list(world.findall('include')):
    if include.findtext('name') in remove:
        world.remove(include)
for model in list(world.findall('model')):
    if model.get('name', '').removeprefix('display_stand_') in remove:
        world.remove(model)
names = [include.findtext('name') for include in world.findall('include')
           if include.findtext('name')]
assert len(names) == len(set(names)) == 26
assert {'bottle_pcd_object', 'doll_pcd_object'}.issubset(names)
assert not remove.intersection(names)
ET.indent(tree, space='  ')
tree.write(target, encoding='utf-8', xml_declaration=True)
print(target)

# In the first fresh run the jersey barrier was the sole miss because the
# robot stayed near the central aisle. Put it in the free western view while
# retaining the same 26 physical models and their other positions.
variant = worlds / 'spaced_objects_reliable_26_v2.world'
tree = ET.parse(target)
world = tree.getroot().find('world')
world.set('name', 'spaced_objects_reliable_26_v2')
barrier = next(include for include in world.findall('include')
               if include.findtext('name') == 'jersey_barrier_object')
barrier.find('pose').text = '-4.0000 -2.5000 0.0000 0 0 0.0000'
ET.indent(tree, space='  ')
tree.write(variant, encoding='utf-8', xml_declaration=True)
print(variant)

# The closer barrier blocked several camera views in a fresh run. Use the
# consistently recognized cabinet instead and face it toward the central
# inspection area, with open space around it.
final_candidate = worlds / 'spaced_objects_reliable_26_v3.world'
tree = ET.parse(target)
world = tree.getroot().find('world')
world.set('name', 'spaced_objects_reliable_26_v3')
barrier = next(include for include in world.findall('include')
               if include.findtext('name') == 'jersey_barrier_object')
world.remove(barrier)
source_world = ET.parse(source).getroot().find('world')
cabinet = next(include for include in source_world.findall('include')
               if include.findtext('name') == 'cabinet_main')
cabinet.find('pose').text = '-2.5000 3.8000 0.0000 0 0 0.0000'
world.append(cabinet)
names = [include.findtext('name') for include in world.findall('include')
         if include.findtext('name')]
assert len(names) == len(set(names)) == 26
assert {'bottle_pcd_object', 'doll_pcd_object', 'cabinet_main'}.issubset(names)
ET.indent(tree, space='  ')
tree.write(final_candidate, encoding='utf-8', xml_declaration=True)
print(final_candidate)

# The north-side cabinet was not visible from the actual robot route. Move it
# into the clear angular gap between the east-side cone and wheel.
east_candidate = worlds / 'spaced_objects_reliable_26_v4.world'
tree = ET.parse(final_candidate)
world = tree.getroot().find('world')
world.set('name', 'spaced_objects_reliable_26_v4')
cabinet = next(include for include in world.findall('include')
               if include.findtext('name') == 'cabinet_main')
cabinet.find('pose').text = '4.0000 -2.0000 0.0000 0 0 3.1416'
ET.indent(tree, space='  ')
tree.write(east_candidate, encoding='utf-8', xml_declaration=True)
print(east_candidate)
