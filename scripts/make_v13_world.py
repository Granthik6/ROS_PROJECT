#!/usr/bin/env python3
"""Make a world with the hard-to-see walking person in an open aisle.

The new target pose is an environment design choice. It is not supplied to
the robot's navigation policy, coverage planner, or detector.
"""

from pathlib import Path


ROOT = Path('/root/ros2_ws/src/robot_tasks/worlds')
source = ROOT / 'spaced_objects_accessible_v9.world'
target = ROOT / 'spaced_objects_accessible_v13.world'
old = '<pose>7.0000 -9.0000 0.0000 0 0 -1.5000</pose>'
new = '<pose>1.8000 -6.8000 0.0000 0 0 1.5000</pose>'
text = source.read_text()
section = text.split('<name>person_walking_main</name>', 1)
if len(section) != 2 or old not in section[1].split('</include>', 1)[0]:
    raise SystemExit('Walking-person pose in v9 world changed unexpectedly')
head, tail = section
tail = tail.replace(old, new, 1)
target.write_text(head + '<name>person_walking_main</name>' + tail)
print(target)

# A further variant moves the cafe table into clear view from the central
# aisle. The table remains separate from the nearby can and beer displays.
v14 = ROOT / 'spaced_objects_accessible_v14.world'
v13_text = target.read_text()
cafe_head, separator, cafe_tail = v13_text.partition('<name>cafe_table_main</name>')
old_cafe = '<pose>-3.5000 -5.3000 0.0000 0 0 0.6000</pose>'
new_cafe = '<pose>-2.4000 -4.8000 0.0000 0 0 0.6000</pose>'
if not separator or old_cafe not in cafe_tail.split('</include>', 1)[0]:
    raise SystemExit('Cafe table pose in v13 world changed unexpectedly')
v14.write_text(cafe_head + separator + cafe_tail.replace(old_cafe, new_cafe, 1))
print(v14)

v15 = ROOT / 'spaced_objects_accessible_v15.world'
v15_text = v14.read_text()
replacements = {
    'first_2015_trash_can_object': (
        '<pose>3.1000 1.1000 0.0000 0 0 0.0000</pose>',
        '<pose>-5.0000 0.8000 0.0000 0 0 0.0000</pose>'),
    'fire_hose_long_object': (
        '<pose>2.5000 2.7000 0.1000 0 0 0.0000</pose>',
        '<pose>-1.8000 3.8000 0.1000 0 0 0.0000</pose>'),
}
for name, (before, after) in replacements.items():
    head, separator, tail = v15_text.partition(f'<name>{name}</name>')
    if not separator or before not in tail.split('</include>', 1)[0]:
        raise SystemExit(f'{name} pose in v14 world changed unexpectedly')
    v15_text = head + separator + tail.replace(before, after, 1)
v15.write_text(v15_text)
print(v15)

v16 = ROOT / 'spaced_objects_accessible_v16.world'
v16_text = v15_text
replacements_v16 = {
    'bookshelf_main': (
        '<pose>-7.0000 -9.0000 0.0000 0 0 0.0000</pose>',
        '<pose>-4.5000 -5.5000 0.0000 0 0 0.0000</pose>'),
    'ball_bearing_object': (
        '<pose>0.6000 -2.7000 0.3600 0.0000 0.0000 0.0000</pose>',
        '<pose>2.5000 -1.6000 0.3600 0.0000 0.0000 0.0000</pose>'),
}
for name, (before, after) in replacements_v16.items():
    head, separator, tail = v16_text.partition(f'<name>{name}</name>')
    if not separator or before not in tail.split('</include>', 1)[0]:
        raise SystemExit(f'{name} pose in v15 world changed unexpectedly')
    v16_text = head + separator + tail.replace(before, after, 1)
stand_name = '<model name="display_stand_ball_bearing_object">'
head, separator, tail = v16_text.partition(stand_name)
old_stand = '<pose>0.6000 -2.7000 0 0 0 0</pose>'
new_stand = '<pose>2.5000 -1.6000 0 0 0 0</pose>'
if not separator or old_stand not in tail.split('</model>', 1)[0]:
    raise SystemExit('Ball bearing display stand in v15 world changed unexpectedly')
v16_text = head + separator + tail.replace(old_stand, new_stand, 1)
v16.write_text(v16_text)
print(v16)
