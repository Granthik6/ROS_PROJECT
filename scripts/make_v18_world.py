#!/usr/bin/env python3
"""Place three small displays in separate open positions for a fixed trial.

The object positions are part of the Gazebo world only. They are not passed
to the DDQN policy or its sensor-map coverage assist.
"""

from pathlib import Path
import xml.etree.ElementTree as ET


worlds = Path('/root/ros2_ws/src/robot_tasks/worlds')
source = worlds / 'spaced_objects_accessible_v17.world'
target = worlds / 'spaced_objects_accessible_v18.world'
text = source.read_text()


def move_include(name, old_pose, new_pose):
    global text
    marker = f'<name>{name}</name>'
    before, separator, after = text.partition(marker)
    if not separator or old_pose not in after.split('</include>', 1)[0]:
        raise SystemExit(f'Unexpected include pose for {name}')
    text = before + separator + after.replace(old_pose, new_pose, 1)


def move_stand(name, old_pose, new_pose):
    global text
    marker = f'<model name="display_stand_{name}">'
    before, separator, after = text.partition(marker)
    if not separator or old_pose not in after.split('</model>', 1)[0]:
        raise SystemExit(f'Unexpected stand pose for {name}')
    text = before + separator + after.replace(old_pose, new_pose, 1)


moves = (
    ('cup_object', '-2.4000 -1.6000', '4.8000 1.3000',
     '0.2950 0.0000 0.0000 0.0000'),
    ('can_object', '-3.3000 -3.2000', '2.5000 -4.8000',
     '0.3575 0.0000 0.0000 0.0000'),
    ('ball_bearing_object', '0.6000 -2.7000', '1.0000 0.8000',
     '0.3600 0.0000 0.0000 0.0000'),
)
for name, old_xy, new_xy, suffix in moves:
    move_include(name,
                 f'<pose>{old_xy} {suffix}</pose>',
                 f'<pose>{new_xy} {suffix}</pose>')
    move_stand(name,
               f'<pose>{old_xy} 0 0 0 0</pose>',
               f'<pose>{new_xy} 0 0 0 0</pose>')

target.write_text(text)
ET.parse(target)
print(target)
