#!/usr/bin/env python3
"""Move the cafe table to a clear display position for an evaluation world.

The robot receives no object coordinates. This changes the fixed Gazebo
environment before a run, and the world file is recorded in its summary.
"""

from pathlib import Path
import xml.etree.ElementTree as ET


worlds = Path('/root/ros2_ws/src/robot_tasks/worlds')
source = worlds / 'spaced_objects_accessible_v13.world'
target = worlds / 'spaced_objects_accessible_v17.world'
text = source.read_text()
name = '<name>cafe_table_main</name>'
before = '<pose>-3.5000 -5.3000 0.0000 0 0 0.6000</pose>'
after = '<pose>0.5000 4.0000 0.0000 0 0 0.6000</pose>'
head, separator, tail = text.partition(name)
if not separator or before not in tail.split('</include>', 1)[0]:
    raise SystemExit('The source cafe table pose has changed')
target.write_text(head + separator + tail.replace(before, after, 1))
ET.parse(target)
print(target)
