#!/usr/bin/env python3
"""Make the ball bearing visible at the TurtleBot camera resolution.

The source bearing is a 3 mm radius Gazebo sphere. At several metres it is
subpixel sized. This fixed evaluation world uses a 30 mm radius bearing on
the same stand; all object positions remain those of v17.
"""

from pathlib import Path
import xml.etree.ElementTree as ET


worlds = Path('/root/ros2_ws/src/robot_tasks/worlds')
source = worlds / 'spaced_objects_accessible_v17.world'
target = worlds / 'spaced_objects_accessible_v19.world'
text = source.read_text()
marker = '<name>ball_bearing_object</name>'
before, separator, after = text.partition(marker)
old = '<scale>1.8 1.8 1.8</scale>'
new = '<scale>10 10 10</scale>'
if not separator or old not in after.split('</include>', 1)[0]:
    raise SystemExit('Unexpected ball-bearing scale in source world')
target.write_text(before + separator + after.replace(old, new, 1))
ET.parse(target)
print(target)
