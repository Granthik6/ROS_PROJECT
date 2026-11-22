"""Add LiDAR-visible display stands in the spaced search world."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path('/root/ros2_ws')
SOURCE = ROOT / 'src/robot_tasks/worlds/spaced_objects.world'
OUTPUT = ROOT / 'src/robot_tasks/worlds/spaced_objects_pedestals.world'
HEIGHT = 0.36
STANDS = {
    'cricket_ball_object': (0.30, 0.30),
    'monkey_wrench_object': (0.55, 0.45),
    'hokuyo_object': (0.30, 0.30),
    'saucepan_object': (0.75, 0.75),
    'ball_bearing_object': (0.30, 0.30),
    'gear_part_object': (0.30, 0.30),
}
EXTRA_STANDS = {
    'cup_object': (0.30, 0.30),
    'bowl_object': (0.40, 0.40),
    'can_object': (0.30, 0.30),
    'drill_object': (0.50, 0.45),
    'hammer_object': (0.45, 0.50),
    'beer_object': (0.30, 0.30),
}
# Model origins differ from visible mesh bottoms. These poses put the mesh
# bottoms on the 0.36 m stand tops rather than leaving props floating.
EXTRA_MODEL_Z = {
    'cup_object': 0.2950,
    'bowl_object': 0.3599,
    'can_object': 0.3575,
    'drill_object': 0.3580,
    'hammer_object': 0.3600,
    'beer_object': 0.2450,
}
WALLS = (
    ('south', 0.0, -11.0, 18.0, 0.12),
    ('north', 0.0, 7.0, 18.0, 0.12),
    ('west', -9.0, -2.0, 0.12, 18.0),
    ('east', 9.0, -2.0, 0.12, 18.0),
)
COMPLETION_POSITIONS = {
    'can_object': (0.0, -4.8),
    'hammer_object': (-2.8, -1.8),
    'gear_part_object': (0.0, 3.5),
}


def stand(name, x, y, width, depth):
    model = ET.Element('model', name=f'display_stand_{name}')
    ET.SubElement(model, 'static').text = 'true'
    ET.SubElement(model, 'pose').text = f'{x:.4f} {y:.4f} 0 0 0 0'
    link = ET.SubElement(model, 'link', name='body')
    ET.SubElement(link, 'pose').text = f'0 0 {HEIGHT / 2:.4f} 0 0 0'
    for kind in ('collision', 'visual'):
        element = ET.SubElement(link, kind, name=kind)
        geometry = ET.SubElement(element, 'geometry')
        ET.SubElement(ET.SubElement(geometry, 'box'), 'size').text = (
            f'{width:.4f} {depth:.4f} {HEIGHT:.4f}'
        )
        if kind == 'visual':
            material = ET.SubElement(element, 'material')
            color = ('0.85 0.90 0.95 1' if name in EXTRA_STANDS
                     else '0.55 0.55 0.55 1')
            ET.SubElement(material, 'ambient').text = color
            ET.SubElement(material, 'diffuse').text = color
    return model


def boundary_wall(name, x, y, width, depth):
    model = ET.Element('model', name=f'search_boundary_{name}')
    ET.SubElement(model, 'static').text = 'true'
    ET.SubElement(model, 'pose').text = f'{x:.4f} {y:.4f} 0 0 0 0'
    link = ET.SubElement(model, 'link', name='body')
    ET.SubElement(link, 'pose').text = '0 0 0.6 0 0 0'
    for kind in ('collision', 'visual'):
        element = ET.SubElement(link, kind, name=kind)
        geometry = ET.SubElement(element, 'geometry')
        ET.SubElement(ET.SubElement(geometry, 'box'), 'size').text = (
            f'{width:.4f} {depth:.4f} 1.2'
        )
        if kind == 'visual':
            material = ET.SubElement(element, 'material')
            ET.SubElement(material, 'ambient').text = '0.55 0.55 0.55 1'
            ET.SubElement(material, 'diffuse').text = '0.60 0.60 0.60 1'
    return model


def build(extended=False, completion=False):
    extended = extended or completion
    stands = {**STANDS, **EXTRA_STANDS} if extended else STANDS
    output = (ROOT / 'src/robot_tasks/worlds/spaced_objects_completion.world'
              if completion else
              ROOT / 'src/robot_tasks/worlds/spaced_objects_safe_stands.world'
              if extended else OUTPUT)
    root = ET.parse(SOURCE).getroot()
    world = root.find('world')
    world.set('name', output.stem)
    placed = set()
    for include in world.findall('include'):
        name = include.findtext('name', '')
        if name == 'fire_hose_long_object':
            # The source world puts this target outside the search boundary.
            # Use the free northeast slot so autonomous search can reach it.
            include.find('pose').text = '7.0000 5.0000 0.1000 0 0 0.0000'
        if name not in stands:
            continue
        pose = include.find('pose')
        values = [float(value) for value in pose.text.split()]
        if completion and name in COMPLETION_POSITIONS:
            values[:2] = COMPLETION_POSITIONS[name]
        x, y = values[:2]
        values[2] = (EXTRA_MODEL_Z[name] if name in EXTRA_MODEL_Z
                     else values[2] + HEIGHT)
        pose.text = ' '.join(f'{value:.4f}' for value in values)
        width, depth = stands[name]
        world.append(stand(name, x, y, width, depth))
        placed.add(name)
    if placed != set(stands):
        raise ValueError(f'Missing models: {set(stands) - placed}')
    for name, x, y, width, depth in WALLS:
        world.append(boundary_wall(name, x, y, width, depth))
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(output, encoding='unicode',
                               xml_declaration=True)
    print(f'Wrote {len(placed)} stands and 36 target objects to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--completion', action='store_true')
    args = parser.parse_args()
    build(args.extended, args.completion)
