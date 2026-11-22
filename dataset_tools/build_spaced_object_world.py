"""Place the same 36 models in a clear grid for detector and search trials."""

import copy
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path('/root/ros2_ws')
SOURCE = ROOT / 'src/robot_tasks/worlds/realistic_room.world'
OUTPUT = ROOT / 'src/robot_tasks/worlds/spaced_objects.world'
LARGE = {
    'bookshelf', 'cabinet', 'table', 'cafe_table', 'person_standing',
    'person_walking', 'construction_barrel', 'mailbox', 'postbox',
    'stop_sign', 'lamp_post', 'first_2015_trash_can', 'grey_tote',
    'jersey_barrier', 'stop_light', 'table_marble',
}


def build():
    source = ET.parse(SOURCE).getroot().find('world')
    if source is None:
        raise ValueError('Source world missing')
    entries = []
    for include in source.findall('include'):
        kind = include.findtext('uri', '').rsplit('/', 1)[-1]
        if kind not in ('sun', 'ground_plane'):
            entries.append((kind, include))
    if len(entries) != 36:
        raise ValueError(f'Expected 36 models, found {len(entries)}')

    columns = [-7.0, -4.2, -1.4, 1.4, 4.2, 7.0]
    rows = [-9.0, -6.2, -3.4, -0.6, 2.2, 5.0]
    perimeter = [(x, y) for j, y in enumerate(rows)
                 for i, x in enumerate(columns)
                 if i in (0, 5) or j in (0, 5)]
    interior = [(x, y) for j, y in enumerate(rows)
                for i, x in enumerate(columns)
                if i not in (0, 5) and j not in (0, 5)]
    hose = next(entry for entry in entries if entry[0] == 'fire_hose_long')
    large = [entry for entry in entries if entry[0] in LARGE]
    small = [entry for entry in entries
             if entry[0] not in LARGE and entry is not hose]
    if len(large) != 16 or len(small) != 19:
        raise ValueError('Unexpected model allocation')
    placements = []
    for entry, position in zip(large, perimeter[:16]):
        placements.append((entry, position))
    for entry, position in zip(small, interior + perimeter[16:19]):
        placements.append((entry, position))
    placements.append((hose, (-11.0, -11.0)))

    root = ET.Element('sdf', version='1.6')
    world = ET.SubElement(root, 'world', name='spaced_objects')
    for element in source:
        if element.tag == 'plugin' or element.tag == 'physics':
            world.append(copy.deepcopy(element))
        elif element.tag == 'include':
            kind = element.findtext('uri', '').rsplit('/', 1)[-1]
            if kind in ('sun', 'ground_plane'):
                world.append(copy.deepcopy(element))
    for (kind, original), (x, y) in placements:
        include = copy.deepcopy(original)
        pose = include.find('pose')
        values = [float(value) for value in pose.text.split()]
        yaw = values[5]
        if kind == 'fire_hose_long':
            yaw = 0.0
        pose.text = f'{x:.4f} {y:.4f} {values[2]:.4f} 0 0 {yaw:.4f}'
        world.append(include)
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(OUTPUT, encoding='unicode',
                               xml_declaration=True)
    print(f'Wrote {len(placements)} models to {OUTPUT}')


if __name__ == '__main__':
    build()
