import csv
import json

from robot_tasks.aggregate_world_coverage import build_aggregate, save_aggregate


def _run(directory, run_id, duration, object_name, x):
    directory.mkdir()
    (directory / 'summary.json').write_text(json.dumps({
        'run_id': run_id, 'status': 'completed', 'duration_sec': duration,
    }), encoding='utf-8')
    with (directory / 'observations.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            'elapsed_s', 'class', 'confidence', 'stamp_s',
            'world_x_m', 'world_y_m',
        ))
        writer.writeheader()
        for elapsed in (3.0, 4.0):
            writer.writerow({
                'elapsed_s': elapsed, 'class': object_name,
                'confidence': 0.8, 'stamp_s': elapsed,
                'world_x_m': x, 'world_y_m': 0.0,
            })


def test_combined_runs_record_first_source_and_do_not_claim_one_run(tmp_path):
    world = tmp_path / 'world.world'
    world.write_text('''<sdf version="1.6"><world name="default">
      <include><uri>model://cup</uri><name>cup_one</name>
        <pose>0 0 0 0 0 0</pose></include>
      <include><uri>model://bowl</uri><name>bowl_one</name>
        <pose>4 0 0 0 0 0</pose></include>
    </world></sdf>''', encoding='utf-8')
    first, second = tmp_path / 'first', tmp_path / 'second'
    _run(first, 'first', 100.0, 'cup', 0.1)
    _run(second, 'second', 90.0, 'bowl', 4.1)

    report = build_aggregate([first, second], world, radius_m=0.5)
    assert report['matched_world_objects'] == 2
    assert report['all_objects_detected_across_runs'] is True
    assert report['all_objects_detected_in_one_run'] is False
    assert report['time_to_all_across_runs_s'] == 104.0
    assert [row['source_run_id'] for row in report['objects']] == [
        'first', 'second',
    ]
    json_path, csv_path = save_aggregate(report, tmp_path / 'report')
    assert json.loads(json_path.read_text())['matched_world_objects'] == 2
    assert len(list(csv.DictReader(csv_path.open()))) == 2
