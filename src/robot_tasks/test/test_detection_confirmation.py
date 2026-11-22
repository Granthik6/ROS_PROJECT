"""A one-frame YOLO candidate should be revisited without target poses."""

from robot_tasks.evaluate_ddqn_discovery import DetectionConfirmation


def test_new_unconfirmed_candidate_triggers_small_view_oscillation():
    confirmation = DetectionConfirmation(enabled=True)
    candidate = {'id': '7', 'confirmed_s': None, 'first_seen_s': 10.0}
    actions = [confirmation.action([candidate], 10.1 + i * 0.1, 0)
               for i in range(12)]
    assert actions == [(2, True), (1, True)] * 6
    assert confirmation.events == 1
    assert confirmation.actions == 12
    assert confirmation.action([candidate], 11.4, 0) == (0, False)


def test_confirmed_or_stale_candidate_does_not_interrupt_policy():
    confirmation = DetectionConfirmation(enabled=True)
    assert confirmation.action([
        {'id': '1', 'confirmed_s': 0.2, 'first_seen_s': 0.1},
        {'id': '2', 'confirmed_s': None, 'first_seen_s': 0.1},
    ], 2.0, 0) == (0, False)
    assert confirmation.events == 0
