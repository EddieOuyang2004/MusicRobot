"""Check v2 selected-exit switching and legacy end-only transitions."""
import argparse
import csv
from pathlib import Path


def verify_trace(rows, entry_max_phase=0.2):
    terminal = None
    bridge = None
    completed = 0
    fallbacks = 0
    for row in rows:
        event = row['event']
        assert event != 'bridge_hold', 'Playback entered terminal hold'
        assert float(row['audio_history_seconds']) <= 30.0001, 'History exceeds 30 seconds'
        if event == 'terminal_frame':
            assert bridge is None, 'Current motion ended during a bridge'
            assert float(row['current_phase']) >= 0.999999, 'Terminal frame was not reached'
            terminal = row
        elif event == 'switch_start':
            authored = row.get('transition_backend', 'quintic') != 'quintic'
            if not authored:
                assert terminal is not None, 'Switch started before terminal frame'
            assert bridge is None, 'Overlapping bridges'
            if authored and row.get('exit_phase'):
                assert abs(float(row['current_phase'])-float(row['exit_phase'])) < 1e-6, 'Wrong source exit'
                assert float(row['trimmed_seconds']) >= -1e-8, 'Exit after clip end'
            else:
                assert float(row['current_phase']) >= 0.999999, 'Early switch'
            if authored:
                assert 0 <= float(row['transition_blend']) < 1., 'Invalid bridge progress'
                for field in ('position', 'velocity', 'acceleration'):
                    assert float(row[f'bridge_{field}_residual']) < 1e-6, 'Planned derivative mismatch'
            else:
                assert float(row['transition_blend']) == 0.0, 'Bridge did not begin at source pose'
            assert float(row['transition_duration_seconds']) > 0, 'Instantaneous bridge'
            if row['transition_reason'].startswith('replay_'):
                fallbacks += 1
                assert row['transition_motion_id'] == row['current_motion_id'], 'Fallback changed clips'
                assert int(row['entry_frame_index']) == 0, 'Replay did not return to frame zero'
            else:
                assert 0 <= float(row['entry_phase']) <= entry_max_phase + 1e-6, 'Entry is outside the allowed motion prefix'
            bridge = row
        if bridge is not None and row['transition_phase']:
            assert abs(float(row['transition_phase']) - float(bridge['transition_phase'])) < 1e-8, 'Incoming clip advanced during bridge'
        if event == 'switch_complete':
            assert bridge is not None, 'Completion without a bridge'
            assert float(row['transition_blend']) == 1.0, 'Bridge did not reach its target'
            if row.get('transition_backend', 'quintic') == 'quintic':
                assert abs(float(row['current_phase']) - float(bridge['transition_phase'])) < 1e-8, 'Playback resumed at wrong frame'
            else:
                assert float(row['current_phase']) >= float(bridge['transition_phase']), 'Playback lost entry carryover'
                assert int(row['bridge_generation']) == int(bridge['bridge_generation'])+1, 'Invalid clip generation'
            completed += 1
            bridge = terminal = None
    assert completed > 0, 'No completed transitions exercised'
    return {'completed_transitions': completed, 'replay_fallbacks': fallbacks}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--entry-max-phase', type=float, default=0.2)
    args = parser.parse_args()
    with args.trace.open(newline='', encoding='utf-8') as handle:
        print(verify_trace(list(csv.DictReader(handle)), args.entry_max_phase))
