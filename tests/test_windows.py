from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from sumodiff.data.config import resolve_config
from sumodiff.data.episodes import Episode, Observation
from sumodiff.data.windows import build_window, candidate_ticks, audit_calibration_episodes
from sumodiff.geometry.maps import RoadMap


@pytest.fixture
def example(tmp_path):
    net = tmp_path / 'network.net.xml'
    net.write_text('''<net><edge id="road"><lane id="road_0" index="0" speed="20" width="4" length="1000" shape="-500,0 500,0"/></edge></net>''')
    frames = []
    for tick in range(101):
        current = {}
        for vid, offset in [('a', 0.), ('b', 10.), ('c', 200.)]:
            raw = dict(length_m=4., width_m=1.8, type_id='passenger', route_edges=['road'], route_index=0)
            current[vid] = Observation(np.array([tick * .5 + offset, 0.]), 0., raw)
        frames.append(current)
    episode = Episode('sha256:' + 'a' * 64, dict(dt=.1, episode_id='example', family='three_lane_straight', geometry_id='g1'), net, 'train', frames, net, [])
    return episode, RoadMap.read(net), resolve_config({})


def test_backward_velocity_exact_time_boundary_and_offsets(example):
    episode, road, config = example
    window, _ = build_window(episode, 30, road, config)
    c, t = window.conditioning, window.targets
    assert window.input_metadata['agent_ids'][:3] == ['a', 'b', None]
    assert window.input_metadata['excluded_by_radius'] == 1
    np.testing.assert_allclose(c['history'][0, :, 0], np.linspace(-10, 0, 21))
    np.testing.assert_allclose(c['history'][:2, :, 2], 5)
    np.testing.assert_allclose(t['future'][:2, :, 0], np.tile(np.arange(1, 41) * .5, (2, 1)))
    assert c['history_mask'][:2].all() and t['future_mask'][:2].all()
    assert not c['agent_mask'][2:].any() and not t['future_mask'][2:].any()
    assert np.count_nonzero(c['history'][2:]) == 0
    assert window.label_metadata['core_training_eligible']
    assert list(candidate_ticks(episode, config)) == list(range(30, 101, 10))
    with pytest.raises(ValueError, match='Reference tick'):
        build_window(episode, 20, road, config)


def test_future_mutation_does_not_change_any_input(example):
    episode, road, config = example
    original, _ = build_window(episode, 30, road, config)
    changed = deepcopy(episode)
    for tick in range(31, 101):
        changed.frames[tick] = {'future_newborn': changed.frames[tick]['c']}
    altered, _ = build_window(changed, 30, road, config)
    assert original.input_metadata == altered.input_metadata
    assert original.exact_map == altered.exact_map
    for key in original.conditioning:
        np.testing.assert_array_equal(original.conditioning[key], altered.conditioning[key])
    assert not altered.targets['future_mask'].any()
    assert not altered.label_metadata['complete_future']


def test_extra_history_sample_and_gap_masks(example):
    episode, road, config = example
    del episode.frames[9]['b']
    del episode.frames[49]['b']
    window, _ = build_window(episode, 30, road, config)
    assert not window.conditioning['history_mask'][1, 0]  # first history tick10 needs tick9
    assert window.conditioning['history_mask'][1, 1:].all()
    assert not window.targets['future_mask'][1, 18:20].any()  # tick49 absent, tick50 cannot differentiate
    assert window.targets['future_mask'][1, 20:].all()
    assert window.label_metadata['history_missing_reasons'] == {'previous_velocity_sample_missing': 1}
    assert not window.label_metadata['core_training_eligible']


def test_exit_padding_and_future_birth_not_selected(example):
    episode, road, config = example
    window, _ = build_window(episode, 90, road, config)
    assert window.targets['future_mask'][:2, :10].all()
    assert not window.targets['future_mask'][:, 10:].any()
    assert window.label_metadata['future_missing_reasons']['outside_episode'] == 60
    assert not window.label_metadata['complete_future']
    episode.frames[90] = {}
    assert build_window(episode, 90, road, config) == (None, 'no_current_reference_vehicle')


def test_capacity_and_reference_use_current_only(example):
    episode, road, config = example
    config['max_agents'] = 1
    window, _ = build_window(episode, 30, road, config)
    assert window.input_metadata['agent_ids'] == ['a']
    assert window.input_metadata['excluded_by_capacity'] == 1
    assert window.conditioning['attributes'][0, 3] == 1


def test_test_split_never_used_for_coverage_calibration(example):
    episode, road, config = example
    poison = deepcopy(episode)
    poison.split = 'test'
    poison.frames[2]['a'].center[:] = 1e9
    report = audit_calibration_episodes([episode, poison], config)
    assert report['max_interval_speed_mps'] == 5.
    assert report['used_splits'] == ['train', 'validation']
    config['map_extent_m'] = [-10., -10., 10., 10.]
    with pytest.raises(ValueError, match='Map extent'):
        resolve_config(config)
