import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from sumodiff.data.episodes import load_registry
from sumodiff.data.dataset import WindowDataset, save_window, write_json, collate_numpy
from sumodiff.experiments.recorder import file_identity
from sumodiff.simulation.scenes import semantic_geometry_id
from sumodiff.data.windows import Window


def make_episode(parent, name, *, offset=0, geometry_shift=0, eligible=True, diagnostic=False, state='completed'):
    directory = parent / name
    (directory / 'scene').mkdir(parents=True)
    net = directory / 'scene' / 'network.net.xml'
    net.write_text(f'<net><edge id="road"><lane id="road_0" index="0" length="100" width="4" speed="10" shape="{geometry_shift},0 {100 + geometry_shift},0"/></edge></net>', encoding='utf-8')
    frames = directory / 'frames.jsonl'
    raw = dict(front_position_m=[5. + offset, 0.], navigation_angle_deg=90., length_m=4., width_m=1.8,
               vehicle_class='passenger', route_edges=['road'], route_index=0)
    rows = [dict(schema_version='sumodiff.raw.frame.v1', tick=0, time_seconds=0., vehicles=[]),
            dict(schema_version='sumodiff.raw.frame.v1', tick=1, time_seconds=.1,
                 vehicles=[dict(vehicle_id='a', sumo_raw=raw)])]
    frames.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')
    events = directory / 'events.jsonl'
    events.write_text('{}\n', encoding='utf-8')
    quality = directory / 'quality.json'
    write_json(quality, dict(eligible_for_normal_training=eligible, normal_traffic_quality_pass=eligible))
    geometry = semantic_geometry_id(net)
    content = dict(geometry_id=geometry, frames=file_identity(frames)['sha256'], events=file_identity(events)['sha256'])
    manifest = dict(schema_version='sumodiff.raw.episode.v1', episode_id='same_basename', family='three_lane_straight',
        state=state, eligible_for_normal_training=eligible, diagnostic_only=diagnostic, dt=.1, geometry_id=geometry,
        scene=dict(files=[file_identity(net)]), outputs=[file_identity(p) for p in (frames, events, quality)],
        data_id='sha256:' + hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest())
    path = directory / 'episode_manifest.json'
    write_json(path, manifest)
    return path


def registry(parent, entries):
    path = parent / 'sources.json'
    write_json(path, dict(schema_version='sumodiff.sources.v1', episodes=[dict(manifest=str(p), split=s) for p, s in entries]))
    return path


def test_geometry_split_isolation_duplicate_content_and_namespacing(tmp_path):
    a = make_episode(tmp_path, 'a')
    b = make_episode(tmp_path, 'b', offset=1)
    path = registry(tmp_path, [(a, 'train'), (b, 'validation')])
    with pytest.raises(ValueError, match='Geometry leakage'):
        load_registry(path)
    path = registry(tmp_path, [(a, 'train'), (b, 'train')])
    episodes, audit = load_registry(path)
    assert len(episodes) == 2 and episodes[0].key != episodes[1].key
    assert episodes[0].manifest['episode_id'] == episodes[1].manifest['episode_id']
    assert len(audit['geometry_assignments']) == 1
    clone = make_episode(tmp_path, 'clone')
    with pytest.raises(ValueError, match='Duplicate episode content'):
        load_registry(registry(tmp_path, [(a, 'train'), (clone, 'train')]))


def test_distinct_maps_and_exclusions_are_audited(tmp_path):
    a = make_episode(tmp_path, 'a')
    b = make_episode(tmp_path, 'b', geometry_shift=10)
    d = make_episode(tmp_path, 'diagnostic', diagnostic=True)
    failed = make_episode(tmp_path, 'failed', eligible=False, state='failed')
    episodes, audit = load_registry(registry(tmp_path, [(a, 'train'), (b, 'test'), (d, 'validation'), (failed, 'train')]))
    assert len(episodes) == 2
    assert len(audit['rejected_episodes']) == 2
    assert audit['family_coverage']['validation'] == []
    assert audit['rejected_episodes'][0]['reasons'] == ['diagnostic_episode']
    assert audit['rejected_episodes'][1]['reasons'] == ['collection_not_completed', 'normal_quality_ineligible']


def test_modified_raw_input_fails_hash_check(tmp_path):
    a = make_episode(tmp_path, 'a')
    (a.parent / 'frames.jsonl').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='Input integrity'):
        load_registry(registry(tmp_path, [(a, 'train')]))


def test_reader_inference_has_no_label_dependency_or_core_filter(tmp_path):
    w = Window(dict(history=np.zeros((1, 21, 6), np.float32), agent_mask=np.ones(1, bool),
        history_mask=np.ones((1, 21), bool), lane_mask=np.ones(1, bool)),
        dict(future=np.zeros((1, 40, 6), np.float32), future_mask=np.ones((1, 40), bool)),
        dict(window_id='tiny', split='train', episode_key='e', geometry_id='g'),
        dict(core_training_eligible=True), dict(lanes=[]))
    entry = save_window(tmp_path / 'windows' / 'tiny', w)
    write_json(tmp_path / 'windows.json', [entry])
    write_json(tmp_path / 'dataset_manifest.json', dict(schema_version='sumodiff.dataset.v1',
        window_index=file_identity(tmp_path / 'windows.json')))
    dataset = WindowDataset(tmp_path)
    assert dataset[0]['targets']['future_mask'].all()
    (tmp_path / 'windows' / 'tiny' / 'targets.npz').unlink()
    (tmp_path / 'windows' / 'tiny' / 'labels.json').unlink()
    inference = dataset.inference(0)
    assert set(inference) == {'conditioning', 'input_metadata', 'exact_map'}
    assert 'future_mask' not in inference['conditioning']
    with pytest.raises(ValueError, match='future-label'):
        WindowDataset(tmp_path, core_only=True).inference(0)
    with pytest.raises(FileNotFoundError):
        dataset[0]
    (tmp_path / 'windows' / 'tiny' / 'input.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='Window integrity'):
        dataset.inference(0)


def test_variable_lane_batch_masks():
    def sample(lanes):
        return dict(conditioning=dict(history=np.ones((12, 21, 6), np.float32),
            lane_polylines=np.ones((lanes, 64, 8), np.float32), lane_mask=np.ones(lanes, bool),
            lane_point_mask=np.ones((lanes, 64), bool), lane_adjacency=np.eye(lanes, dtype=bool),
            route_lane_mask=np.ones((12, lanes), bool)), input_metadata={}, exact_map={})
    batch = collate_numpy([sample(2), sample(4)])
    assert batch['conditioning']['history'].shape == (2, 12, 21, 6)
    assert batch['conditioning']['lane_polylines'].shape == (2, 4, 64, 8)
    assert not batch['conditioning']['lane_mask'][0, 2:].any()
    assert not batch['conditioning']['route_lane_mask'][0, :, 2:].any()
    assert not batch['conditioning']['lane_adjacency'][0, 2:].any()
