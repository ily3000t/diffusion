"""Independent raw-to-window audit and bundled SUMO rule cross-check."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import psutil
from sumodiff.data.dataset import WindowDataset, write_json, collate_numpy
from sumodiff.data.episodes import load_registry
from sumodiff.experiments.recorder import RunRecorder, file_identity, load_config
from sumodiff.geometry.maps import RoadMap
from sumodiff.simulation.runtime import load_runtime


def center_from_raw(raw):
    # Independent trigonometric form: SUMO navigation angle from north clockwise.
    angle = np.deg2rad(raw['navigation_angle_deg'])
    return np.asarray(raw['front_position_m']) - raw['length_m'] / 2 * np.array([np.sin(angle), np.cos(angle)])


def audit(dataset_path, registry_path):
    episodes, splits = load_registry(registry_path)
    by_key = {e.key: e for e in episodes}
    maps = {e.key: RoadMap.read(e.network_path) for e in episodes}
    dataset_path = Path(dataset_path)
    manifest = json.loads((dataset_path / 'dataset_manifest.json').read_text(encoding='utf-8'))
    collection = json.loads((dataset_path / 'manifest.json').read_text(encoding='utf-8'))
    assert collection['formal'] and not collection['git']['dirty']
    assert collection['git']['commit_sha'] == manifest['run_sha']
    assert json.loads((dataset_path / 'status.json').read_text(encoding='utf-8'))['state'] == 'completed'
    assert file_identity(registry_path)['sha256'] == manifest['source_registry']['sha256']
    for identity in collection['data']['files']:
        assert file_identity(identity['location'])['sha256'] == identity['sha256']
    for name, identity in manifest['outputs'].items():
        assert file_identity(dataset_path / name)['sha256'] == identity['sha256']
    assert file_identity(dataset_path / 'resolved_config.yaml')['sha256'] == collection['resolved_config_sha256']
    assert file_identity(dataset_path / 'environment.json')['sha256'] == collection['environment_sha256']
    assert load_config(dataset_path / 'resolved_config.yaml')['window'] == manifest['window_config']
    actual_split_audit = json.loads((dataset_path / 'split_audit.json').read_text(encoding='utf-8'))
    assert actual_split_audit == splits
    calibration = json.loads((dataset_path / 'coverage_calibration.json').read_text(encoding='utf-8'))
    assert calibration['used_splits'] == ['train', 'validation']
    assert calibration['records'] == sum(sum(len(frame) for frame in e.frames[1:]) for e in episodes if e.split != 'test')
    dataset = WindowDataset(dataset_path)
    counts = {s: Counter(windows=0, core_windows=0, incomplete_future_windows=0, incomplete_history_windows=0,
                         selected_agents=0, excluded_current_vehicles=0, valid_future_points=0)
              for s in ('train', 'validation', 'test')}
    error, future_points, history_points, max_selected, zero_lanes = 0., 0, 0, 0, 0
    seen = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        c, t, meta, labels = [sample[k] for k in ('conditioning', 'targets', 'input_metadata', 'label_metadata')]
        entry = dataset.entries[index]
        assert entry['window_id'] not in seen
        seen.add(entry['window_id'])
        episode = by_key[meta['episode_key']]
        assert meta['split'] == entry['split'] == episode.split
        assert meta['geometry_id'] == entry['geometry_id'] == episode.manifest['geometry_id']
        tick, ids, agent_mask = meta['reference_tick'], meta['agent_ids'], c['agent_mask']
        assert tick >= 21 and tick % manifest['window_config']['reference_stride_ticks'] == 0
        assert meta['reference_time_s'] == tick * .1
        selected = [vid for vid in ids if vid is not None]
        assert len(selected) == len(set(selected)) and set(selected).issubset(episode.frames[tick])
        assert ids[0] == meta['reference_id'] == sorted(episode.frames[tick])[0]
        assert np.array_equal(agent_mask, np.array([vid is not None for vid in ids]))
        assert meta['excluded_vehicle_count'] == len(episode.frames[tick]) - len(selected)
        origin, yaw = np.asarray(meta['fixed_frame']['origin_world_m']), meta['fixed_frame']['yaw_world_rad']
        reference = episode.frames[tick][ids[0]]
        np.testing.assert_allclose(origin, center_from_raw(reference.raw), atol=1e-10)
        assert abs(yaw - np.deg2rad(90 - reference.raw['navigation_angle_deg'])) < 1e-12
        matrix = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        road, exact = maps[episode.key], sample['exact_map']
        assert exact['connections'] == road.connections
        assert len(exact['lanes']) == len(road.lanes)
        for lane_index, (saved_lane, raw_lane) in enumerate(zip(exact['lanes'], road.lanes)):
            points = (np.asarray(raw_lane['points_world_m']) - origin) @ matrix.T
            np.testing.assert_allclose(saved_lane['points_local_m'], points, atol=1e-10)
            np.testing.assert_allclose(c['lane_polylines'][lane_index, [0, -1], :2], points[[0, -1]], atol=4e-5)
            np.testing.assert_allclose(c['lane_polylines'][lane_index, :, 4:],
                np.tile([raw_lane['width_m'], raw_lane['speed_limit_mps'], raw_lane['internal'], raw_lane['priority']],
                        (c['lane_polylines'].shape[1], 1)), atol=1e-6)
        for saved_junction, raw_junction in zip(exact['junctions'], road.junctions):
            assert saved_junction['requests'] == raw_junction['requests']
            raw_points = np.asarray(raw_junction['points_world_m']).reshape(-1, 2)
            np.testing.assert_allclose(np.asarray(saved_junction['points_local_m']).reshape(-1, 2),
                                      (raw_points - origin) @ matrix.T, atol=1e-10)
        from shapely import affinity
        from shapely.geometry import shape
        inverse = [matrix[0, 0], matrix[1, 0], matrix[0, 1], matrix[1, 1], *origin]
        recovered_road = affinity.affine_transform(shape(exact['drivable']), inverse)
        assert recovered_road.symmetric_difference(road.drivable).area < 1e-6
        for slot, route in enumerate(meta['planned_routes']):
            if route is None:
                continue
            route_lanes = road.route_lane_ids(route)
            assert np.array_equal(c['route_lane_mask'][slot], [lane['id'] in route_lanes for lane in road.lanes])
            recovered_route = affinity.affine_transform(shape(exact['route_corridors'][slot]), inverse)
            assert recovered_route.symmetric_difference(road.route_area(route)).area < 1e-6
        agents = manifest['window_config']['max_agents']
        assert c['history'].shape == (agents, 21, 6) and t['future'].shape == (agents, 40, 6)
        assert c['map_raster'].shape == (3, 256, 256)
        assert c['map_raster'].dtype == np.uint8 and set(np.unique(c['map_raster'])).issubset({0, 1})
        for slot, vid in enumerate(ids):
            if vid is None:
                for key in ('history', 'history_mask', 'initial_positions', 'attributes', 'route_lane_mask'):
                    assert not c[key][slot].any()
                assert not t['future'][slot].any() and not t['future_mask'][slot].any()
                continue
            obs = episode.frames[tick][vid]
            anchor = center_from_raw(obs.raw)
            np.testing.assert_allclose(c['initial_positions'][slot], matrix @ (anchor - origin), atol=2e-5)
            assert meta['planned_routes'][slot] == obs.raw['route_edges']
            assert meta['route_indices_at_t0'][slot] == obs.raw['route_index']
            np.testing.assert_allclose(c['attributes'][slot], [obs.raw['length_m'], obs.raw['width_m'], 1, slot == 0])
            for name, times, states, masks in [('history', range(tick-20, tick+1), c['history'], c['history_mask']),
                                             ('future', range(tick+1, tick+41), t['future'], t['future_mask'])]:
                for step, raw_tick in enumerate(times):
                    present = 0 < raw_tick < len(episode.frames) and vid in episode.frames[raw_tick] and vid in episode.frames[raw_tick-1]
                    assert masks[slot, step] == present
                    if not present:
                        assert not states[slot, step].any()
                        continue
                    this_raw, previous_raw = episode.frames[raw_tick][vid].raw, episode.frames[raw_tick-1][vid].raw
                    world = center_from_raw(this_raw)
                    position = matrix @ (world - (anchor if name == 'future' else origin))
                    velocity = matrix @ (world - center_from_raw(previous_raw)) / .1
                    heading = np.deg2rad(90 - this_raw['navigation_angle_deg']) - yaw
                    expected = np.r_[position, velocity, np.sin(heading), np.cos(heading)]
                    error = max(error, float(np.max(np.abs(states[slot, step] - expected))))
                    np.testing.assert_allclose(states[slot, step], expected, atol=2e-5, rtol=2e-6)
                    if name == 'future':
                        future_points += 1
                    else:
                        history_points += 1
        assert labels['complete_history'] == bool(c['history_mask'][agent_mask].all())
        assert labels['complete_future'] == bool(t['future_mask'][agent_mask].all())
        assert labels['core_training_eligible'] == bool(labels['complete_history'] and labels['complete_future']) == entry['core_training_eligible']
        inference = dataset.inference(index)
        assert set(inference) == {'conditioning', 'input_metadata', 'exact_map'}
        assert set(inference['conditioning']) == set(c) and 'future_mask' not in c
        for key in c:
            np.testing.assert_array_equal(inference['conditioning'][key], c[key])
        counts[episode.split].update(windows=1, core_windows=int(labels['core_training_eligible']),
            incomplete_future_windows=int(not labels['complete_future']), incomplete_history_windows=int(not labels['complete_history']),
            selected_agents=len(selected), excluded_current_vehicles=meta['excluded_vehicle_count'], valid_future_points=int(t['future_mask'].sum()))
        max_selected = max(max_selected, len(selected))
    quality = json.loads((dataset_path / 'quality.json').read_text(encoding='utf-8'))
    for split, count in counts.items():
        for key, value in count.items():
            assert quality['by_split'][split][key] == value
    # SDK provides an independent parser/index implementation for priority rules.
    runtime, _ = load_runtime()
    import sumolib
    rule_links, foe_checks = 0, 0
    for episode in episodes:
        road = maps[episode.key]
        zero_lanes += sum(l['zero_geometry_length'] for l in road.lanes)
        sdk = sumolib.net.readNet(str(episode.network_path), withInternal=True, withFoes=True)
        for connection in road.connections:
            if 'request_index' not in connection:
                continue
            candidates = sdk.getLane(connection['source_lane']).getOutgoing()
            match = [c for c in candidates if c.getToLane().getID() == connection['target_lane'] and
                     c.getViaLaneID() == (connection['via_lane'] or '')]
            assert len(match) == 1
            assert match[0].getJunctionIndex() == connection['request_index']
            junction = next(j for j in road.junctions if j['sumo_raw']['id'] == connection['junction_id'])
            request = next(r for r in junction['requests'] if int(r['index']) == connection['request_index'])
            for other in range(len(junction['requests'])):
                assert match[0].getJunction().areFoes(connection['request_index'], other) == (other in request['foe_indices'])
                foe_checks += 1
            rule_links += 1
    batch = collate_numpy([dataset[0], dataset[len(dataset)-1]])
    assert batch['conditioning']['history'].shape == (2, manifest['window_config']['max_agents'], 21, 6)
    return dict(passed=True, windows=len(dataset), valid_history_points=history_points, valid_future_points=future_points,
        max_abs_state_error=error, max_selected_agents=max_selected, stored_local_maps_checked=len(dataset), priority_links_checked=rule_links,
        sdk_foe_relations_checked=foe_checks, zero_geometry_internal_lanes_retained=zero_lanes,
        family_coverage=splits['family_coverage'], by_split={k: dict(v) for k, v in counts.items()},
        source_rejections=len(splits['rejected_episodes']), preprocessing_run_sha=manifest['run_sha'],
        sdk_source=runtime['clients']['sumolib'], sampled_audit_rss_bytes=psutil.Process().memory_info().rss)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--formal', action='store_true')
    args = p.parse_args()
    with RunRecorder(args.output, Path.cwd(), dict(schema_version='sumodiff.window.audit.config.v1',
        dataset_manifest=file_identity(args.dataset / 'dataset_manifest.json'), sources=file_identity(args.sources)),
        [sys.executable, *sys.orig_argv[1:]], {'audit': 0}, purpose='stage2_independent_window_audit',
        data_files=[args.dataset / 'dataset_manifest.json', args.sources], formal=args.formal) as run:
        result = audit(args.dataset, args.sources)
        write_json(run.output / 'validation_summary.json', result)
        run.write_metrics(result)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
