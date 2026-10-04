"""Verified raw episode loading and explicit geometry-isolated splits."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from sumodiff.experiments.recorder import file_identity
from sumodiff.geometry.coordinates import sumo_center
from sumodiff.simulation.scenes import semantic_geometry_id


@dataclass
class Observation:
    center: np.ndarray
    yaw: float
    raw: dict


@dataclass
class Episode:
    key: str
    manifest: dict
    manifest_path: Path
    split: str
    frames: list
    network_path: Path
    input_files: list

    @property
    def dt(self):
        return self.manifest['dt']


def _verified_file(identity, directory):
    # Resolve within the episode, allowing relocation without trusting stale paths.
    path = directory / Path(identity['location']).name
    actual = file_identity(path)
    if actual['sha256'] != identity['sha256'] or actual['size_bytes'] != identity['size_bytes']:
        raise ValueError(f'Input integrity failure: {path}')
    return path


def load_episode(path, split):
    path = Path(path).resolve(strict=True)
    manifest = json.loads(path.read_text(encoding='utf-8-sig'))
    if manifest.get('schema_version') != 'sumodiff.raw.episode.v1':
        raise ValueError('Unsupported raw episode schema')
    if manifest.get('state') != 'completed' or not manifest.get('eligible_for_normal_training') or manifest.get('diagnostic_only'):
        raise ValueError('Episode is not eligible for normal training')
    if abs(manifest['dt'] - .1) > 1e-12:
        raise ValueError('Only dt=0.1 is supported')
    files = [path]
    outputs = {}
    for identity in manifest['outputs']:
        verified = _verified_file(identity, path.parent)
        outputs[verified.name] = verified
        files.append(verified)
    if any(name not in outputs for name in ('frames.jsonl', 'events.jsonl', 'quality.json')):
        raise ValueError('Missing raw collection outputs')
    quality = json.loads(outputs['quality.json'].read_text(encoding='utf-8'))
    if not quality.get('eligible_for_normal_training') or not quality.get('normal_traffic_quality_pass'):
        raise ValueError('Manifest and quality disagree')
    network = None
    for identity in manifest['scene']['files']:
        verified = _verified_file(identity, path.parent / 'scene')
        files.append(verified)
        if verified.name == 'network.net.xml':
            network = verified
    if network is None or semantic_geometry_id(network) != manifest['geometry_id']:
        raise ValueError('Semantic geometry identity mismatch')
    content = dict(geometry_id=manifest['geometry_id'], frames=file_identity(outputs['frames.jsonl'])['sha256'],
                   events=file_identity(outputs['events.jsonl'])['sha256'])
    data_id = 'sha256:' + hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    if data_id != manifest['data_id']:
        raise ValueError('Raw data identity mismatch')
    frames = []
    with outputs['frames.jsonl'].open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if row['schema_version'] != 'sumodiff.raw.frame.v1' or row['tick'] != index or abs(row['time_seconds'] - index * .1) > 1e-7:
                raise ValueError('Frame index/time grid mismatch')
            observations = {}
            for vehicle in row['vehicles']:
                vehicle_id, raw = vehicle['vehicle_id'], vehicle['sumo_raw']
                if vehicle_id in observations:
                    raise ValueError('Duplicate vehicle ID within one frame')
                if raw['vehicle_class'] != 'passenger':
                    raise NotImplementedError('Only passenger vehicles are supported')
                if raw['length_m'] <= 0 or raw['width_m'] <= 0:
                    raise ValueError('Invalid vehicle dimensions')
                center, yaw = sumo_center(raw['front_position_m'], raw['navigation_angle_deg'], raw['length_m'])
                if center.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(yaw):
                    raise ValueError('Non-finite vehicle pose')
                if not raw['route_edges'] or not 0 <= raw['route_index'] < len(raw['route_edges']):
                    raise ValueError('Invalid currently known route')
                observations[vehicle_id] = Observation(center, float(yaw), raw)
            frames.append(observations)
    if not frames:
        raise ValueError('Empty raw episode')
    return Episode(data_id, manifest, path, split, frames, network, files)


def load_registry(path):
    """All split assignments are validated before any window is constructed.

    Explicit assignments avoid a geometry group changing split when later data
    is appended. Same geometry's seeds/flows/episodes must use the same split.
    Episode content IDs, not output directory basenames, are unique keys.
    """
    path = Path(path).resolve(strict=True)
    document = json.loads(path.read_text(encoding='utf-8-sig'))
    if set(document) != {'schema_version', 'episodes'} or document['schema_version'] != 'sumodiff.sources.v1':
        raise ValueError('Invalid source registry schema')
    episodes, rejected, identities, geometry_splits, episode_splits = [], [], [], {}, {}
    for source in document['episodes']:
        if set(source) != {'manifest', 'split'} or source['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Every source requires an explicit valid split')
        manifest_path = Path(source['manifest'])
        if not manifest_path.is_absolute():
            manifest_path = path.parent / manifest_path
        identity = file_identity(manifest_path)
        identities.append(identity)
        raw = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        if raw.get('schema_version') != 'sumodiff.raw.episode.v1':
            raise ValueError('Unsupported source episode schema')
        reasons = []
        if raw.get('state') != 'completed':
            reasons.append('collection_not_completed')
        if raw.get('diagnostic_only'):
            reasons.append('diagnostic_episode')
        if not raw.get('eligible_for_normal_training'):
            reasons.append('normal_quality_ineligible')
        if reasons:
            rejected.append(dict(manifest=identity, assigned_split=source['split'], reasons=reasons))
            continue
        episode = load_episode(manifest_path, source['split'])
        geometry = episode.manifest['geometry_id']
        if geometry in geometry_splits and geometry_splits[geometry] != episode.split:
            raise ValueError('Geometry leakage across splits')
        if episode.key in episode_splits:
            raise ValueError('Duplicate episode content (possibly relocated or renamed)')
        geometry_splits[geometry] = episode.split
        episode_splits[episode.key] = episode.split
        episodes.append(episode)
    if not episodes:
        raise ValueError('Registry has no eligible episodes')
    coverage = {split: sorted({e.manifest['family'] for e in episodes if e.split == split})
                for split in ('train', 'validation', 'test')}
    return episodes, dict(schema_version='sumodiff.split.audit.v1', registry=file_identity(path),
        source_manifests=identities, rejected_episodes=rejected, family_coverage=coverage,
        geometry_assignments=geometry_splits, episode_assignments=episode_splits,
        episodes=[dict(episode_key=e.key, source_episode_id=e.manifest['episode_id'], split=e.split,
                       family=e.manifest['family'], geometry_id=e.manifest['geometry_id'], manifest=str(e.manifest_path)) for e in episodes])
