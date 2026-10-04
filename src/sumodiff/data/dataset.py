"""Separate conditioning and labels on disk, with an auditable window index."""
from pathlib import Path
import json
import numpy as np
from sumodiff.experiments.recorder import file_identity


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_window(directory, window):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / 'conditioning.npz', **window.conditioning)
    np.savez_compressed(directory / 'targets.npz', **window.targets)
    write_json(directory / 'input.json', window.input_metadata)
    write_json(directory / 'labels.json', window.label_metadata)
    write_json(directory / 'map.json', window.exact_map)
    return dict(window_id=window.input_metadata['window_id'], split=window.input_metadata['split'],
                episode_key=window.input_metadata['episode_key'], geometry_id=window.input_metadata['geometry_id'],
                core_training_eligible=window.label_metadata['core_training_eligible'],
                files={name: {**file_identity(directory / name), 'relative_path': str((directory / name).relative_to(directory.parent.parent))}
                       for name in ('conditioning.npz', 'targets.npz', 'input.json', 'labels.json', 'map.json')})


class WindowDataset:
    """NumPy reader; no torch or old-project dependency.

    Core-only selection is for training, never an inference task filter.
    Inference does not open target/label files or expose future mask/exit times.
    """
    def __init__(self, directory, split=None, *, core_only=False, verify=True):
        self.directory = Path(directory).resolve(strict=True)
        self.core_only, self.verify = core_only, verify
        manifest = json.loads((self.directory / 'dataset_manifest.json').read_text(encoding='utf-8'))
        if manifest['schema_version'] != 'sumodiff.dataset.v1':
            raise ValueError('Unsupported dataset schema')
        index = self.directory / 'windows.json'
        if verify and file_identity(index)['sha256'] != manifest['window_index']['sha256']:
            raise ValueError('Window index integrity failure')
        if split is not None and split not in ('train', 'validation', 'test'):
            raise ValueError('Invalid split')
        self.entries = [entry for entry in json.loads(index.read_text(encoding='utf-8'))
                        if (split is None or entry['split'] == split) and (not core_only or entry['core_training_eligible'])]

    def __len__(self):
        return len(self.entries)

    def _path(self, entry, filename):
        identity = entry['files'][filename]
        path = (self.directory / identity['relative_path']).resolve(strict=True)
        if not path.is_relative_to(self.directory):
            raise ValueError('Window path escapes dataset root')
        if self.verify:
            actual = file_identity(path)
            if actual['sha256'] != identity['sha256'] or actual['size_bytes'] != identity['size_bytes']:
                raise ValueError(f'Window integrity failure: {path}')
        return path

    def _json(self, entry, filename):
        return json.loads(self._path(entry, filename).read_text(encoding='utf-8'))

    def _arrays(self, entry, filename):
        with np.load(self._path(entry, filename), allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def inference(self, index):
        if self.core_only:
            raise ValueError('Inference tasks cannot be selected using future-label completeness')
        entry = self.entries[index]
        return dict(conditioning=self._arrays(entry, 'conditioning.npz'),
                    input_metadata=self._json(entry, 'input.json'), exact_map=self._json(entry, 'map.json'))

    def __getitem__(self, index):
        entry = self.entries[index]
        return dict(conditioning=self._arrays(entry, 'conditioning.npz'), targets=self._arrays(entry, 'targets.npz'),
                    input_metadata=self._json(entry, 'input.json'), label_metadata=self._json(entry, 'labels.json'),
                    exact_map=self._json(entry, 'map.json'))


def collate_numpy(samples):
    """Batch variable lane counts without binding vehicle slots to map channels."""
    if not samples:
        raise ValueError('Empty batch')
    batch, lanes = len(samples), max(s['conditioning']['lane_mask'].shape[0] for s in samples)
    lane_axes = {'lane_polylines': (0,), 'lane_mask': (0,), 'lane_point_mask': (0,),
                 'lane_adjacency': (0, 1), 'route_lane_mask': (1,)}
    output = {}
    keys = samples[0]['conditioning'].keys()
    if any(s['conditioning'].keys() != keys for s in samples):
        raise ValueError('Inconsistent conditioning keys')
    for key, array in samples[0]['conditioning'].items():
        shape = list(array.shape)
        for axis in lane_axes.get(key, ()):
            shape[axis] = lanes
        target = np.zeros((batch, *shape), dtype=array.dtype)
        for i, sample in enumerate(samples):
            value = sample['conditioning'][key]
            target[(i, *[slice(0, dim) for dim in value.shape])] = value
        output[key] = target
    result = dict(conditioning=output, input_metadata=[s['input_metadata'] for s in samples],
                  exact_map=[s['exact_map'] for s in samples])
    has_targets = ['targets' in s for s in samples]
    if any(has_targets) != all(has_targets):
        raise ValueError('Cannot mix inference and labeled samples')
    if all(has_targets):
        result['targets'] = {key: np.stack([s['targets'][key] for s in samples]) for key in samples[0]['targets']}
    return result
