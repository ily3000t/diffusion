"""Explicit recovery of byte-identical historical Git inputs into a new run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from sumodiff.data.dataset import write_json
from sumodiff.experiments.recorder import RunRecorder, file_identity


def matching_git_bytes(blob, expected):
    """Accept exact bytes only; LF/CRLF candidates never relax the source hash."""
    candidates=(blob,blob.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))
    for raw in candidates:
        if len(raw)==expected['size_bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256']:
            return raw
    raise ValueError('Historical Git bytes cannot match the recorded source hash')


def load_snapshot_map(source, snapshot_file):
    if snapshot_file is None:
        return {}
    snapshot_file=Path(snapshot_file).resolve(strict=True)
    data=json.loads(snapshot_file.read_text(encoding='utf-8'))
    identity=file_identity(Path(source)/'manifest.json')
    if data.get('schema_version')!='sumodiff.historical_git_inputs.v1' or data['source_manifest']['sha256']!=identity['sha256']:
        raise ValueError('Historical input bundle belongs to a different source run')
    original=json.loads((Path(source)/'manifest.json').read_text(encoding='utf-8'))['data']['files']
    declared={str(Path(row['location']).resolve()):row for row in original}
    aliases={}
    for entry in data['snapshots']:
        name=str(Path(entry['original']['location']).resolve())
        if name not in declared or name in aliases:
            raise ValueError('Unknown or duplicate historical source input')
        expected=declared[name]
        found=file_identity(entry['snapshot']['location'])
        if any(entry['original'][k]!=expected[k] or found[k]!=expected[k] for k in ('sha256','size_bytes')):
            raise ValueError('Historical snapshot does not match the recorded exact bytes')
        aliases[name]=Path(found['location'])
    return aliases


def run(source, output, formal=False):
    source=Path(source).resolve(strict=True)
    source_manifest=source/'manifest.json'
    manifest=json.loads(source_manifest.read_text(encoding='utf-8'))
    root=Path(manifest['git']['root']).resolve(strict=True)
    commit=manifest['git']['commit_sha']
    modified=[];unchanged=0;inputs=[source_manifest,source/'resolved_config.yaml',source/'status.json']
    for entry in manifest['data']['files']:
        path=Path(entry['location']).resolve()
        current=file_identity(path) if path.exists() else None
        if current is not None:
            inputs.append(path)
        if current is not None and current['sha256']==entry['sha256'] and current['size_bytes']==entry['size_bytes']:
            unchanged+=1
            continue
        if not path.is_relative_to(root) or path.suffix not in ('.json','.yaml','.yml','.toml','.txt','.md','.py'):
            raise ValueError('Cannot recover changed untracked/binary source data')
        relative=path.relative_to(root).as_posix()
        result=subprocess.run(['git','-C',str(root),'show',commit+':'+relative],capture_output=True,check=False)
        if result.returncode:
            raise ValueError('Input absent from its recorded Git commit: '+relative)
        raw=matching_git_bytes(result.stdout,entry)
        modified.append((entry,current,relative,raw))
    effective=dict(schema_version='sumodiff.historical_git_inputs.v1',source_manifest=file_identity(source_manifest),
                   source_git=manifest['git'],unchanged_inputs_verified=unchanged,
                   recovery='recorded Git blob; exact LF or CRLF bytes must match original SHA and size')
    with RunRecorder(output,Path.cwd(),effective,[sys.executable,*sys.orig_argv[1:]],
                     manifest['seeds'],purpose='recover_byte_identical_historical_git_inputs',
                     data_files=sorted(set(inputs)),data_id=manifest['data']['id'],formal=formal) as recorder:
        snapshots=[]
        for original,current,relative,raw in modified:
            path=recorder.output/'snapshots'/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(raw)
            snapshots.append(dict(original=original,current=current,snapshot=file_identity(path)))
            print('Recovered exact historical input: '+relative,flush=True)
        bundle=dict(schema_version='sumodiff.historical_git_inputs.v1',
                    source_manifest=file_identity(source_manifest),source_git=manifest['git'],snapshots=snapshots)
        write_json(recorder.output/'snapshots.json',bundle)
        recorder.write_metrics(dict(recovered_inputs=len(snapshots),unchanged_inputs_verified=unchanged,
                                    optimizer_updates=0,source_files_modified=0))
    return bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--formal',action='store_true')
    args=parser.parse_args()
    run(args.source,args.output,args.formal)


if __name__=='__main__':
    main()
