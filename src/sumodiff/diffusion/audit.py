"""Compare actual checkpoint and optimizer/RNG states after a resumed run."""
import json
from pathlib import Path
import torch
from sumodiff.experiments.recorder import RunRecorder,file_identity
from sumodiff.data.dataset import write_json
from .checkpoint import load_checkpoint


def exact_tree(left,right):
    if type(left)!=type(right): return False
    if isinstance(left,torch.Tensor): return left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right)
    if isinstance(left,dict): return left.keys()==right.keys() and all(exact_tree(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)): return len(left)==len(right) and all(exact_tree(a,b) for a,b in zip(left,right))
    return left==right


def audit_continuation(uninterrupted,resumed,output,repository,command,formal=False):
    a=load_checkpoint(uninterrupted);b=load_checkpoint(resumed)
    left_run=Path(uninterrupted).parent;right_run=Path(resumed).parent
    sources=[uninterrupted,resumed,*[p/name for p in (left_run,right_run) for name in ('manifest.json','resolved_config.yaml','steps.jsonl','status.json')]]
    with RunRecorder(output,repository,dict(schema_version='sumodiff.continuation.audit.config.v1',uninterrupted=file_identity(uninterrupted),resumed=file_identity(resumed),comparison='bitwise tensors, optimizer and RNG; step trace excludes timing'),
        command,{'audit':0},purpose='stage5_actual_continuation_audit',data_files=sources,formal=formal) as run:
        for p in (left_run,right_run):
            if json.loads((p/'status.json').read_text())['state']!='completed': raise ValueError('Source training run is not completed')
        parent=b['metadata']['parent_checkpoint']
        if not parent: raise ValueError('Second checkpoint did not resume training')
        parent_file=Path(parent['location']);sources_parent=file_identity(parent_file)
        if sources_parent['sha256']!=parent['sha256']: raise ValueError('Resume source changed')
        mid=load_checkpoint(parent_file)
        if mid['metadata']['run_sha']!=a['metadata']['run_sha'] or mid['step']>=a['step']: raise ValueError('Resume parent is not an earlier checkpoint of the uninterrupted code run')
        first=[json.loads(line) for line in (left_run/'steps.jsonl').read_text().splitlines()]
        second=[json.loads(line) for line in (right_run/'steps.jsonl').read_text().splitlines()]
        fields=('step','noise_mse','gradient_norm_before_clip','window_ids','timesteps')
        first=[{k:r[k] for k in fields} for r in first if r['step']>mid['step']]
        second=[{k:r[k] for k in fields} for r in second]
        result=dict(schema_version='sumodiff.continuation.audit.metrics.v1',run_sha=run.manifest['git']['commit_sha'],
            final_step=a['step'],resume_step=mid['step'],training_signature_equal=exact_tree(a['metadata']['training_signature'],b['metadata']['training_signature']),
            final_step_equal=a['step']==b['step'],model_equal=exact_tree(a['model'],b['model']),optimizer_equal=exact_tree(a['optimizer'],b['optimizer']),
            noise_generator_equal=exact_tree(a['generator_state'],b['generator_state']),cpu_rng_equal=exact_tree(a['cpu_rng'],b['cpu_rng']),
            cuda_rng_equal=exact_tree(a['cuda_rng'],b['cuda_rng']),continuation_trace_equal=first==second,compared_updates=len(second),
            source_uninterrupted_sha=a['metadata']['run_sha'],source_resumed_sha=b['metadata']['run_sha'],parent_checkpoint=sources_parent,
            scope='identical local runtime/device/precision and data; cross-version bitwise equality not promised')
        write_json(run.output/'continuation_audit.json',result);run.write_metrics(result)
        if not all(result[k] for k in ('training_signature_equal','final_step_equal','model_equal','optimizer_equal','noise_generator_equal','cpu_rng_equal','cuda_rng_equal','continuation_trace_equal')):
            raise RuntimeError('Actual continuation audit failed; differences retained')
    return result
