from copy import deepcopy
import pytest
from sumodiff.diffusion.stage5c import training_command
from sumodiff.diffusion.unified_training import DEFAULTS,unified_config,unified_signature
from sumodiff.diffusion.unified_losses import UnifiedLossConfig
from sumodiff.models import ModelConfig
from test_controlled_training import parent_fixture


def test_direct_command_records_replayable_inputs_and_original_budget():
    c=dict(dataset='data',parent_checkpoint='original.pt',scale_audit='audit.json',
           training_configs={'unified':'loss.yaml','epsilon_cosine':'control.yaml'})
    command=training_command(c,'unified','run',True,resume='mid.pt',prefix_updates=16)
    assert 'train-direct' in command and '--formal' in command
    for key,value in [('dataset','data'),('parent-checkpoint','original.pt'),('scale-audit','audit.json'),
                      ('config','loss.yaml'),('output','run'),('resume','mid.pt'),('prefix-updates','16')]:
        assert command[command.index('--'+key)+1]==value
    assert '--allow-long-run' not in command
    assert '--allow-long-run' in training_command(c,'epsilon_cosine','run',True)


def test_recovery_signature_does_not_let_objective_budget_or_schedule_change():
    parent=parent_fixture()
    value=dict(DEFAULTS,expected_parent_sha256='1'*64,loss=UnifiedLossConfig().to_dict())
    c=unified_config(value,parent)
    m=parent['metadata']
    reference=unified_signature(ModelConfig(),c,m['data'],m['scale_audit'],['train'],['val'])
    for mutation in (lambda x:x['schedule'].update(peak_lr=2e-4),
                     lambda x:x['loss'].update(imitation_weight=.02),
                     lambda x:x['monitor'].update(seed=99),
                     lambda x:x.update(planned_additional_updates=3000)):
        other=deepcopy(c);mutation(other)
        assert unified_signature(ModelConfig(),other,m['data'],m['scale_audit'],['train'],['val'])!=reference
