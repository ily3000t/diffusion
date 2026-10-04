# Configuration contracts

Stage 0 consumes experiments/manifest_smoke.yaml. Stage 1 consumes individual scenarios/*.yaml profiles through strict defaults and override validation, saving all effective settings. scenarios/families.yaml remains a scope description, not a runnable profile. Other categories still describe future candidates, not implemented engines or calibrated values. Diagnostic fixture profiles are explicitly excluded from normal training.
