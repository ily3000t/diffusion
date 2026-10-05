# models

Stage4 implements shared current-history/map/route encoders, equal-parameter hierarchical/parallel fusion and a temporal U-Net with per-time vehicle attention. See docs/conditional_model.md and docs/progress.md.

Public exports: ModelConfig, prepare_conditioning, ConditionEncoder, ConditionalDenoiser. This is an untrained epsilon predictor. No diffusion sampling, training or compatible checkpoint exists yet. Unsupported extension flags fail explicitly.
