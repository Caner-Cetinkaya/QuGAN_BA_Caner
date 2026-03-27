# CGAN Hyperparameter Search Report

Generated: 2026-03-23T20:20:02
Total runs: 20
Successful runs: 20

## Method
- Phase 1: coarse search over balanced learning rates, discriminator steps, and gradient clipping.
- Phase 2: local refinement around best coarse candidates.
- Scoring objective: balanced adversarial equilibrium, low oscillation, and no gradient pathologies.

## Top Configurations
- run_018: score=-0.0347, d_lr=0.001, g_lr=0.001, d_steps=1, clip=1.0, loss_d=0.6920, loss_g=0.6923, score_real=0.5120, score_fake_d=0.4936, notes=balanced-discriminator
- run_015: score=-0.1008, d_lr=0.004, g_lr=0.004, d_steps=1, clip=1.0, loss_d=0.6912, loss_g=0.7191, score_real=0.5080, score_fake_d=0.4861, notes=balanced-discriminator
- run_016: score=-0.1295, d_lr=0.00025, g_lr=0.00025, d_steps=1, clip=1.0, loss_d=0.6654, loss_g=0.7344, score_real=0.4806, score_fake_d=0.5057, notes=balanced-discriminator
- run_009: score=-0.1812, d_lr=0.002, g_lr=0.002, d_steps=1, clip=0.5, loss_d=0.6932, loss_g=0.7558, score_real=0.5345, score_fake_d=0.4838, notes=ok
- run_013: score=-0.1966, d_lr=0.001, g_lr=0.001, d_steps=1, clip=1.0, loss_d=0.6886, loss_g=0.7385, score_real=0.5375, score_fake_d=0.4561, notes=ok

## Recommended Next Default
- disc_learning_rate: 0.001
- gen_learning_rate: 0.001
- disc_steps_per_gen: 1
- grad_clip_norm: 1.0
- training_steps (search context): 700