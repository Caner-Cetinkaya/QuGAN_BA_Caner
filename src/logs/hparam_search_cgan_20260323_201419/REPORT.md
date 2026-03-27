# CGAN Hyperparameter Search Report

Generated: 2026-03-23T20:15:26
Total runs: 4
Successful runs: 4

## Method
- Phase 1: coarse search over balanced learning rates, discriminator steps, and gradient clipping.
- Phase 2: local refinement around best coarse candidates.
- Scoring objective: balanced adversarial equilibrium, low oscillation, and no gradient pathologies.

## Top Configurations
- run_004: score=-0.1137, d_lr=0.0005, g_lr=0.0005, d_steps=1, clip=1.0, loss_d=0.6887, loss_g=0.7105, score_real=0.5676, score_fake_d=0.5090, notes=ok
- run_002: score=-0.1692, d_lr=0.001, g_lr=0.001, d_steps=1, clip=1.0, loss_d=0.6676, loss_g=0.7391, score_real=0.5033, score_fake_d=0.4454, notes=ok
- run_001: score=-0.3160, d_lr=0.0005, g_lr=0.0005, d_steps=1, clip=1.0, loss_d=0.6876, loss_g=0.6713, score_real=0.6072, score_fake_d=0.5848, notes=balanced-discriminator
- run_003: score=-0.3730, d_lr=0.002, g_lr=0.002, d_steps=1, clip=1.0, loss_d=0.6475, loss_g=0.8611, score_real=0.4886, score_fake_d=0.4721, notes=balanced-discriminator

## Recommended Next Default
- disc_learning_rate: 0.0005
- gen_learning_rate: 0.0005
- disc_steps_per_gen: 1
- grad_clip_norm: 1.0
- training_steps (search context): 400