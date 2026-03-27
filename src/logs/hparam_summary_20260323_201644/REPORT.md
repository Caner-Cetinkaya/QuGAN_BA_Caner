# CGAN Hyperparameter Test Report

Generated: 2026-03-23T20:16:44
Evaluated completed runs: 21

## Ranking (best first)
- 1. score=-0.1137, d_lr=0.0005, g_lr=0.0005, d_steps=1, clip=1.0, loss_d=0.6887, loss_g=0.7105, real=0.5676, fake_d=0.5090, fake_g=0.5113, notes=ok, source=grid-search
- 2. score=-0.1692, d_lr=0.001, g_lr=0.001, d_steps=1, clip=1.0, loss_d=0.6676, loss_g=0.7391, real=0.5033, fake_d=0.4454, fake_g=0.4558, notes=ok, source=grid-search
- 3. score=-0.1812, d_lr=0.002, g_lr=0.002, d_steps=1, clip=0.5, loss_d=0.6932, loss_g=0.7558, real=0.5345, fake_d=0.4838, fake_g=0.4958, notes=ok, source=grid-search
- 4. score=-0.2122, d_lr=0.0005, g_lr=0.0005, d_steps=1, clip=0.5, loss_d=0.6563, loss_g=0.7486, real=0.5304, fake_d=0.4521, fake_g=0.4577, notes=ok, source=grid-search
- 5. score=-0.2273, d_lr=0.001, g_lr=0.001, d_steps=1, clip=1.0, loss_d=0.6946, loss_g=0.7827, real=0.4741, fake_d=0.4971, fake_g=0.5040, notes=balanced-discriminator, source=manual
- 6. score=-0.2977, d_lr=0.0005, g_lr=0.0005, d_steps=1, clip=1.0, loss_d=0.6315, loss_g=0.8206, real=0.5059, fake_d=0.4369, fake_g=0.4549, notes=ok, source=grid-search
- 7. score=-0.2977, d_lr=0.0005, g_lr=0.0005, d_steps=1, clip=1.0, loss_d=0.6315, loss_g=0.8206, real=0.5059, fake_d=0.4369, fake_g=0.4549, notes=ok, source=grid-search
- 8. score=-0.3160, d_lr=0.0005, g_lr=0.0005, d_steps=1, clip=1.0, loss_d=0.6876, loss_g=0.6713, real=0.6072, fake_d=0.5848, fake_g=0.5895, notes=balanced-discriminator, source=grid-search
- 9. score=-0.3730, d_lr=0.002, g_lr=0.002, d_steps=1, clip=1.0, loss_d=0.6475, loss_g=0.8611, real=0.4886, fake_d=0.4721, fake_g=0.4883, notes=balanced-discriminator, source=grid-search
- 10. score=-0.4358, d_lr=0.001, g_lr=0.001, d_steps=1, clip=1.0, loss_d=0.6698, loss_g=0.8469, real=0.6145, fake_d=0.4492, fake_g=0.5019, notes=ok, source=grid-search
- 11. score=-0.4358, d_lr=0.001, g_lr=0.001, d_steps=1, clip=1.0, loss_d=0.6698, loss_g=0.8469, real=0.6145, fake_d=0.4492, fake_g=0.5019, notes=ok, source=grid-search
- 12. score=-0.4912, d_lr=0.002, g_lr=0.002, d_steps=1, clip=1.0, loss_d=0.6954, loss_g=0.8239, real=0.5147, fake_d=0.4525, fake_g=0.4586, notes=ok, source=grid-search
- 13. score=-0.4912, d_lr=0.002, g_lr=0.002, d_steps=1, clip=1.0, loss_d=0.6954, loss_g=0.8239, real=0.5147, fake_d=0.4525, fake_g=0.4586, notes=ok, source=grid-search
- 14. score=-1.1512, d_lr=0.001, g_lr=0.001, d_steps=1, clip=0.5, loss_d=0.6465, loss_g=1.1367, real=0.8310, fake_d=0.2559, fake_g=0.3744, notes=ok, source=grid-search
- 15. score=-1.6655, d_lr=0.002, g_lr=0.002, d_steps=2, clip=1.0, loss_d=0.6047, loss_g=0.9893, real=0.5582, fake_d=0.4222, fake_g=0.5212, notes=possible-gradient-explosion, source=grid-search
- 16. score=-2.0669, d_lr=0.0005, g_lr=0.0005, d_steps=2, clip=0.5, loss_d=0.6995, loss_g=1.2410, real=0.5157, fake_d=0.5089, fake_g=0.5842, notes=possible-gradient-explosion,balanced-discriminator, source=grid-search
- 17. score=-2.4354, d_lr=0.0005, g_lr=0.0005, d_steps=2, clip=1.0, loss_d=0.5344, loss_g=1.2985, real=0.7188, fake_d=0.2045, fake_g=0.2182, notes=possible-gradient-explosion, source=grid-search
- 18. score=-2.4354, d_lr=0.0005, g_lr=0.0005, d_steps=2, clip=1.0, loss_d=0.5344, loss_g=1.2985, real=0.7188, fake_d=0.2045, fake_g=0.2182, notes=possible-gradient-explosion, source=grid-search
- 19. score=-2.6906, d_lr=0.001, g_lr=0.001, d_steps=2, clip=0.5, loss_d=0.4496, loss_g=1.7220, real=0.4468, fake_d=0.4107, fake_g=0.4662, notes=possible-gradient-explosion,balanced-discriminator, source=grid-search
- 20. score=-2.8391, d_lr=0.001, g_lr=0.001, d_steps=2, clip=1.0, loss_d=0.5209, loss_g=1.6269, real=0.6118, fake_d=0.2457, fake_g=0.3230, notes=possible-gradient-explosion, source=grid-search
- 21. score=-3.2743, d_lr=0.001, g_lr=0.001, d_steps=2, clip=1.0, loss_d=0.4794, loss_g=2.1864, real=0.4546, fake_d=0.3267, fake_g=0.7046, notes=possible-gradient-explosion, source=grid-search

## Recommended Configuration
- disc_learning_rate: 0.0005
- gen_learning_rate: 0.0005
- disc_steps_per_gen: 1
- grad_clip_norm: 1.0
- disc_warmup_steps: 0
- latent_distribution: uniform
- label_real / label_fake: 1.0 / 0.0

## Interpretation
- Ziel war ein balanciertes adversariales Spiel ohne Gradienten-Explosion und ohne Generator-Stillstand.
- Die besten Runs zeigen moderate, stabile Verlaufswerte statt extremer Sättigung.
- Für finale Entscheidungen sollte ein längerer Follow-up-Run (z.B. >= 1000 Schritte) mit der Top-Konfiguration erfolgen.