# Hyperparameter Experiment Results

**Member Name:** Dan Paul Dushime

**Environment:** `ALE/Pong-v5`
**Fixed settings for all 10 runs:** total steps = 500,000, buffer size = 50,000, learning starts = 10,000, train every 4 steps, target update every 1,000 steps.

_Models, logs and plots: `models/Dan_Paul_Dushime/exp_XX/`. "Mean reward" is the average over the last 20 episodes in `training_log.csv`. A Pong score ranges from -21 (lost every point) to +21 (won every point)._

| Exp | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Mean reward (last 20 ep.) | Best episode | Observed behaviour |
|-----|---------------|-------|------------|---------------|-------------|-----------------------|-------------|--------------|--------------------|
| 1 | 0.00025 | 0.99 | 32 | 1.0 | 0.05 | 100000 | -20.9 | -18 | Did not learn: reward stayed near -21 (start -20.1, end -20.9); short games (~800 steps) because the agent lost every point quickly. |
| 2 | 0.00025 | 0.95 | 32 | 1.0 | 0.05 | 100000 | -20.9 | -17 | Did not learn: reward stayed near -21 (start -20.1, end -20.9); short games (~809 steps) because the agent lost every point quickly. |
| 3 | 0.00025 | 0.90 | 32 | 1.0 | 0.05 | 100000 | -16.8 | -11 | Learned: average reward improved from -20.1 to -16.8; games got longer (932 -> 1736 steps) as the agent returned more balls. |
| 4 | 0.00025 | 0.999 | 32 | 1.0 | 0.05 | 100000 | -20.9 | -18 | Did not learn: reward stayed near -21 (start -20.1, end -20.9); short games (~809 steps) because the agent lost every point quickly. |
| 5 | 0.00025 | 0.97 | 32 | 1.0 | 0.05 | 100000 | -20.8 | -18 | Did not learn: reward stayed near -21 (start -20.1, end -20.8); short games (~813 steps) because the agent lost every point quickly. |
| 6 | 0.00025 | 0.995 | 32 | 1.0 | 0.05 | 100000 | -20.7 | -18 | Did not learn: reward stayed near -21 (start -20.4, end -20.7); short games (~824 steps) because the agent lost every point quickly. |
| 7 | 0.00025 | 0.80 | 32 | 1.0 | 0.05 | 100000 | -20.2 | -13 | Very little learning: average reward still about -20 (start -20.2, end -20.2), but games got longer (~1150 steps) and its best game reached -13. |
| 8 | 0.00025 | 0.95 | 64 | 1.0 | 0.05 | 100000 | -20.9 | -18 | Did not learn: reward stayed near -21 (start -20.0, end -20.9); short games (~796 steps) because the agent lost every point quickly. |
| 9 | 0.0001 | 0.999 | 32 | 1.0 | 0.05 | 100000 | -18.9 | -13 | Learned: average reward improved from -20.3 to -18.9; games got longer (918 -> 1821 steps) as the agent returned more balls. |
| 10 | 0.0001 | 0.97 | 64 | 1.0 | 0.05 | 100000 | -10.3 | 8 | Learned: average reward improved from -20.1 to -10.3; games got longer (934 -> 3283 steps) as the agent returned more balls. |

## Best and worst configuration

- **Best: Exp 10** (lr 0.0001, gamma 0.97, batch 64) reached **-10.3**, and its best game was **+8**: it won that game.
- **Worst: Exp 4** (lr 0.00025, gamma 0.999) stayed at **-20.9**. A gamma close to 1 makes the target depend on rewards very far in the future, so targets are large and noisy. Together with a high learning rate, the network never stabilised.

## What helped and what hurt

- **Learning rate (the main lesson):** Exp 1-8 used lr 0.00025, and almost all of them failed (about -20.9), whatever the gamma. When I lowered the learning rate to 0.0001 (Exp 9, 10), the agent learned. So 0.00025 was too high for our setup, and this hid most of the gamma effect.
- **Gamma:** at lr 0.00025, only **gamma 0.9 learned clearly (Exp 3, -16.8)**. A lower gamma makes targets smaller (future rewards count less), which makes the updates more stable. But **gamma 0.8 (Exp 7)** barely learned (-20.2): it looks only about 5 steps ahead (1 / (1 - 0.8)), and in Pong a point is scored many steps after the paddle hits the ball, so the agent cannot link its action to the reward.
- **Gamma 0.999 (Exp 4, 9)** looks about 1,000 steps ahead. Even with the lower learning rate (Exp 9, -18.9), it learned much more slowly than gamma 0.97 (Exp 10, -10.3).
- **Batch size:** batch 64 helped at lr 0.0001 (Exp 10) but could not rescue lr 0.00025 (Exp 8, -20.9).

## Final configuration and justification

**lr 0.0001, gamma 0.97, batch 64, epsilon 1.0 -> 0.05 over 100,000 steps** (Exp 10). Gamma 0.97 looks about 33 steps ahead. That is long enough to connect hitting the ball with scoring, but short enough to keep the targets stable. This was the 3rd best run of the whole group (after Jolly's Exp 7 and 6), and the only run in my set that won a game.
