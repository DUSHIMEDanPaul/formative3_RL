# Hyperparameter Experiment Results

**Member Name:** Thierry Alain Tresor Ibyishaka

**Environment:** `ALE/Pong-v5`
**Fixed settings for all 10 runs:** total steps = 500,000, buffer size = 50,000, learning starts = 10,000, train every 4 steps, target update every 1,000 steps.

_Models, logs and plots: `models/Thierry_Alain_Tresor_Ibyishaka/exp_XX/`. "Mean reward" is the average over the last 20 episodes in `training_log.csv`. A Pong score ranges from -21 (lost every point) to +21 (won every point)._

| Exp | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Mean reward (last 20 ep.) | Best episode | Observed behaviour |
|-----|---------------|-------|------------|---------------|-------------|-----------------------|-------------|--------------|--------------------|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.1 | 100000 | -17.1 | -13 | Learned: average reward improved from -20.4 to -17.1; games got longer (902 -> 2285 steps) as the agent returned more balls. |
| 2 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 100000 | -16.5 | -10 | Learned: average reward improved from -20.4 to -16.5; games got longer (892 -> 2928 steps) as the agent returned more balls. |
| 3 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 50000 | -13.8 | 4 | Learned: average reward improved from -20.5 to -13.8; games got longer (924 -> 2595 steps) as the agent returned more balls. |
| 4 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 200000 | -13.4 | -8 | Learned: average reward improved from -20.0 to -13.4; games got longer (964 -> 2254 steps) as the agent returned more balls. |
| 5 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 25000 | -19.0 | -17 | Learned: average reward improved from -20.4 to -19.0; games got longer (896 -> 1642 steps) as the agent returned more balls. |
| 6 | 0.0001 | 0.99 | 32 | 0.5 | 0.05 | 100000 | -11.4 | -1 | Learned: average reward improved from -20.4 to -11.4; games got longer (913 -> 2323 steps) as the agent returned more balls. |
| 7 | 0.0001 | 0.99 | 32 | 1.0 | 0.2 | 100000 | -17.6 | -12 | Learned: average reward improved from -20.4 to -17.6; games got longer (903 -> 1837 steps) as the agent returned more balls. |
| 8 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 250000 | -13.1 | -7 | Learned: average reward improved from -20.1 to -13.1; games got longer (969 -> 2306 steps) as the agent returned more balls. |
| 9 | 0.0001 | 0.99 | 32 | 0.8 | 0.02 | 150000 | -12.4 | -5 | Learned: average reward improved from -20.3 to -12.4; games got longer (905 -> 2310 steps) as the agent returned more balls. |
| 10 | 0.00025 | 0.99 | 64 | 1.0 | 0.05 | 100000 | -20.9 | -17 | Did not learn: reward stayed near -21 (start -20.3, end -20.9); short games (~816 steps) because the agent lost every point quickly. |

## Best and worst configuration

- **Best: Exp 6** (epsilon start 0.5) reached **-11.4**, my best average. Starting at 0.5 instead of 1.0 means the agent uses what it has learned earlier, instead of spending the first 100k steps playing mostly at random. In Pong, random play almost never scores, so fully random early experience teaches little.
- **Worst: Exp 10** (lr 0.00025, batch 64) stayed at **-20.9**. This shows the learning rate was too high, not an epsilon problem. Among the runs that only changed epsilon, the worst was **Exp 5** (decay over 25k steps, **-19.0**): exploration stopped before the agent had collected varied experience, so it locked in a poor strategy.

## What helped and what hurt

All runs except Exp 10 used lr 0.0001, gamma 0.99 and batch 32. The comparison point is the default schedule 1.0 -> 0.05 over 100k steps (-17.4, Jolly's Exp 1).

- **Epsilon decay speed:** too fast hurt (25k steps -> -19.0). 50k steps (Exp 3, -13.8) and slower decays helped: 200k steps -> -13.4 (Exp 4), and 250k steps with end 0.01 -> -13.1 (Exp 8).
- **Epsilon end:** keeping 20% random actions forever (end 0.2, Exp 7) gave -17.6, and end 0.1 (Exp 1) gave -17.1. Random moves keep costing points even after the agent has learned. A lower end value (0.01, Exp 2) was slightly better (-16.5).
- **Epsilon start:** lower starts were the best runs: 0.5 (Exp 6, -11.4) and 0.8 -> 0.02 over 150k (Exp 9, -12.4).
- **Caveat:** each setting was run once (one random seed). Differences of 1-2 points may be partly luck. The big effects (very fast decay, high epsilon end, high learning rate) are clear.

## Final configuration and justification

**lr 0.0001, gamma 0.99, batch 32, epsilon 0.5 -> 0.05 over 100,000 steps** (Exp 6). It was my best run and the 4th best in the group. Exp 9 (0.8 -> 0.02 over 150k) was close behind. Together they suggest the agent needs some early exploration, but not 100% random play, and that epsilon should end low so random moves don't waste points.
