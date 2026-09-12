# Baseline, feature batch of 2026-09-12

Recorded BEFORE any change, on commit `048716a` ("baseline: carsim before feature batch").

```
python3 -m drive.validate              ->  82/82 pass   0 HARD   0 soft   [110.6 s]
python3 -m drive.validate --modules    -> 100/100 pass  0 HARD   0 soft   [229.6 s]
```

So "100/100" is the `--modules` figure (82 sim groups + 18 module self-checks);
the plain suite is 82/82. Both are green at baseline. Every task below must end
with BOTH numbers restored.

Two known findings the suite itself reports and which are NOT ours to fix
(CONTRACT.md section 10): `qss.residuals` Y_r algebra, and `alpha_peak_deg = 7.0`.
They print at the end of every run. They are not failures.

Deps in this environment: pygame 2.5.2, numpy 2.4.4, scipy 1.17.1. torch IS importable in this environment (2.x) but the repo uses none of it; task 9 stays numpy-only on purpose (determinism + no new dependency).
