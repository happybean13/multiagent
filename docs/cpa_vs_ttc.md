# CPA Stackelberg vs. range-rate TTC Stackelberg

> A self-contained writeup of the leader-selection upgrade used by
> `copo_code/copo/torch_stackelberg/env_stackelberg_cpa.py` /
> `copo_code/copo/torch_stackelberg/ttc_cpa.py`, plus the empirical
> comparison against the original range-rate TTC and CoPO baselines.

---

## 1. What the original "TTC" Stackelberg actually computes

The pre-existing leader selector (`compute_pairwise_ttc_leaders` in
`CoPO/copo_code/copo/torch_copo/utils/env_wrappers.py`) is a **1D
range-rate TTC**: it projects the relative velocity onto the line of
sight between two cars and divides current distance by that
projection.

Let

- $p_i, p_j \in \mathbb{R}^2$ be the 2D positions of two agents,
- $v_i, v_j \in \mathbb{R}^2$ their velocities,
- $\hat u_{ij} = (p_j - p_i) / \lVert p_j - p_i \rVert$ the unit bearing.

Then

$$
\mathrm{TTC}^{\text{RR}}_{ij} \;=\;
\begin{cases}
\dfrac{\lVert p_j - p_i \rVert}{(v_i - v_j)\cdot \hat u_{ij}}
& \text{if } (v_i - v_j)\cdot \hat u_{ij} > 0, \\[6pt]
+\infty & \text{otherwise.}
\end{cases}
$$

This formula has three structural problems for an intersection environment:

1. **No miss-distance gate.** Any positive closing component along the
   bearing produces a finite TTC, even when the two trajectories pass
   several metres apart at closest approach. A car overtaking in the
   next lane gets flagged as a conflict.
2. **2D geometry collapsed to 1D.** A perpendicular crossing whose
   paths intersect exactly at the junction can have a small
   bearing-line projection (most of the velocity is tangential to the
   bearing) and therefore an *under*-estimated urgency. Conversely a
   non-crossing tangential pass can look "closing".
3. **Implicit dependence on bearing alignment.** Same-road behind /
   beside cars only register as conflicts when the closing component
   along the bearing is positive — that depends on relative speed, not
   on whether the paths actually meet.

So the formula is *not* anchored to the junction (that was a slight
misconception in early discussion), but it still ignores the actual
collision geometry.

---

## 2. The CPA (Closest Point of Approach) upgrade

CPA replaces the 1D bearing projection with the **actual extremum** of

$$
d(t) \;=\; \lVert p_i(t) - p_j(t) \rVert
$$

under a constant-velocity assumption. Let

$$
\Delta p \;=\; p_i - p_j, \qquad \Delta v \;=\; v_i - v_j,
$$

so that

$$
\Delta p_t \;=\; \Delta p + t\,\Delta v.
$$

Setting $\tfrac{d}{dt} \lVert \Delta p_t\rVert^2 = 0$ gives the time of
closest approach

$$
t^\ast \;=\; -\,\dfrac{\Delta p \cdot \Delta v}{\lVert \Delta v \rVert^2},
$$

and the miss distance at that time

$$
d_{\min} \;=\; \lVert \Delta p + t^\ast \Delta v \rVert
\;=\; \dfrac{\lvert \Delta p \times \Delta v \rvert}{\lVert \Delta v \rVert}.
$$

(Both forms are algebraically equivalent in 2D; the cross-product form
is sometimes more numerically stable.)

A pair is declared a **conflict** iff all three gates pass:

$$
\lVert \Delta v\rVert^2 > \varepsilon, \qquad
t^\ast > 0, \qquad
d_{\min} < r_{\text{safe}}.
$$

If any gate fails, the pairwise CPA TTC is $+\infty$ (no future
conflict). Otherwise

$$
\mathrm{TTC}^{\text{CPA}}_{ij} \;=\; t^\ast.
$$

### Why this is a strict upgrade over $\mathrm{TTC}^{\text{RR}}$

| Failure mode of range-rate TTC | How CPA handles it |
|---|---|
| Lane-adjacent overtake with positive bearing-projection but $d_{\min} \approx$ lane width | $d_{\min}$ gate rejects it. |
| Perpendicular crossing meeting at junction, small bearing closing speed | $t^\ast$ is the true crossing time, $d_{\min}\approx 0$, fires correctly. |
| Parallel motion, $\lVert \Delta v\rVert \to 0$ | First gate rejects (no real closure). |
| Diverging pair, already past closest approach ($t^\ast \le 0$) | Second gate rejects. |

The implementation in `ttc_cpa.py` uses $r_{\text{safe}} = 3.0$ m
(MetaDrive vehicles are $4.4$ m long, $1.8$ m wide — so $3.0$ m
center-to-center is a meaningful "near-miss" radius rather than an
exact contact). The 10 m current-distance prune from the original is
kept to avoid global $O(N^2)$ cost across all agents in the scene.

---

## 3. Plugging CPA into the Stackelberg game

The Stackelberg coordination is *decoupled* from the per-pair TTC
formula: the algorithm only consumes `info[i]["leader_id"]`. So the
leader-assignment rule is **identical to the original** — only the
per-pair urgency that feeds into it changes.

For each agent $i$, let

$$
\mathrm{TTC}^\ast_i \;=\; \min_{j\ne i}\ \mathrm{TTC}^{\text{CPA}}_{ij},
\qquad
j_i^\ast \;=\; \arg\min_{j\ne i}\ \mathrm{TTC}^{\text{CPA}}_{ij}.
$$

The leader of agent $i$ is then

$$
\ell(i) \;=\;
\begin{cases}
j_i^\ast & \text{if } \mathrm{TTC}^\ast_{j_i^\ast} \;<\; \mathrm{TTC}^\ast_i, \\
\varnothing & \text{otherwise.}
\end{cases}
$$

Interpretation: *the agent whose own earliest conflict arrives sooner
becomes leader*. They get to "go first"; the agent with more headroom
yields. The Stackelberg PPO policy then conditions its centralised
critic on the leader's observation/action signal:

$$
V_\phi\!\Big(o_i,\ \hat a_{\ell(i)},\ \mathbb{1}[\ell(i) \ne \varnothing]\Big),
\qquad
\hat a_{\ell(i)} \;=\; \pi_\theta(o_{\ell(i)})\text{.mean (detached).}
$$

**Nothing about the actor or the reward changes.** The improvement
comes entirely from feeding the critic a cleaner, geometrically
grounded leader signal: fewer false conflicts on overtakes, no missed
perpendicular crossings, and a stable tie-breaking gate via $d_{\min}$
which makes the leader-assignment less brittle to noisy velocity
estimates.

---

## 4. Empirical fingerprint (n = 3 seeds, 1 M env steps)

Final-iteration means and per-seed spread across seeds $\{0, 100,
200\}$ on `MultiAgentIntersectionEnv`:

| Metric | TTC (range-rate) | CoPO baseline | **CPA (ours)** |
|---|---|---|---|
| episode reward | $105.3 \pm 8.7$ | $96.7 \pm 11.8$ | **$110.3 \pm 3.7$** |
| success rate | $0.703 \pm 0.114$ | $0.574 \pm 0.165$ | **$0.776 \pm 0.041$** |
| crash rate | $0.215 \pm 0.069$ | $0.183 \pm 0.030$ | $0.183 \pm 0.029$ |
| out-of-road | $0.067 \pm 0.025$ | $0.075 \pm 0.016$ | **$0.042 \pm 0.008$** |

Two things to read here:

1. **Mean improvement.** CPA gains $+7.3$ pp success over TTC and
   $+20$ pp over CoPO; reward $+5$ over TTC, $+14$ over CoPO;
   out-of-road is lowest of the three.
2. **Variance collapse.** CPA's standard deviation on success drops to
   $\approx 0.04$ versus $\approx 0.11$ for TTC (about $3\times$) and
   $\approx 0.17$ for CoPO (about $4\times$). TTC has a "lottery" seed
   (seed 200, success $0.54$) that CPA does not produce. This is
   consistent with the theoretical argument: the $d_{\min}$ gate
   suppresses spurious conflict assignments, so the leader signal is
   less seed-noisy and the critic learns a more transferable yielding
   policy.

### The honest negative finding

CPA's crash rate matches CoPO's ($0.183$) rather than improving on
it. CPA shifts decisions from "wrong / no leader" to "right leader"
but it does not directly enforce a deceleration: collisions that
happen *despite* a correct leader assignment are unaffected.

A natural next step (left to future work) is to use $t^\ast$ and
$d_{\min}$ as a **graded urgency signal** — either as additional
inputs to the critic, or as a small shaping reward $\propto -1 / t^\ast$
near conflict — rather than only as a binary leader selector.

---

## 5. Reproducibility

Training was done with
`CoPO/copo_code/copo/torch_stackelberg/train_minimal_stackelberg_cpa_1M.py`:

```
PYTHONPATH=$PWD/copo_code python copo_code/copo/torch_stackelberg/train_minimal_stackelberg_cpa_1M.py \
    --exp-name minimal_stackelberg_cpa_1M \
    --num-seeds 3 --start-seed 0 \
    --cpa-r-safe 3.0 --cpa-neighbor-radius 10.0
```

Plots are reproduced with `plot_publishable_ablation.py`:

```
python plot_publishable_ablation.py \
    --stack-dir minimal_stackelberg_1M,minimal_stackelberg_1M_s200 \
    --copo-dir  copo_baseline_1M,copo_baseline_1M_s200 \
    --cpa-dir   minimal_stackelberg_cpa_1M \
    --out-prefix publishable_ablation_3way_n3 \
    --title-suffix "n=3 seeds (0, 100, 200)"
```

Outputs: `publishable_ablation_3way_n3_curves.png`,
`publishable_ablation_3way_n3_final_bar.png`,
`publishable_ablation_3way_n3_summary.csv`.
