# Measured-Threshold, Delay-Driven Supervisory Control for Teleoperation

Simulation code for the paper *"Measured-Threshold, Delay-Driven Supervisory Control for Teleoperation under Normalized Delay: A Simulation Study on a Nonlinear Two-Link Manipulator"* by Nandana Mihira Narendra, Giri Pranesh K, and M P Anbarasi (Dept. of Robotics and Automation, PSG College of Technology, Coimbatore, India).

The code simulates a planar two-link manipulator teleoperated over a delayed channel. It compares three fixed controllers with a supervisor that switches between them according to a measured one-way delay. The switching thresholds are not tuned by hand. They are the first persistent sign changes in mean tracking error measured on the plant, and the hysteresis bands come from bootstrap confidence intervals over random seeds.

Delay is normalized to the task duration (T_d / 28 s). Nothing in this repository makes a claim about Mars light time in seconds.

## Contents

```
delay_driven_supervisor.py   simulation, calibration, evaluation, and figures
requirements.txt             Python dependencies
LICENSE                      MIT
README.md                    this file
```

Running the script creates a `results/` folder with all tables, raw per-run results, summaries, and figures.

## What the code does

**Plant.** A planar two-link revolute arm with rigid-body dynamics, integrated with semi-implicit Euler at 2 ms. The controller runs at 10 ms. Encoder and velocity measurements carry Gaussian noise. In each run, the mass in the controller's gravity-compensation model carries a random error (10% standard deviation, clipped to 25%).

**Task.** A four-waypoint Cartesian path executed with minimum-jerk segments over 28 s inside a 30 s mission window. The true goal adds two unplanned joint-space retargets of 0.30 rad, each smoothed over 4 s, which the stored plan cannot anticipate. Performance is the RMSE of the end-effector's Cartesian distance to the forward kinematics of the true goal. A run succeeds if the mean final error over the last second is below 0.02 m.

**Operator and channel.** The operator observes the robot state delayed by the downlink delay and commands the true goal plus a correction (gain 0.3) on the stale error. Uplink and downlink delays are T_d(1 + 0.10 ε) per packet, with ε a standard normal draw, floored at zero. The controller uses the newest packet that has arrived.

**Controllers.** All use the same saturated joint PD law with gravity compensation and differ only in the reference.

| Mode | Reference |
|------|-----------|
| `classical` | Most recently received command and its velocity |
| `predictive` | That command extrapolated by the packet age at constant velocity (Eq. 3) |
| `autonomous` | Stored plan plus the operator's latest retarget as an offset (Eq. 4) |

Before any packet arrives, `autonomous` tracks the plan alone, while `classical` and `predictive` hold the initial pose.

**Supervisors.**

| Mode | Decision variable |
|------|-------------------|
| `supervisor` | Exponentially weighted delay estimate from packet timestamps (Eq. 5, λ = 0.10) |
| `supervisor_age` | `d_eff = max(d_hat, packet age)` (Eq. 6) |
| `supervisor_oracle` | The true instantaneous delay |
| `supervisor_hand` | Hand-picked thresholds: T_cross = 1 s, band 0.9 to 1.1 s, hand-off at 10 s |
| `supervisor_nohyst` | No band, no exit margin, no dwell |

The switching logic is: from any mode, switch to autonomous when the decision delay reaches T_A. From autonomous, stay while it exceeds T_A,exit, then go to classical if it exceeds T_high and to predictive otherwise. From predictive, switch to classical above T_high. From classical, switch to predictive below T_low. A minimum dwell of 2 s separates switches. If T_A < T_high, the classical-only band is empty and the supervisor moves only between predictive and autonomous.

**Threshold rule.** On a 21-point calibration grid (0 to 22 s) with 30 seeds and no delay prior, the script computes the mean RMSE of each fixed controller. T_cross is the first persistent sign change of RMSE_P − RMSE_C, and T_A is the first persistent sign change of RMSE_A − min(RMSE_C, RMSE_P). A new sign is persistent when it holds for two further nonzero samples, and exact zeros (at T = 0, where C and P coincide) are skipped. Crossings are interpolated linearly between grid points. A bootstrap over seeds (500 resamples) gives 95% confidence intervals. The hysteresis band is T_low = min(CI_lo, 0.95 T_cross) and T_high = max(CI_hi, 1.05 T_cross), and the autonomy exit threshold is the lower bound of the interval for T_A. If a threshold does not exist, as when no retargets occur and autonomy is always better than the best continuous controller, the script reports that and continues.

**Evaluation.** Held-out seeds (100 to 129) and held-out delays are disjoint from calibration. The best fixed controller at each delay is the one with the lowest mean RMSE on the held-out seeds, which is conservative for the supervisor. Relative regret is 100 · mean(RMSE_sup − RMSE_best) / mean(RMSE_best), with a paired 95% bootstrap interval (1000 resamples). The equivalence criterion, fixed in advance, is that the upper end of the interval is at most 5%. Wilcoxon signed-rank tests with Holm correction are reported as a secondary difference test.

## Installation

Python 3.9 or later is recommended.

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Check that every code path executes (one seed, a few seconds to a minute):

```bash
python delay_driven_supervisor.py --smoke
```

Run the full pipeline:

```bash
python delay_driven_supervisor.py
python delay_driven_supervisor.py --processes 8     # limit the number of workers
```

The full pipeline runs, in order: sanity checks, threshold calibration, the prior-error sweep (which includes the primary 10% prior), the supervisor-variant table, the time-varying delay profiles, the chattering test, the scenario-family sweep, and the plant-sensitivity sweep. It involves tens of thousands of simulations spread over `multiprocessing.Pool`, so running it on a multi-core machine is advisable.

To run a single simulation from Python:

```python
from delay_driven_supervisor import simulate

th = dict(cross=4.64, low=4.11, high=5.55, auto=3.19, auto_exit=2.98)  # seconds
result = simulate("supervisor_age", 12.0, 100, th, prior_err=0.10)
print(result["rmse"], result["switches"], result["t_first_auto"])
```

## Output files (`results/`)

| File | Contents |
|------|----------|
| `raw_calibration_<label>.csv`, `table_calibration_<label>.csv` | Per-run and mean RMSE for the three fixed controllers |
| `thresholds_<label>.csv` | T_cross, T_low, T_high, T_A, T_A,exit |
| `fig_calibration_<label>.pdf` | Calibration curves with threshold markers |
| `raw_heldout_<label>.csv`, `table_heldout_<label>.csv` | Held-out per-run results and the regret table |
| `summary_heldout_<label>.json` | Equivalence counts, mean regret, first hand-off time, peak error after a switch |
| `fig_heldout_<label>.pdf` | Held-out RMSE against the best fixed controller |
| `table_variants.csv` | Supervisor variants at the primary prior |
| `table_prior_sweep.csv` | Effect of the prior on the initial delay |
| `table_timevarying.csv`, `fig_timevarying_<profile>.pdf` | Ramp, step, and sinusoid profiles, with traces |
| `table_chattering.csv` | Constant delay at T_A, with and without hysteresis |
| `table_scenario_families.csv` | Thresholds and own-threshold regret for five event families |
| `table_plant_sensitivity.csv` | Thresholds and regret under mass and gain perturbations |
| `config.json` | Parameters, thresholds, and library versions |

The `<label>` for the primary held-out condition is `prior_0.1`, and the others are `prior_none`, `prior_0.0`, and `prior_0.5`.

## Correspondence with the paper

| Paper element | Function in the script | Output |
|---------------|------------------------|--------|
| Measured thresholds (Section VI-A, Fig. 2) | `derive_thresholds` | `thresholds_nominal.csv`, `fig_calibration_nominal.pdf` |
| Held-out regret (Section VI-B, Fig. 3, Table II) | `heldout` | `table_heldout_prior_0.1.csv` |
| Supervisor variants (Table III) | `variants_table` | `table_variants.csv` |
| Initial delay knowledge (Table IV) | `prior_sweep` | `table_prior_sweep.csv` |
| Time-varying delay (Table V, Fig. 4) | `time_varying` | `table_timevarying.csv`, `fig_timevarying_step.pdf` |
| Chattering test (Section VI-F) | `chattering_test` | `table_chattering.csv` |
| Event families and plant changes (Section VI-G, Table VI) | `scenario_family_sweep`, `plant_sensitivity` | `table_scenario_families.csv`, `table_plant_sensitivity.csv` |

Reference values reported in the manuscript, for checking a run: T_cross = 4.64 s (95% CI 4.11 to 5.55) and T_A = 3.19 s (95% CI 2.98 to 3.41), with a mean relative regret of 0.16% at the 10% prior and the equivalence criterion met at 13 of 14 delays.

## Reproducibility

Calibration uses seeds 0 to 29 and held-out evaluation uses seeds 100 to 129. All modes share common random numbers for a given seed and delay, so differences between modes come from the control logic rather than from noise. Every parameter, seed range, and delay grid is defined at the top of the script and written to `config.json` together with the Python and library versions. Results are deterministic for a fixed library version, but small floating-point differences across platforms and NumPy versions are possible.

Design choices that are fixed rather than tuned, and were not swept in the paper: the 2 s dwell time, the EWMA constant of 0.10, the ±5% floor on the T_cross band, and the 5% equivalence margin. In the hand-picked-threshold variant, the autonomy exit threshold is set to 9.5 s, and the controller-gain perturbation scales both Kp and Kd.

## Citation

If you use this code, please cite the paper. The reference below will be updated once the paper is published.

```bibtex
@unpublished{narendra2026measured,
  title  = {Measured-Threshold, Delay-Driven Supervisory Control for Teleoperation under Normalized Delay: A Simulation Study on a Nonlinear Two-Link Manipulator},
  author = {Narendra, Nandana Mihira and K, Giri Pranesh and Anbarasi, M P},
  year   = {2026},
  note   = {Under review. Preprint: <arXiv link>. Code archive: <Zenodo DOI>}
}
```

## Limitations

The study is simulation-only, on a planar two-link arm without contact, with a simple operator model and no packet loss or bandwidth limits. Delay measurement assumes timestamps and synchronized clocks, and the autonomous mode assumes a valid stored plan. The switched system has no stability proof, and torque saturation is the only safety mechanism modeled. Predictive control here is constant-velocity extrapolation, not model-based prediction. The age-aware supervisor was designed after observing the step-profile failure of the base supervisor, so the step result demonstrates the mechanism and does not independently validate it. See the paper for the full discussion.

## License

Released under the MIT License. See `LICENSE`.

## Contact

Corresponding author: [name and email, matching the manuscript].
