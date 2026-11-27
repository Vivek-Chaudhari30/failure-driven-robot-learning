# Environment Setup

> Facts below were verified against official docs on **2026-11-26** (sources linked). Versions
> move fast: re-verify before installing. Nothing here should be installed on shared
> infrastructure without the cluster admins' and your advisor's agreement.

## 0. Current status

| Machine | Status | Implication |
|---------|--------|-------------|
| Dev laptop | Inspected: Apple M5 Pro (arm64), macOS, 24 GB RAM, ~190 GB free, no NVIDIA GPU. Python 3.11 (python.org) and 3.12 (Anaconda), `uv` and `docker` available. | **Isaac Sim / Isaac Lab cannot run here** (no macOS support). Use the laptop for sim-free code (training, diagnosis, analysis, tests), docs, and a MuJoCo/robosuite learning sandbox. |
| Lab GPU workstations | **Inspected 2026-11-26** (kyoto): RTX A6000 / RTX 6000 Ada, driver 580.178.04, Ubuntu 24.04, Python 3.12, no SLURM, no containers | Isaac-class GPUs; driver below Isaac Lab 3.0's stated minimum (untested). Plain `uv` venv over ssh. See §6. |

## 1. Verified requirements (as of 2026-11-26)

### Isaac Lab 3.0.0 (released 2026-10-20), [release notes](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)
- Targets **Isaac Sim 6.2**, **Python 3.12**, PyTorch 2.11 + CUDA 13.0, Warp 1.17, Newton 1.6.1.
- OS: Ubuntu 22.04+ x86_64, Linux aarch64 (incl. DGX Spark), Windows 11. **No macOS.**
  ([installation](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html))
- GPU (Isaac Lab itself): "A CUDA-capable NVIDIA GPU supported by the selected Newton solver and
  PyTorch"; the CUDA 13 stack needs compute capability ≥ 7.5 (Turing+).
- "Kit-less" mode: most workflows can run on the Newton backend *without* Isaac Sim.
- Breaking changes vs 2.x: task IDs renamed (lift: `IsaacContrib-Lift-Cube-Franka`,
  `-IK-Abs`, `-IK-Rel`; new core task `Isaac-Lift-Franka`, Newton by default). Quaternion order
  WXYZ → **XYZW**. See the [migration guide](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/migrating.html).
- Isaac Lab Mimic: **deprecated** in favour of AutoData. Linux-only. Requires cuRobo.

### Isaac Sim 6.2, [requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)
| | Minimum | Good | Ideal |
|---|---|---|---|
| OS | Ubuntu 22.04 (deprecated) / 24.04, Windows 11 | | |
| CPU / RAM | 4 cores / 32 GB | 8 / 64 GB | 16 / 64 GB |
| GPU / VRAM | RTX 4080 / 16 GB | RTX 5080 / 16 GB | RTX PRO 6000 Blackwell / 48 GB |
| Disk | 50 GB SSD | 500 GB SSD | 1 TB NVMe |
| Driver (Linux) | **595.58.03** | | |

- Exact quote: **"GPUs without RT Cores (A100, H100) are not supported."** The docs do *not*
  say headless mode lifts this (unverified, so assume it doesn't).
- Python 3.12; Linux needs **GLIBC ≥ 2.35** (manylinux_2_35)
  ([pip install](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_python.html)).
- The Isaac Sim container is Linux-only.

### Fallbacks
- **ManiSkill3 v3.0.1** (2026-04-21): GPU simulation on Linux + NVIDIA; macOS supports CPU sim
  and rendering only; rendering needs Vulkan (A100 needs an extra `nvidia_layers.json` for
  rendering, which state-only sim doesn't need). PickCube-v1 with Panda; motion-planning demo
  generation included. Apache-2.0.
  ([install](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html))
- **robosuite v1.5.2** (MuJoCo; officially supports macOS and Linux; Lift + Panda). MIT.
  **robomimic** v0.5.0: install from GitHub, because PyPI is stale at 0.3.0.

## 2. Phase A procedure: inspect before installing

1. Copy the repo (or just `scripts/check_env.sh`) to the cluster.
2. Run it on a **GPU compute node**, since login nodes often have no GPU:
   ```bash
   srun --gres=gpu:1 --time=00:10:00 bash scripts/check_env.sh
   ```
   (Adapt to the scheduler. The script installs nothing and only writes `env_report_*.txt`.)
3. Also find out from the admins or your advisor (the script can't tell):
   - the GPU-hour allocation and max job length
   - whether **Apptainer/Singularity** is allowed
   - whether compute nodes have internet access (the Isaac Lab cluster guide assumes they do)
   - scratch-space quotas and purge policies
   - whether Docker image builds must happen elsewhere
4. Review the report (it contains hostnames and paths) and add a summary to §0.

## 3. Decision table (fill in from the report)

| Cluster finding | Decision |
|-----------------|----------|
| RTX-class GPU with ≥ 16 GB (e.g., RTX 4080/4090/5080/5090, RTX 6000 Ada, RTX PRO 6000, L40/L40S ⚠ check the list), driver ≥ 595.58.03, GLIBC ≥ 2.35, Apptainer available | **Path 1: Isaac Lab 3.0.0 + Isaac Sim 6.2 in an Apptainer container** (PhysX lift task) |
| RTX GPU but driver < 595.58.03 (cannot upgrade) | Ask admins about a driver upgrade. Otherwise consider Isaac Lab 2.3.2 + Isaac Sim 5.1 (⚠ its driver requirement is not verified here; check the 5.1 requirements page), or Path 2/3 |
| Only A100/H100/V100-class GPUs | **Path 2: spike Isaac Lab 3.0 kit-less (Newton).** Time-box it to 3 days. It must support: per-env friction/mass randomization, finger–cube contact forces, and a differential-IK action. If any is missing → **Path 3** |
| No usable GPU, or no containers and no way to meet GLIBC/Python | **Path 3: ManiSkill3** (GPU if any NVIDIA GPU works; else CPU), or robosuite on CPU nodes |

Record the decision and its evidence in a new section §6 of this file. It also feeds
RESEARCH §6 (risk table).

## 4. Path 1: Isaac Lab 3.0.0 on a SLURM cluster (outline, to be verified in M1)

The Isaac Lab HPC guide ([hpc_clusters](https://isaac-sim.github.io/IsaacLab/main/source/features/hpc_clusters.html))
describes this workflow: **build a Docker image → convert to an Apptainer/Singularity image →
push to the cluster → submit jobs** with `docker/cluster/cluster_interface.sh push|job`.
SLURM and PBS are supported. The official image tag is `nvcr.io/nvidia/isaac-lab:3.0.0`
(there is no `latest` tag).

```bash
# On a machine with Docker + NVIDIA runtime (NOT the Mac: the image is x86_64 Linux + GPU)
git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab && git checkout v3.0.0
# follow docs/cluster instructions to build the image and convert to .sif
# then, on the cluster, smoke test headless:
apptainer exec --nv isaac-lab_3.0.0.sif \
  python scripts/environments/zero_agent.py --task IsaacContrib-Lift-Cube-Franka-IK-Rel --num_envs 64 --headless
```

⚠ Script paths and flags above follow 2.x conventions and **must be checked against the v3.0.0
docs** in M1. If you can pull directly on the cluster
(`apptainer pull docker://nvcr.io/nvidia/isaac-lab:3.0.0`, which may need NGC credentials), you
can skip the local Docker build.

**Pinning:** record the exact image digest, Isaac Lab git tag, Isaac Sim version, driver version
and GPU model in every run's metadata (ARCHITECTURE §7). Do not upgrade mid-study.

## 5. Project environment

Our own package (`src/fdrl`) is split so that **only `fdrl.sim` needs the simulator**:

| Context | Environment | Contains |
|---------|-------------|----------|
| Mac / CPU nodes (sim-free) | `uv venv --python 3.12` + `uv pip install -e ".[dev]"` | torch (CPU/MPS), numpy, h5py, pyarrow, omegaconf, scipy, rliable, matplotlib, pytest |
| Cluster sim jobs | inside the Isaac Lab container: `pip install -e . --no-deps` + the few extra deps | Isaac Lab's Python 3.12 + torch; our package on top |
| Mac learning sandbox (optional) | separate venv: `robosuite` (+ `mjpython` for the viewer) | For learning only, not for reported results |

`pyproject.toml` exists with lower-bound pins only. Tighten the pins in M1 to match what is
actually installed on the GPU machines, and record them in §6.

## 6. Decision record

### 6.1 What the lab machines are (inspected 2026-11-26, `scripts/check_env.sh` on `kyoto`)

The lab GPUs are **shared Ubuntu workstations reached over ssh**, not a SLURM cluster.

| Item | Finding | Consequence |
|------|---------|-------------|
| Machines | hokkaido, tokyo, umibozu, kyoto, saitama (2 GPUs), bippu, osaka, fukuyama, kawasaki, tottori | Pick a free one with `gpus` before every run |
| GPUs | RTX A6000 48 GB (Ampere, RT cores); fukuyama/kawasaki RTX 6000 Ada 48 GB | RTX-class with RT cores: Isaac Sim-class GPUs. Not on NVIDIA's tested list, but not excluded either (only GPUs without RT cores are) |
| Driver | **580.178.04** (CUDA 13.0) | **Below** Isaac Lab 3.0's stated 595.58.03 minimum for its default CUDA 13 stack. Untested whether it works anyway |
| OS / Python | Ubuntu 24.04.4, system Python 3.12.3 | Matches Isaac Lab 3.0 (Python 3.12) |
| Scheduler | none (no `sinfo`) | No `srun`; run in `tmux`, pin a GPU with `CUDA_VISIBLE_DEVICES` |
| Containers | none (no docker / podman / apptainer) | §4 (Apptainer) does not apply; use a plain `uv` virtualenv |
| CPU / RAM | 22 cores, 125 GB (kyoto) | Fine |
| Disk | local `/` 1.8 TB, 90% full (187 GB free); NFS home on a 104 TB share | Put the virtualenv on local disk (Isaac Sim has thousands of small files, slow on NFS); keep `runs/` on NFS or local disk as space allows |
| Network | pypi, pypi.nvidia.com, github reachable; nvcr.io needs login | pip installs work |
| Health | kyoto GPU0 reports "Unknown Error" in `nvidia-smi`; only GPU1 usable | Report to admins. The `gpus` helper shows one line per machine and hides this |

### 6.2 Chosen path

**Try Isaac Lab 3.0 (pip/uv install, no container) on the current driver first.** If it fails
because of the driver, either (a) ask the admins for a 595-series driver, or (b) fall back to
Isaac Lab 2.3.2 + Isaac Sim 5.1, whose docs recommend driver ≥ 580.65.06 (met) but need
Python 3.11 (installable with `uv` without root) and list Ubuntu 22.04. Isaac Sim 5.1 is no longer
supported by NVIDIA. If both fail: ManiSkill3 (Path 3).

Still to record here after M1: exact versions installed, measured throughput (episodes/min for
1,024 envs × 250 steps), peak VRAM, and the `scripts/isaac_smoke.py` result.

### 6.4 Isaac Lab 3.0 status (tottori, 2026-11-27)

Working machine: **tottori** (Ubuntu 24.04, driver 595.99, RTX A6000). Isaac Lab 3.0 installed with
`uv` into `/tmp/$USER/fdrl-venv`; do not set `CUDA_VISIBLE_DEVICES` (Isaac warns against it).
Machines on Ubuntu 20.04 (hokkaido, bippu, osaka: glibc 2.31, driver 550-555) cannot install
Isaac Lab 3.0. The 580-driver 24.04 machines are untested.

- `isaac_smoke.py`: cube pose, yaw and friction read back correctly; finger contacts ~5 N.
- Adapter fixes found on the real sim: Warp arrays from PhysX views, relative IK realises only
  ~10% of each Δ (actions now integrated into a target pose), descend hysteresis in the expert.
- Calibration (calibration seeds only, 10 episodes per cell): see the comments in
  `configs/env/isaac.yaml`. Expert success 0.34 / 0.93 / 0.93 by friction bin, infeasible at
  μ ≤ 0.25, mean accepted demo 216 steps (T = 400). The feasible bins are just below the 95%
  target, with a drop at μ = 1.0 (0.75, mostly F6 / F1). Not yet fixed.

### 6.3 Running on the lab GPUs (procedure)

```bash
ssh kyoto                                  # pick a FREE machine with `gpus` first
git clone <your private repo> ~/fdrl && cd ~/fdrl
export UV_CACHE_DIR=/tmp/$USER/uv-cache     # keep caches off NFS
uv venv --python 3.12 /tmp/$USER/fdrl-venv  # local disk; recreate if /tmp is cleaned
source /tmp/$USER/fdrl-venv/bin/activate
uv pip install -e ".[dev]"
python -m pytest -q                         # 51 sim-free tests; should all pass
python scripts/run_experiment.py --config configs/experiment/smoke.yaml   # toy backend, ~30 s
```

Then install Isaac Lab into the same environment following the **official** Isaac Lab 3.0
installation page (pip/uv path), and check it with the adapter smoke test:

```bash
tmux new -s fdrl
export CUDA_VISIBLE_DEVICES=1               # the free GPU on that machine
python scripts/isaac_smoke.py --config configs/env/isaac.yaml sim.num_envs=16
python scripts/calibrate.py --config configs/env/isaac.yaml --per-cell 50
```

`isaac_smoke.py` checks that the applied θ (cube pose, yaw, friction) reads back correctly and
that the scripted expert can lift. Every line marked `VERIFY` in `src/fdrl/sim/isaaclab_lift.py`
is an Isaac Lab API detail to confirm at this step.
