#!/usr/bin/env bash
# Read-only inspection of a (shared) GPU machine for Isaac Lab / ManiSkill feasibility.
# Installs NOTHING, writes only to the report file. Safe to run on a login or compute node.
#
# Usage:
#   bash scripts/check_env.sh                 # writes env_report_<host>_<date>.txt in CWD
#   srun --gres=gpu:1 --time=00:10:00 bash scripts/check_env.sh   # on a SLURM compute node
#
# Then share the report (review it first: it contains hostnames and paths).

set -u
OUT="env_report_$(hostname -s)_$(date +%Y%m%d_%H%M).txt"
exec > >(tee "$OUT") 2>&1

section() { printf '\n==== %s ====\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }
run() { echo "\$ $*"; "$@" 2>&1 || echo "(command failed: exit $?)"; }

section "Host / OS"
run hostname
run uname -a
[ -f /etc/os-release ] && run cat /etc/os-release
run ldd --version | head -1          # GLIBC version (Isaac Sim pip wheels need a recent glibc)
run nproc
run free -g

section "NVIDIA GPU / driver / CUDA"
if have nvidia-smi; then
  run nvidia-smi
  run nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
  echo
  echo "RT-core heuristic (Isaac Sim needs RTX-class GPUs; data-center parts like A100/H100/V100 lack RT cores):"
  nvidia-smi --query-gpu=name --format=csv,noheader | while read -r g; do
    case "$g" in
      *A100*|*H100*|*H200*|*V100*|*P100*|*A30*|*B100*|*B200*|*GH200*)
        echo "  $g -> likely NO RT cores: Isaac Sim UNSUPPORTED (ManiSkill3/MuJoCo fallback)";;
      *RTX*|*L40*|*L4*|*A40*|*A10*|*A6000*|*A5000*|*A4000*|*T4*|*Quadro*)
        echo "  $g -> has RT cores: check against Isaac Sim min spec (VRAM, generation)";;
      *) echo "  $g -> unknown: check manually";;
    esac
  done
else
  echo "nvidia-smi not found (no NVIDIA driver on this node; try a GPU compute node)"
fi
have nvcc && run nvcc --version
ls -d /usr/local/cuda* 2>/dev/null || echo "no /usr/local/cuda*"

section "Vulkan (needed by Isaac Sim and ManiSkill rendering, incl. headless)"
if have vulkaninfo; then run vulkaninfo --summary; else echo "vulkaninfo not found"; fi
ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null || echo "no Vulkan ICD dirs found"

section "Display / headless"
echo "DISPLAY=${DISPLAY:-<unset>}"
have Xvfb && echo "Xvfb available" || echo "Xvfb not found"

section "Python / package managers"
for p in python3 python3.10 python3.11 python3.12 conda mamba micromamba uv pip3; do
  have "$p" && echo "$p: $(command -v "$p") $("$p" --version 2>&1 | head -1)"
done
have module && { echo "\$ module avail (filtered)"; module avail 2>&1 | grep -iE 'cuda|python|anaconda|conda|apptainer|singularity|gcc' | head -40; }

section "Containers"
for c in docker podman apptainer singularity enroot; do
  have "$c" && echo "$c: $("$c" --version 2>&1 | head -1)"
done
have docker && { docker info 2>&1 | grep -iE 'runtimes|nvidia|server version' || true; }
groups

section "Scheduler"
if have sinfo; then
  run sinfo -o "%P %a %l %D %G %c %m"
  run sinfo -o "%G" --noheader | sort -u
  have sacctmgr && run sacctmgr -n show assoc user="$USER" format=account,partition,qos,maxwall,grptres
  have squeue && run squeue -u "$USER"
else
  echo "no SLURM (sinfo) found"
fi

section "Storage"
run df -h "$HOME"
for d in /scratch /scratch/"$USER" /work /work/"$USER" /projects /tmp "${TMPDIR:-}"; do
  [ -n "$d" ] && [ -d "$d" ] && run df -h "$d"
done
have quota && run quota -s
have lfs && run lfs quota -h -u "$USER" "$HOME"

section "Network egress (needed for pip / NGC container pulls)"
for u in https://pypi.org/simple/ https://pypi.nvidia.com https://nvcr.io/v2/ https://github.com; do
  code=$(curl -s -o /dev/null -m 8 -w '%{http_code}' "$u" 2>/dev/null || echo "fail")
  echo "$u -> $code"
done

section "Done"
echo "Report written to $OUT"
