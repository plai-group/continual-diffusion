#!/bin/bash
#SBATCH --partition=plai
#SBATCH --job-name=vdt-act-gen-69
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0
#SBATCH --mem=32gb
#SBATCH --output=logs/train/slurm-%j.out
#SBATCH --error=logs/train/slurm-%j.err
#SBATCH --exclude=plai03,plai07,plai08,plai09,plai10
# plai03: all three issue-38 runs landed here and hung (plaicraft-debug#38).
# plai07-10: broken/degrading GPUs, carried over from model-pi0's train scripts.
#
# Issue #59 -- action-conditioned VDT (plai-group/plaicraft-debug#59).
# PLAIN CONDITIONING variant: action_dropout_prob=0, cfg_scale=1 (no guidance).
# The CFG variant is scripts/train_vdt_action_cfg.sh; the ONLY differences
# between the two files are ACTION_DROPOUT_PROB and CFG_SCALE.
#
# Trained FROM SCRATCH, not fine-tuned from #58's checkpoint, so the loss curve
# is directly comparable to #58's own from-scratch curve. Everything except the
# conditioning is held identical to #58.
#
# No-arg launch from repo root:   sbatch ./scripts/train_vdt_action.sh
# Edit the exports below to change a run; never pass overrides on the sbatch
# command line. The record of what a run used is the wandb config at start.
set -euo pipefail

REPO=/ubc/cs/research/plai-scratch/ctardy/projects/continual-diffusion.worktrees/issue-69
CORPUS_SRC=/ubc/cs/research/plai-scratch/ctardy/projects/plaicraft-data-preprocessing/processed/vdt_corpus/debug_24x40
VAL_ROOT=/ubc/cs/research/plai-scratch/ctardy/projects/plaicraft-data-preprocessing/processed/debug_v2
VAL_DB=/ubc/cs/research/plai-scratch/ctardy/projects/plaicraft-model-pi0/data/debug_v2/validation_debug_static_40x24.db
VAL_OUT=$REPO/results/action_gen_validation

SIF=/ubc/cs/research/ubc_ml/plaicraft/containers/plaicraft_ubuntu2404.sif
OVERLAY=/ubc/cs/research/plai-scratch/ctardy/envs/cd_overlay
MPILIB=/ubc/cs/research/plai-scratch/ctardy/envs/mpiroot/lib

# ---- run config: EVERY argument passed to the trainer is named here ---------
RUN_NAME="69 - vdt_action_gen"

# -- action conditioning (the ONLY axis that differs from #58)
ACTION_DIM=10              # 6 keys [w,a,s,d,space,shift] + 2 clicks [l,r]
                           # + symlog(dx), symlog(dy). Causal: row i holds the
                           # action from window [i-1,i), i.e. the one that
                           # CAUSED frame i. Verified by correlating mouse dx
                           # against measured frame-to-frame image shift.
ACTION_DROPOUT_PROB=0.0    # 0 = plain conditioning. The CFG variant uses 0.1.
CFG_SCALE=1.0              # 1.0 = no guidance at validation sampling.

# -- data
DATASET=debug_toy          # registered in improved_diffusion/video_datasets.py;
                           # its path is overridden by DEBUG_TOY_ROOT below.
NUM_WORKERS=4

# -- architecture (T and patch size are baked into the VDT weights)
MODEL_NAME=VDT-SM          # SM/M/L set num_classes=0; VDT-S would enable a
                           # phantom 10% label-dropout on a signal-free label.
DIFFUSION_SPACE=pixel      # in_channels=3; encode/decode are identity.
PATCH_SIZE=4               # 24x40 / 4 -> 6x10 grid = 60 tokens/frame.
T=20                       # must equal MAX_FRAMES (video_train_vdt.py:76).
MAX_FRAMES=20
MASKING_MODE=autoregressive  # fixed first-half observed / second-half latent,
                             # frames in order. Also REQUIRED by the action
                             # plumbing: train_util asserts this mode's frame
                             # selection is the identity before gathering
                             # actions, so flexible/autoflex would raise rather
                             # than silently mis-pair actions with frames.
LEARN_SIGMA=False          # MUST stay false: p_mean_variance
                           # (gaussian_diffusion.py:267) does `B, C = x.shape[:2]`,
                           # which on VDT's 5-D (B,T,C,H,W) binds C=T and asserts
                           # the wrong shape. Trains fine, then dies at the first
                           # heun_sample -- i.e. at the first sample interval.

# -- optimisation
BATCH_SIZE=6               # measured on 2080 Ti (11GiB): 4->5.2, 6->7.9, 8->OOM.
                           # --microbatch is inert (train_util.py:376 HACK).
LR=1e-4
CLIP_GRAD=1.0
USE_FP16=False             # issue-39 runs went NaN on fp16; 114.7M fits fp32.
EMA_RATE=0.999,0.9999      # index 0 drives log_samples; 0.9999 alone is a
                           # ~6900-step half-life = near-init blur for ages.

# -- logging / eval cadence
SAMPLE_INTERVAL=2500       # drives the built-in video-0 sample, the 8 issue-22
                           # validation overlays + val/video/* metrics, and the
                           # val/swap/* action-swap test.
SAVE_INTERVAL=20000
PER_TASK_SCALARS=False     # True adds 8 tasks x 6 metrics = 48 wandb channels.
# -----------------------------------------------------------------------------

cd "$REPO"
mkdir -p logs/train checkpoints results

set +u; source /ubc/cs/research/plai-scratch/ctardy/projects/plaicraft-model-pi0/.env 2>/dev/null || true; set -u
export WANDB_ENTITY="${WANDB_ENTITY:-plaicraft-academic}"
export WANDB_PROJECT="${WANDB_PROJECT:-debug-experiments}"
export WANDB_NAME="$RUN_NAME"

# Build the per-session action caches in the SOURCE corpus first, so the
# staging copy below picks them up and every job does not rebuild 200 of them.
# load_or_build is idempotent and atomic across workers.
echo "pre-building action caches in $CORPUS_SRC"
singularity exec -B /ubc/cs/research/plai-scratch \
  --env PYTHONPATH="$OVERLAY:$REPO" --env LD_LIBRARY_PATH="$MPILIB" \
  "$SIF" /opt/venv/bin/python -c "
import glob, os
from improved_diffusion.debug_actions import load_or_build
d = sorted(glob.glob(os.path.join('$CORPUS_SRC', '*')))
d = [p for p in d if os.path.isdir(p)]
for i, p in enumerate(d):
    load_or_build(p)
print(f'action caches ready for {len(d)} sessions')
"

# Stage the corpus into RAM (plaicraft-debug#43): CPU/RAM are reserved by the
# job anyway, and NFS reads would starve the dataloader.
if [ -w /dev/shm ]; then TMPDIR=/dev/shm/$USER/$SLURM_JOB_ID
elif [ -d /scratch-ssd ]; then TMPDIR=/scratch-ssd/$USER/$SLURM_JOB_ID
else TMPDIR=${SLURM_TMPDIR:-/tmp}/$USER/$SLURM_JOB_ID; fi
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT TERM
# Neither /dev/shm nor /scratch-ssd is slurm-managed; sweep anything stale.
find /dev/shm/$USER -mindepth 1 -maxdepth 1 -mmin +2880 -exec rm -rf {} + 2>/dev/null || true

echo "staging corpus -> $TMPDIR/corpus"
cp -r "$CORPUS_SRC" "$TMPDIR/corpus"
export DEBUG_TOY_ROOT="$TMPDIR/corpus"
echo "staged $(find "$DEBUG_TOY_ROOT" -name '*.hdf5' | wc -l) sessions, $(find "$DEBUG_TOY_ROOT" -name 'actions_10d.npy' | wc -l) action caches"

export WANDB_DIR="$TMPDIR/wandb"; mkdir -p "$WANDB_DIR"

singularity exec --nv \
  -B /ubc/cs/research/plai-scratch -B "$TMPDIR" \
  --env PYTHONPATH="$OVERLAY:$REPO" \
  --env LD_LIBRARY_PATH="$MPILIB" \
  --env DEBUG_TOY_ROOT="$DEBUG_TOY_ROOT" \
  --env WANDB_ENTITY="$WANDB_ENTITY" \
  --env WANDB_PROJECT="$WANDB_PROJECT" \
  --env WANDB_NAME="$WANDB_NAME" \
  --env WANDB_DIR="$WANDB_DIR" \
  --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
  "$SIF" /opt/venv/bin/python scripts/video_train_vdt.py \
    --dataset "$DATASET" \
    --model_name "$MODEL_NAME" \
    --diffusion_space "$DIFFUSION_SPACE" \
    --patch_size "$PATCH_SIZE" \
    --T "$T" \
    --max_frames "$MAX_FRAMES" \
    --masking_mode "$MASKING_MODE" \
    --learn_sigma "$LEARN_SIGMA" \
    --action_dim "$ACTION_DIM" 
    --generate_actions True 
    --action_loss_weight 1.0 \
    --action_dropout_prob "$ACTION_DROPOUT_PROB" \
    --cfg_scale "$CFG_SCALE" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --clip_grad "$CLIP_GRAD" \
    --ema_rate "$EMA_RATE" \
    --use_fp16 "$USE_FP16" \
    --sample_interval "$SAMPLE_INTERVAL" \
    --save_interval "$SAVE_INTERVAL" \
    --num_workers "$NUM_WORKERS" \
    --debug_validation_db "$VAL_DB" \
    --debug_validation_root "$VAL_ROOT" \
    --debug_validation_out "$VAL_OUT" \
    --debug_validation_per_task "$PER_TASK_SCALARS"
