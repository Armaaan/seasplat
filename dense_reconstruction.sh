#!/bin/bash
#SBATCH --job-name=eiffel_dense
#SBATCH --output=logs/eiffel_dense_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --time=12:00:00
#SBATCH --partition=acltr

set -euo pipefail

module purge
module load Anaconda3
module load CUDA/12.1.1

set +u
eval "$(conda shell.bash hook)"
conda activate seasplat
set -u

WORKSPACE="/home/arua/projects/3dgs/datasets/prepared/Eiffel_Tower/subset_50/undistorted"

echo "--- Step 1: Patch Match Stereo ---"
colmap patch_match_stereo \
  --workspace_path "$WORKSPACE" \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true \
  --PatchMatchStereo.max_image_size 2000

echo "--- Step 2: Stereo Fusion ---"
colmap stereo_fusion \
  --workspace_path "$WORKSPACE" \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path "$WORKSPACE/fused.ply"

echo "--- Done ---"