#!/bin/bash
# Batch sweep on the cluster: QCG-PilotJob distributes run.py configurations across the Slurm
# allocation.  Usage (from this folder):  sbatch submit_sweep.sh
#SBATCH --job-name=qlauncher-sweep
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --time=01:00:00
#SBATCH --output=qlauncher-sweep-%j.out
# GPU (optional): request one and use a GPU Aer simulator as the backendv1v2 for the AQT-noise
# runs (AerSimulator(device='GPU')); the classical statevector optimisation stays on CPU/numpy.
## SBATCH --gres=gpu:1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs results

# load Python 3.11 (this cluster provides it as a module) and activate the QLauncher venv
module load Python/python-3.11.0 2>/dev/null || true
source "${QLAUNCHER_VENV:-$HOME/qlauncher_env}/bin/activate"

python -c "import qlauncher, qcg.pilotjob" 2>/dev/null || { echo "env: activate the venv with QLauncher installed"; exit 1; }

# QCG-PilotJob detects the Slurm allocation and schedules the pilot jobs within it.
python sweep_qcgpj.py

echo "[done] one JSON per configuration in results/"
