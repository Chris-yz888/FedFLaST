# FedFLaST

Code release for **FedFLaST: Improving Generalization in Federated Dynamic Sparse Training via Flatness-Aware Topology Adaptation**.

FedFLaST adds server-side global perturbation and client-side SAM-guided pruning to federated dynamic sparse training. The implementation uses the following pruning score:

```text
s0_i = |w_i * g0_i|
s+_i = |(w_i + epsilon_i) * g+_i|
S_i  = max(s0_i, s+_i)
```

At a mask-adjustment round, the client computes `g0` on a minibatch, perturbs currently active weights with `epsilon`, then computes `g+` on the same minibatch. Among the `Bottom-(2k)` active weights by magnitude, the `k` connections with the smallest `S_i` are pruned. Growth keeps the original FedDST gradient-based rule. The SAM probe affects the pruning decision; it is not applied as a local optimizer update.

## Contents

- `api/`, `core/`, `experiments/`: source code and baseline implementations.
- `scripts/eval_robustness.py`: CIFAR-10-C checkpoint evaluator.
- `requirements.txt`: Python dependencies.

This repository contains source code only: no datasets, pretrained weights, logs, or experiment records are included.

## Environment

The reported experiments used Python 3.10, PyTorch 2.7.1 with CUDA 11.8, and `mpi4py` 4.0.3. A working MPI installation is required (the original environment used MPICH/HYDRA).

```bash
conda create -n fedflast python=3.10 -y
conda activate fedflast
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build compatible with your driver if the default pip package is unsuitable. MPI must be installed and available to `mpi4py` and `mpirun`.

## Data

No data is bundled. From the repository root, download CIFAR-10 with torchvision:

```bash
python - <<'PY'
from torchvision.datasets import CIFAR10
CIFAR10("./data/cifar10", train=True, download=True)
CIFAR10("./data/cifar10", train=False, download=True)
PY
```

The CIFAR-10 loader expects the torchvision dataset under `data/cifar10`. Other datasets must be prepared separately according to their loaders. CIFAR-10-C evaluation requires the extracted standard directory containing `labels.npy` and one `.npy` file for each corruption.

## Common training recipe

The reference launch script uses the group's commonly used CIFAR-10/ResNet18 settings: 100 clients, 10 clients per round, 500 communication rounds, 5 local epochs, batch size 64, Adam with learning rate `0.001` and weight decay `1e-3`, Dirichlet partition `alpha=0.5`, mask period `10`, `T_end=300`, and adjustment alpha `0.2`. The example uses target density `0.1`, seed `0`, and enables both FedFLaST modules.

```bash
cd experiments/fedflast
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_cifar10_resnet18_reference.sh
```

To change the density or seed, pass them as optional arguments:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_cifar10_resnet18_reference.sh 0.2 1
```

The script writes the final checkpoint under `weights/` and run logs under `runs/`; these outputs are git-ignored. It expects four visible GPUs according to the default mapping in `experiments/fedflast/gpu_mapping.yaml`. Change the GPU mapping file if using a different topology.

The equivalent direct command is:

```bash
cd experiments/fedflast
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash run_fedflast_distributed_pytorch.sh \
  resnet18 cifar10 100 10 500 5 0.1 0.001 \
  --batch_size 64 \
  --partition_alpha 0.5 \
  --delta_T 10 \
  --T_end 300 \
  --adjust_alpha 0.2 \
  --num_eval 128 \
  --frequency_of_the_test 10 \
  --client_optimizer adam \
  --wd 1e-3 \
  --seed 0 \
  --save_final_checkpoint true \
  --checkpoint_path ../../weights/fedflast_cifar10_resnet18_density010_seed0.pt \
  --use_fedflast_perturbation true \
  --flast_rho 0.05 \
  --flast_momentum 0.05 \
  --flast_server_lr 1.0 \
  --use_flast_client_sam_prune true \
  --flast_local_rho 0.05 \
  --flast_candidate_multiplier 2.0 \
  --flast_local_sam_norm l2
```

`run_fedflast_distributed_pytorch.sh` launches MPI ranks and creates `mpi_host_file` in the current experiment directory. The default GPU mapping allocates one server process and ten client processes over four visible GPUs.

## Evaluate a trained checkpoint on CIFAR-10-C

After preparing CIFAR-10-C, run from the repository root:

```bash
python scripts/eval_robustness.py \
  --checkpoint FedFLaST:weights/fedflast_cifar10_resnet18_density010_seed0.pt \
  --data_dir ./data/cifar10-c/CIFAR-10-C \
  --output_dir ./results/pretrained_cifar10c
```

The evaluator reports accuracy and cross-entropy over the standard 15 corruptions and five severity levels. It does not compute normalized mCE.

## Clone

```bash
git clone https://github.com/Chris-yz888/FedFLaST.git
cd FedFLaST
```

## License

No license has been selected for this repository yet.
