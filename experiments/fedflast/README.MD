# FedFLaST experiment entry point

Run commands from this directory. The complete release instructions and default Module 1 + Module 2 command are in the repository-level `README.md`.

General syntax:

```bash
CUDA_VISIBLE_DEVICES=<gpu-list> \
bash run_fedflast_distributed_pytorch.sh \
  <model> <dataset> <total-clients> <clients-per-round> \
  <communication-rounds> <local-epochs> <target-density> <learning-rate> \
  [additional options]
```

FedFLaST requires the following two switches for the full method:

```text
--use_fedflast_perturbation true
--use_flast_client_sam_prune true
```

Without them, the entry point follows the FedDST-compatible fallback behavior.

