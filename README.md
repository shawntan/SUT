# Sparse Universal Transformers (SUT)

Implementation for WMT'14 En-De for SUT.

[Link to paper](https://arxiv.org/abs/2310.07096)

---

### Steps to run:
1. Clone and follow the preprocessing steps in [Transformer-Clinic](https://github.com/LiyuanLucasLiu/Transformer-Clinic/blob/master/nmt-experiments/iwslt14_de-en.md#preprocessing)
   
   ```sh
   export TRANSFORMER_CLINIC_ROOT=/path/to/Transformer-Clinic
   ```
2. Initialise evironment variables for experiment (base or big):
   ```sh
   source scripts/params_de_base.sh
   export EXP_NAME=sut_base
   # Submit job to cluster (32 compute nodes)
   sbatch slurm/launch.slurm
   ```
3. Generate and calculate BLEU:
   ```sh
   scripts/eval_wmt14ende.sh checkpoints/sut_base
   ```
   
