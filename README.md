# Multi-Fidelity Active Learning with GFlowNets

This repository contains code for the paper [Multi-Fidelity Active Learning with GFLowNets](http://arxiv.org/abs/2306.11715). 

## Abstract

In the last decades, the capacity to generate large amounts of data in science and
engineering applications has been growing steadily. Meanwhile, the progress in
machine learning has turned it into a suitable tool to process and utilise the available
data. Nonetheless, many relevant scientific and engineering problems present
challenges where current machine learning methods cannot yet efficiently leverage
the available data and resources. For example, in scientific discovery, we are often
faced with the problem of exploring very large, high-dimensional spaces, where
querying a high fidelity, black-box objective function is very expensive. Progress
in machine learning methods that can efficiently tackle such problems would help
accelerate currently crucial areas such as drug and materials discovery. In this paper,
we propose the use of GFlowNets for multi-fidelity active learning, where multiple
approximations of the black-box function are available at lower fidelity and cost.
GFlowNets are recently proposed methods for amortised probabilistic inference
that have proven efficient for exploring large, high-dimensional spaces and can
hence be practical in the multi-fidelity setting too. Here, we describe our algorithm
for multi-fidelity active learning with GFlowNets and evaluate its performance in
both well-studied synthetic tasks and practically relevant applications of molecular
discovery. Our results show that multi-fidelity active learning with GFlowNets
can efficiently leverage the availability of multiple oracles with different costs and
fidelities to accelerate scientific discovery and engineering design.

![Overview of the algorithm](https://github.com/nikita-0209/mf-al-gfn/blob/main/assets/figs/overview.png)

## Key Results 
We evaluate the proposed MF-GFN approach in both synthetic tasks (Branin, Hartmann) and benchmark tasks of practical relevance, such as DNA
aptamer generation, antimicrobial peptide design and molecular modelling (read section 4 of the paper). Through comparisons
with previously proposed methods as well as with variants of our method designed to understand the
contributions of different components, we conclude that multi-fidelity active learning with GFlowNets
not only outperforms its single-fidelity active learning counterpart in terms of cost effectiveness and
diversity of sampled candidates, but it also offers an advantage over other multi-fidelity methods due
to its ability to learn a stochastic policy to jointly sample objects and the fidelity of the oracle to be
used to evaluate them.

![Results on AMP, DNA and Molecules (IP)](https://github.com/nikita-0209/mf-al-gfn/blob/main/assets/figs/key_results.png)

## Installation

+ This code **requires** `python 3.10` and `CUDA 11.7`.
+ Clone the repository.
+ If you are installing this in a compute cluster (for example, Mila's), you can load the required modules by running `source ./prereq_cluster_cpu.sh`.
+ Setup is currently only supported on Ubuntu. It should also work on OSX, but you will need to handle the package dependencies.
+ The recommended installation is as follows:

```bash
python3.10 -m venv ~/venvs/mfgfn  # Initalize your virtual env.
source ~/envs/mfgfn/bin/activate  # Activate your environment.
./prereq_python.sh  # Updates pip and forces the installation of potentially problematic libraries
python -m pip install .[all]  # Install the remaining dependencies of this package.
```

Finally, consider renaming the log directory and data directory arguments (if necessary) in `config/user/anonymous.yaml` 

## Usage

The project uses [Hydra](https://hydra.cc/) for configuration and [Weights and Biases](https://docs.wandb.ai/) for logging.
For reproducing the results, configuration files with the default settings for experiments with the synthetic and benchmark tasks have been created in `config/`. Run
```
python activelearning.py --config-name=<config-filename>
```

### Default Config Options
Options `sf` and `mf` for the `<fid>` placeholder correspond to the single-fidelity and multi-fidelity variants respectively. 

- `<fid>_branin`: Branin 
- `<fid>_hartmann`: Hartmann 
- `<fid>_amp`: Antimicrobial Peptides
- `<fid>_aptamers`: DNA
- `<fid>_mols_ea`: Molecules with the objective of maximizing electron affinity 
- `<fid>_mols_ip`: Molecules with the objective of maximizing (negative) ionization potential  

### Additional Config Options
Below we list other configuration options. See the config files in `./config` for all configurable parameters. Note that any config field can be overridden from the command line, and some configurations are not supported.

**Environment Options**
- `grid` (for the synthetic tasks, Branin and Hartmann)
- `aptamers`  (for DNA)
- `amp` (for antimicrobial peptides)
- `mols` (for molecules represented as SELFIES strings)

**Oracle Options**

The oracles are prefixed by the environment (`branin`, `hartmann`, `amp`, `aptamers`, `mols_ea`, `mols_ip`) and indexed by increasing level of fidelity. 

**Sampler Options**
- `gflownet` (gflownet with the flowmatch objective)
- `trajectorybalance` (gflownet with the trajectory balance objective, recommended for reproducing results)
- `ppo` (RL baseline, proximal policy optimization algorithm)

**Proxy Options**

Options `sf` and `mf` for the `<fid>` placeholder correspond to the single-fidelity and multi-fidelity variants repectively.
- `<fid>_gp` (exact gaussian process for regression)
- `<fid>_svgp` (stochastic vartiation gaussian process for regression)
- `<fid>_dkl` (deep kernel regressor with backbone as one of the model options and index kernel for learning the fidelity parameter)
- `mf_dkl_linear` (multi-fidelity deep kernel regressor with backbone as one of the model options (below) and linear downsampling kernel for learning the fidelity parameter)

**Model Options** 
- `mlp`
- `transformer`
- `mlm_cnn` (masked language model based on CNN)
- `mlm_transformer` (masked language model based transformer)
- `regressive` (based on the [DNN-MFBO implementation](https://github.com/shib0li/DNN-MFBO))
- `mf_mlp` (MLP with an additional layer over the concatenated representation of the feature with fidelity)


**Acquisition Function Options**

Single and multi-fidelity variants of the Max Entropy Search (MES) acquisiton function have been implemented. The suffix (`<gp>`, `<svgp>`, `<dkl>` indicate with which regressor the implementation is compatible with.)


## Citation
If you use any part of this code for your own work, please cite
```
@misc{hernandezgarcia2023multifidelity,
      title={Multi-Fidelity Active Learning with GFlowNets}, 
      author={Alex Hernandez-Garcia and Nikita Saxena and Moksh Jain and Cheng-Hao Liu and Yoshua Bengio},
      year={2023},
      eprint={2306.11715},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
