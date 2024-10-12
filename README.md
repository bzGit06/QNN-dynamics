# QNN_dynamics
This repository contains the official Python implementation of [*Dynamical transition in controllable quantum neural networks with large depth*](https://arxiv.org/abs/2311.18144), an article by [Bingzhi Zhang](https://sites.google.com/view/bingzhi-zhang/home), [Peng Xu](https://francis-hsu.github.io/), [Xiao-Chuan Wu](https://scholar.google.com.hk/citations?user=ADEnvRwAAAAJ&hl=zh-CN), [Liang Jiang](https://pme.uchicago.edu/group/jiang-group), and [Quntao Zhuang](https://sites.usc.edu/zhuang).

## Citation
```
@misc{zhang2023dynamical,
      title={Dynamical phase transition in quantum neural networks with large depth}, 
      author={Zhang, Bingzhi and Liu, Junyu and Wu, Xiao-Chuan and Jiang, Liang and Zhuang, Quntao},
      year={2023},
      eprint={2311.18144},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```

## Prerequisite
The simulation of quantum circuits is performed via the [TensorCircuit](https://tensorcircuit.readthedocs.io/en/latest/#) package with [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) backend. Use of GPU is not required, but highly recommended. 

Additionally, the packages [`opt_einsum`](https://optimized-einsum.readthedocs.io/en/stable/) is used for speeding up certain evaluation, and [Pennylane](https://docs.pennylane.ai/en/stable) is needed for perform experiment on IBM Quantum device.



## File Structure
The file `RPA_jax.ipynb' contains all numerical simulation with random Pauli ansatz. The file `HEA.ipynb' contains dynamics results from simulation with Hardware efficient ansatz. The file `HEA_qiskitv2.ipynb' contains experimental results on noisy simulation and IBM Quantum device. Code in `qntk_theory.py' were used to calculate analytical average results from Haar ensemble and restricted Haar ensemble.
