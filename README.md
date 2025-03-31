# Understanding-the-Stability-based-Generalization-of-Personalized-Federated-Learning
Official code for ICLR2025
> [Understanding-the-Stability-based-Generalization-of-Personalized-Federated-Learning](https://openreview.net/forum?id=znhZbonEoe).


This repository contains the official implementation for the manuscript: 



# Experiments

The implementations of each method are provided in the folder `/fedml_api/standalone`, while experiments are provided in the folder `/fedml_experiments/standalone`.


Directly python the corresponding files to run the code in /fedml_experiments/standalone.

```
python fedml_experiments/standalone/dfedalt/main_dfedalt_dis.py
python fedml_experiments/standalone/fedalt/main_fedalt_dis.py
```

This code is based on the project in [DisPFL](https://github.com/rong-dai/DisPFL) and [DFedPGP](https://github.com/YingqiLiu1999/DFedPGP). 


# Citation

If you find this repo useful for your research, please consider citing the paper

```
@inproceedings{liuunderstanding,
  title={Understanding the Stability-based Generalization of Personalized Federated Learning},
  author={Liu, Yingqi and Li, Qinglun and Tan, Jie and Shi, Yifan and Shen, Li and Cao, Xiaochun},
  booktitle={The Thirteenth International Conference on Learning Representations}
}

```
