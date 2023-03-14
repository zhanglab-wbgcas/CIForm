# CIForm
![image]()

># Introduction

CIForm, a Transformer-based model, can annotate cell types. 

Instructions and examples are provided in the following tutorials.

># Requirement
```
Python 3.9.12
PyTorch >= 1.5.0
numpy
pandas
scipy
sklearn
Scanpy
random
```
## Input file
```
reference dataset.

cell type label of reference dataset.

query dataset.
```

## Output file
```
After training the CIForm model, the model will be save at: "log/CIForm.tar".
The model prediction is saved in the log/y_predicts.npy.
```

[//]: # (```)

># Tutorial and Usage
- [Cell Type annotation on Intra-datasets](https://github.com/zhanglab-wbgcas/CIForm/blob/main/Tutorial/Tutorial_Intra.ipynb)
- [Cell Type annotation on Inter-datasets](https://github.com/zhanglab-wbgcas/CIForm/blob/main/Tutorial/Tutorial_Inter.ipynb)
- [Cell Type annotation on Inter-datasets using multi-source](https://github.com/zhanglab-wbgcas/CIForm/blob/main/Tutorial/Tutorial_multi-sources.ipynb)

