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

## Usage

### 
```Python
import CIForm as CI
pred_result = CI.ciForm(s, referece_datapaths, Train_names, Testdata_path,Testdata_name)
```

in which 

- **s=The length of sub-vector**,
- **referece_datapaths=The path of annotated scRNA-seq datasets**
- **Train_names=The name of annotated scRNA-seq datasets** 
- **Testdata_path=The path of query scRNA-seq datasets**
- **Testdata_name=The name of query scRNA-seq datasets** 

It is recommended that The label file be in the same directory as the corresponding data set and be named Labels.csv
The label file should be a n rows \* 1 column vector. For example,
![image]("https://github.com/zhanglab-wbgcas/CIForm/blob/main/Tutorial/Labels.png")


># Tutorial
- [Cell Type annotation on Intra-datasets](https://github.com/zhanglab-wbgcas/CIForm/blob/main/Tutorial/Tutorial_Intra.ipynb)
- [Cell Type annotation on Inter-datasets](https://github.com/zhanglab-wbgcas/CIForm/blob/main/Tutorial/Tutorial_Inter.ipynb)
- [Cell Type annotation on Inter-datasets using multi-source](https://github.com/zhanglab-wbgcas/CIForm/blob/main/Tutorial/Tutorial_multi-sources.ipynb)

