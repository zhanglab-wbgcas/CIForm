import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
import math
import pandas as pd
import scanpy as sc
import os
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score ,f1_score ,recall_score ,precision_score
from torch.utils.data import (DataLoader ,Dataset)
torch.set_default_tensor_type(torch.DoubleTensor)
import numpy as np
import random
from sklearn import preprocessing

def same_seeds(seed):
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(2021)


def getXY(gap, adata, y_trains):
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.todense()
    X = np.asarray(X)

    single_cell_list = []
    for single_cell in X:
        feature = []
        length = len(single_cell)
        for k in range(0, length, gap):
            if (k + gap <= length):
                a = single_cell[k:k + gap]
            else:
                a = single_cell[length - gap:length]

            a = preprocessing.scale(a)
            feature.append(a)
        feature = np.asarray(feature)
        single_cell_list.append(feature)

    single_cell_list = np.asarray(single_cell_list)

    cell_types = []

    if(len(y_trains) > 0):
        for i in y_trains:
            i = str(i).upper()
            if (not cell_types.__contains__(i)):
                cell_types.append(i)

        return single_cell_list, y_trains, cell_types
    else:
        return single_cell_list

def getNewData(cells, cell_types):
    labels = []

    label_to_cell_type = {}  # 用于存储 label 和 cell_types 的映射关系
    label_to_cell_type[0] = "unassigned"  # 将0映射为unknown类型

    for i in range(len(cells)):
        cell = cells[i]
        cell = str(cell).upper()

        if (cell_types.__contains__(cell)):
            indexs = cell_types.index(cell)
            labels.append(indexs + 1)
            label_to_cell_type[indexs + 1] = cell
        else:
            labels.append(0)  # 0 denotes the unknowns cell types

    return np.asarray(labels),label_to_cell_type

class TrainDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = torch.from_numpy(self.data)
        label = torch.from_numpy(self.label)

        return data[index], label[index]

class TestDataSet(Dataset):
    def __init__(self, data):
        self.data = data

        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = torch.from_numpy(self.data)
        return data[index]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CIForm(nn.Module):
    def __init__(self, input_dim, nhead=2, d_model=80, num_classes=2, dropout=0.1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=1024, nhead=nhead, dropout=dropout
        )
        self.positionalEncoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, mels):
        out = mels.permute(1, 0, 2)
        out = self.positionalEncoding(out)
        out = self.encoder_layer(out)
        out = out.transpose(0, 1)
        out = out.mean(dim=1)
        out = self.pred_layer(out)
        return out

def ciForm(s ,X_train,train_labels ,X_test,n_epochs=20):
    gap = s
    d_models = s
    heads = 8
    lr = 0.0001
    dp = 0.1
    batch_sizes = 32
    n_epochs = n_epochs

    train_data, train_cells, train_cellTypes = getXY(gap,X_train,train_labels)
    cell_types = train_cellTypes
    Train_labels,label_to_cell_type = getNewData(train_cells, cell_types)
    labels = Train_labels
    num_classes = len(cell_types) + 1

    query_data = getXY(gap, X_test, [])

    model = CIForm(input_dim=d_models, nhead=heads, d_model=d_models,
                   num_classes=num_classes ,dropout=dp)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)


    train_dataset = TrainDataSet(data=train_data, label=labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True,
                              pin_memory=True)

    test_dataset = TestDataSet(data=query_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False,
                             pin_memory=True)


    model.train()
    for epoch in range(n_epochs):
        # model.train()
        # These are used to record information in training.
        train_loss = []
        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            data, labels = batch
            logits = model(data)
            labels = torch.tensor(labels, dtype=torch.long)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_loss = sum(train_loss) / len(train_loss)


        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")

    model.eval()
    y_predict = []
    for batch in tqdm(test_loader):
        data = batch
        with torch.no_grad():
            logits = model(data)
        preds = logits.argmax(1)
        preds = preds.cpu().numpy().tolist()
        y_predict.extend(preds)
    y_predicts_char = [label_to_cell_type[pred] for pred in y_predict]

    return y_predicts_char

import os
import time as tm
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

s = 1024  ##建议每次调整为2的n次方  比如  64 128 256 512
path = "data/"
ref_name = "03SRP330542"
query_name = "04SRP235541"
topgenes = 2000


ref_adata = sc.read(path + ref_name + ".h5ad")
query_adata = sc.read(path + query_name + ".h5ad")

y_train = ref_adata.obs['Celltype'].to_numpy()

adata = ref_adata.concatenate(query_adata)
adata.var_names_make_unique()

###标准化(提供数据已经行标准化，因此不进行该步骤)
# sc.pp.normalize_total(adata)
# sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=topgenes)
adata = adata[:, adata.var['highly_variable']]

ref_adata = adata[:len(ref_adata)]
query_adata = adata[len(ref_adata):]

print("ref_adata",ref_adata)
print("query_adata",query_adata)

start = tm.time()
preds = ciForm(s ,ref_adata ,y_train,query_adata,n_epochs=50)

log_path = "log/cross/" + "/" + ref_name + "_" + query_name + "/"

if (not os.path.isdir(log_path)):
    os.makedirs(log_path)

np.save(log_path + "_pred", preds)


