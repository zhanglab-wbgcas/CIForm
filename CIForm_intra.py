import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
import math
import pandas as pd
import scanpy as sc
import os
from tqdm.auto import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from torch.utils.data import (DataLoader,Dataset)
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

def getData(gap, data_path,topgenes,label_path=None):
    adata = sc.read_csv(data_path, delimiter=",")

    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=topgenes)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]

    X = adata.X
    single_cell_list = []
    for single_cell in X:
        feature = []
        length = len(single_cell)
        for k in range(0, length, gap):
            if (k + gap > length):
                a = single_cell[length - gap:length]
            else:
                a = single_cell[k:k + gap]

            a = preprocessing.scale(a,axis=0, with_mean=True, with_std=True, copy=True)
            feature.append(a)

        feature = np.asarray(feature)
        single_cell_list.append(feature)

    single_cell_list = np.asarray(single_cell_list)
    if(label_path != None):
        y_train = pd.read_csv(label_path)
        y_train = y_train.T
        y_train = y_train.values[0]

        cell_types = []
        labelss = []
        for i in y_train:
            i = str(i).upper()
            if (not cell_types.__contains__(i)):
                cell_types.append(i)
            labelss.append(cell_types.index(i))

        labelss = np.asarray(labelss)

        return single_cell_list, labelss, cell_types
    else:
        return single_cell_list

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

class Classifier(nn.Module):
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


def CIForm(referece_datapath,label_path,query_datapath,s):
    gap = s
    topgenes = 2000
    d_models = s
    heads = 64

    lr = 0.0001
    dp = 0.1
    batch_sizes = 256


    train_data, labels, cell_types = getData(gap, referece_datapath,topgenes,label_path=label_path)
    query_data = getData(gap, query_datapath,topgenes,label_path=None)
    num_classes = np.unique(cell_types) + 1

    model = Classifier(input_dim=d_models, nhead=heads, d_model=d_models,
                       num_classes=num_classes,dropout=dp)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)


    train_dataset = TrainDataSet(data=train_data, label=labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True,
                              pin_memory=True)
    test_dataset = TestDataSet(data=query_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False,
                             pin_memory=True)

    print("num_classes", num_classes)

    new_cellTypes = []
    new_cellTypes.append("unassigned")
    # for i in cell_types:
    new_cellTypes.extend(cell_types)

    n_epochs = 20

    model.train()
    for epoch in range(n_epochs):
        # model.train()
        # These are used to record information in training.
        train_loss = []
        train_accs = []
        train_f1s = []
        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            data, labels = batch
            logits = model(data)
            labels = torch.tensor(labels, dtype=torch.long)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(1)
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()

            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels,preds,average='macro')
            train_loss.append(loss.item())
            train_accs.append(acc)
            train_f1s.append(f1)
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        train_f1 = sum(train_f1s) / len(train_f1s)

        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}, f1 = {train_f1:.5f}")

        model.eval()
        test_accs = []
        test_f1s = []
        y_predict = []
        labelss = []
        for batch in tqdm(test_loader):
            # A batch consists of image data and corresponding labels.
            data, labels = batch
            with torch.no_grad():
                logits = model(data)
            preds = logits.argmax(1)
            preds = preds.cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='macro')
            test_f1s.append(f1)
            test_accs.append(acc)

            y_predict.extend(preds)
            labelss.extend(labels)
        test_acc = sum(test_accs) / len(test_accs)
        test_f1 = sum(test_f1s) / len(test_f1s)
        print("---------------------------------------------end test---------------------------------------------")
        print("len(y_predict)",len(y_predict))
        all_acc = accuracy_score(labelss, y_predict)
        all_f1 = f1_score(labelss, y_predict, average='macro')
        print("all_acc:", all_acc,"all_f1:", all_f1)

        labelsss = []
        y_predicts = []
        for i in labelss:
            labelsss.append(cell_types[i])
        for i in y_predict:
            y_predicts.append(cell_types[i])

        log_dir = "log/"
        log_txt = "log/"

        if (not os.path.isdir(log_dir)):
            os.makedirs(log_dir)

        last_path = log_dir + str(n_epochs) + "/"
        if (not os.path.isdir(last_path)):
            os.makedirs(last_path)
        with open(log_txt + "end_norm.txt", "a") as f:
            f.writelines("log_dir:" + last_path + "\n")
            f.writelines("acc:" + str(all_acc) + "\n")
            f.writelines('f1:' + str(all_f1) + "\n")
