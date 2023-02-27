import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
import math
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm
from torch.utils.data import (DataLoader,Dataset)
torch.set_default_tensor_type(torch.DoubleTensor)
import numpy as np
import random
from sklearn.model_selection import KFold, StratifiedKFold
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

def getXY(gap, data_path, data_name,topgenes):
    adata = sc.read_csv(data_path + data_name + ".csv", delimiter=",")

    sc.pp.filter_genes(adata, min_cells=1)
    if(not data_name.__contains__("log")):
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

    y_train = pd.read_csv(data_path + "Labels.csv")
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

class myDataSet(Dataset):
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


def trainTransformer(model_name, species,lai, epochs,
                     train_loader,test_loader,
                     heads,d_models,num_classes,dp,lr,cell_types):
    log_dir = "log/" + model_name + "/" + species + "/" + lai + "/"
    log_txt = "log/" + model_name + "/" + species + "/"

    if (not os.path.isdir(log_dir)):
        os.makedirs(log_dir)

    print("num_classes",num_classes)
    model = Classifier(input_dim=d_models, nhead=heads, d_model=d_models, num_classes=num_classes,dropout=dp)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    n_epochs = epochs
    acc_record = {'train': [], 'dev': []}
    loss_record = {'train': [], 'dev': []}
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
        acc_record['train'].append(train_acc)
        loss_record['train'].append(train_loss)

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


    last_path = log_dir + str(epochs) + "/"
    if (not os.path.isdir(last_path)):
        os.makedirs(last_path)
    with open(log_txt + "end_norm.txt", "a") as f:
        f.writelines("log_dir:" + last_path + "\n")
        f.writelines("acc:" + str(all_acc) + "\n")
        f.writelines('f1:' + str(all_f1) + "\n")
        

    np.save(last_path + model_name + 'y_test.npy', labelss)
    np.save(last_path + model_name + 'y_predict.npy', y_predict)

    np.save(last_path + model_name + 'y_tests.npy', labelsss)
    np.save(last_path + model_name + 'y_predicts.npy', y_predicts)

    torch.save(model.state_dict(), last_path + model_name + '.tar')

#'02Goolam'
model_name = "Tcell_std_end5"
indexs = ['Segerstolpe_c3',] #'Segerstolpe_c3' ,'Muraro_log','Human'  ,'Mouse',,'Mouse','Segerstolpe' ,'Xin_log','Muraro_log','Xin_log','Muraro_log' ,'Human', 'Human','Mouse','Mouse', ,'Xin_log','Muraro_log'

for j in indexs:
    x_Traindata_path = 'Dataset/scRNAseq_Benchmark_datasets/Intra-dataset/Pancreatic_data/'+j+"//"
    Traindata_name = j
    species = Traindata_name
    print("species",species)
    species = "Intra/Pancreatic_data/"+ species


    dim = 1024
    topgenes = 2000
    Data, labels, cell_types = getXY(dim, x_Traindata_path, Traindata_name,topgenes)
    l = Data.shape[0]
    if(l > 5000):
        batch_sizes = 512
    else:
        batch_sizes = 256

    print("batch_sizes",batch_sizes)

    num_classes = len(cell_types)
    print("Data.shape",Data.shape)
    skf = StratifiedKFold(n_splits=5,random_state=2021, shuffle=True)
    fold = 0
    Indexs = []
    for index in range(len(Data)):
        Indexs.append(index)
    Indexs = np.asarray(Indexs)
    for train_index, test_index in skf.split(Data, labels):
        fold = fold + 1
        X_train, X_test = Data[train_index], Data[test_index]
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)

        y_train, y_test = labels[train_index], labels[test_index]

        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        train_dataset = myDataSet(data=X_train,label=y_train)
        train_loader = DataLoader(train_dataset,batch_size=batch_sizes,shuffle=True,pin_memory=True)

        test_dataset = myDataSet(data=X_test,label=y_test)
        test_loader = DataLoader(test_dataset,batch_size=batch_sizes,shuffle=False,pin_memory=True)

        num_classes = len(cell_types)
        number_encoder = 1
        heads = 64
        dim_feedforward = 1024
        dp = 0.1
        lr = 0.0001
        lai = "RELU_dataloader_comGenes_norm/mean/fold"+str(fold)+"/"+str(batch_sizes)+"_"+str(dim)+"_"+ str(number_encoder)+"_"+str(heads)+"_"+str(dp)+"_"+str(lr)  + "_"+str(dim_feedforward)+"_"+str(topgenes)

        epochs = 20
        trainTransformer(model_name,species,lai,epochs,
                         train_loader,test_loader,
                         heads,dim,num_classes,dp,lr,cell_types)

