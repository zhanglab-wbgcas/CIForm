
##Import the required packages
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
import math
import pandas as pd
import scanpy as sc
import os
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from torch.utils.data import (DataLoader,Dataset)
torch.set_default_tensor_type(torch.DoubleTensor)
import numpy as np
import random
from sklearn import preprocessing
##Set random seeds
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


##Gene embedding, 
#Function
    #The pre-processed scRNA-seq data is converted into a form acceptable to the Transformer encoder
#Parameters 
    #gap: The length of a sub-vector
    #adata: pre-processed scRNA-seq data. The rows represent the cells and the columns represent the genes
    #Traindata_paths: the paths of the cell type labels file(.csv) corresponding to the training data
def getXY(gap, adata, Traindata_paths):
    # Converting the gene expression matrix into sub-vectors
    #(n_cells,n_genes) -> (n_cells,gap_num,gap)  gap_num = int(gene_num / gap) + 1
    X = adata.X  #getting the gene expression matrix
    single_cell_list = []
    for single_cell in X:
        feature = []
        length = len(single_cell)
        #spliting the gene expression vector into some sub-vectors whose length is gap
        for k in range(0, length, gap):
            if (k + gap <= length):
                a = single_cell[k:k + gap]
            else:
                a = single_cell[length - gap:length]
            #scaling each sub-vectors 
            a = preprocessing.scale(a)
            feature.append(a)
        feature = np.asarray(feature)
        single_cell_list.append(feature)
    
    single_cell_list = np.asarray(single_cell_list) #(n_cells,gap_num,gap)
    
    ##Obtaining cell types and all cell type labels of training data
    if(Traindata_paths != None):
        #get cell type labels
        y_trains = []
        for path in Traindata_paths:
            y_train = pd.read_csv(path + "Labels.csv")
            y_train = y_train.T
            y_train = y_train.values[0]
            y_trains.extend(y_train)
        # all cell types 
        cell_types = []
        for i in y_trains:
            i = str(i).upper()
            if (not cell_types.__contains__(i)):
                cell_types.append(i)
        
        return single_cell_list, y_trains, cell_types
    else:
        return single_cell_list


##Function
    #Converting label annotation to numeric form
##Parameters
    #cells:  all cell type labels
    #cell_types: all cell types of Training datasets
def getNewData(cells, cell_types):
    labels = []
    for i in range(len(cells)):
        cell = cells[i]
        cell = str(cell).upper()

        if (cell_types.__contains__(cell)):
            indexs = cell_types.index(cell)
            labels.append(indexs + 1)
        else:
            labels.append(0)  # 0 denotes the unknowns cell types

    return np.asarray(labels)


##function
    #getting the input of CIForm
##parameters
    #gap              :the length of a sub-vector 
    #Traindata_paths  :the paths of training datasets
    #Train_names      :the names of training datasets 
    #Testdata_path    :the path of testing dataset
    #Testdata_name    :the name of testing dataset
    #topgenes         :the number of highly variable genes(HVGs)
def getData(gap, Traindata_paths, Train_names,
            Testdata_path, Testdata_name, topgenes):
    all_adata = sc.AnnData
    train_adata = sc.AnnData
    for sa in range(0, len(Train_names)):
        #getting the training datasets and making var names unique
        temp_adata = sc.read_csv(Traindata_paths[sa] + Train_names[sa] + ".csv", first_column_names=True)
        temp_adata.var_names = [str(i).upper() for i in temp_adata.var_names]
        temp_adata.var_names_make_unique()
        #integrating multiple training RNA-Seq datasets in the inner way to obtain their common genes
        train_adata = train_adata.concatenate(temp_adata)
        train_adata.var_names_make_unique()
    
    Trainadata_num = len(train_adata)
    
    #Getting the testing dataset and making var names unique
    test_adata = sc.read_csv(Testdata_path + Testdata_name + ".csv")
    test_adata.var_names_make_unique()
    
    #Preprocessing the scRNA-seq data
    sc.pp.log1p(train_adata)
    sc.pp.log1p(test_adata)
    
    #Integrating the training dataset and testing dataset in the inner way to obtain their common genes
    all_adata = all_adata.concatenate(train_adata)
    all_adata = all_adata.concatenate(test_adata)
    
    ## Filtering the genes which not express in all cells
    sc.pp.filter_genes(all_adata, min_cells=1)

    #If the number of HVGs is greater than the number of common genes in the training dataset and the testing dataset, 
    #the number of HVGs is set as the number of common genes
    width = all_adata.X.shape[1
    if (width < topgenes):
        topgenes = width
    
    #Obtaining the HVGs
    sc.pp.highly_variable_genes(all_adata, n_top_genes=topgenes)
    all_adata.raw = all_adata
    all_adata = all_adata[:, all_adata.var.highly_variable]
    
    #obtaining the pre-processed training dataset and test dataset 
    Train_adata = all_adata[:Trainadata_num]
    Test_adata = all_adata[Trainadata_num:]

    del all_adata   #removing all_adata to freeing memory
    
    #Converting the pre-processed training dataset into the input of Transformer Encoder and 
    #obtaining its cell type annotations and cell types
    train_data, train_cells, train_cellTypes = getXY(gap, Train_adata, Traindata_paths)
    
    #Converting the preprocessed test dataset into the input of Transformer Encoder 
    Testdata_paths = []
    Testdata_paths.append(Testdata_path)
    test_data = getXY(gap, Test_adata, Label_path = None)
    cell_types = train_cellTypes
    
    ##Converting cell type annotation of the training dataset to numeric form
    Train_labels = getNewData(train_cells, cell_types)

    return train_data, Train_labels, test_data, cell_types


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



##Positional Encoder Layer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        ##the sine function is used to represent the odd-numbered sub-vectors
        pe[:, 0::2] = torch.sin(position * div_term)
        ##the cosine function is used to represent the even-numbered sub-vectors
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
##CIForm
##function
    #annotating cell type identification of scRNA-seq data
##parameters
    #input_dim  :Default is equal to gap
    #nhead      :Number of heads in the attention mechanism
    #d_model    :Default is equal to gap
    #num_classes:Number of cell types
    #dropout    :dropout rate which is used to prevent model overfitting
class CIForm(nn.Module):
    def __init__(self, input_dim, nhead=2, d_model=80, num_classes=2, dropout=0.1):
        super().__init__()
        #TransformerEncoderLayer with self-attention 
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=1024, nhead=nhead, dropout=dropout
        )
        
        #Positional Encoding with self-attention 
        self.positionalEncoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        
        #Classification layer
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, mels):
        out = mels.permute(1, 0, 2)
        #Positional Encoding layer
        out = self.positionalEncoding(out)
        #Transformer Encoder layer layer
        out = self.encoder_layer(out)
        out = out.transpose(0, 1)
        #Pooling layer
        out = out.mean(dim=1)
        #Classification layer
        out = self.pred_layer(out)
        return out

##main
##parameters
    #s                  :the length of a sub-vector
    #referece_datapath  :the paths of referece datasets
    #Train_names        :the names of referece datasets  
    #Testdata_path      :the path pf test dataset
    #Testdata_name      :the name of test dataset
def ciForm(s,referece_datapaths,Train_names,Testdata_path,Testdata_name):
    gap = s           #the length of a sub-vector
    topgenes = 2000   #the number of HVGs
    d_models = s
    heads = 64        #the number of heads in self-attention mechanism

    lr = 0.0001         #learning rate
    dp = 0.1            #dropout rate
    batch_sizes = 256  #the size of batch
    n_epochs = 20      #the number of epoch
    
    #Getting the data which input into the CIForm
    train_data, labels, query_data, cell_types = getData(gap, referece_datapaths,Train_names,Testdata_path,
                                             Testdata_name,topgenes)
    
    #Number of cell types plus unassigned cell type
    num_classes = np.unique(cell_types) + 1
    
    #Constructing the CIForm model
    model = CIForm(input_dim=d_models, nhead=heads, d_model=d_models,
                       num_classes=num_classes,dropout=dp)
    
    
    #Setting loss function
    criterion = nn.CrossEntropyLoss()
    #Setting optimization function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    #Setting the training dataset
    train_dataset = TrainDataSet(data=train_data, label=labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True,
                              pin_memory=True)
    #Setting the test dataset
    test_dataset = TestDataSet(data=query_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False,
                             pin_memory=True)
    #Known Cell Types
    new_cellTypes = []
    new_cellTypes.append("unassigned")
    new_cellTypes.extend(cell_types)

    
    #startting training CIForm.Using training data to train CIForm
    #n_epochs: the times of Training 
    model.train()
    for epoch in range(n_epochs):
        # These are used to record information in training.
        train_loss = []
        train_accs = []
        train_f1s = []
        for batch in tqdm(train_loader):
            # A batch consists of scRNA-seq data and corresponding cell type annotation.
            data, labels = batch
            logits = model(data)
            labels = torch.tensor(labels, dtype=torch.long)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #Getting the predicted cell type
            preds = logits.argmax(1)
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            #Metrics
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels,preds,average='macro')
            train_loss.append(loss.item())
            train_accs.append(acc)
            train_f1s.append(f1)
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        train_f1 = sum(train_f1s) / len(train_f1s)

        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}, f1 = {train_f1:.5f}")
        
        ##Starting the validation model, which predicts the cell types in the test dataset
        model.eval()
        test_accs = []
        test_f1s = []
        y_predict = []
        labelss = []
        for batch in tqdm(test_loader):
            # A batch consists of scRNA-seq data and corresponding cell type annotations.
            data, labels = batch
            with torch.no_grad():
                logits = model(data)
            # Getting the predicted cell type
            preds = logits.argmax(1)
            preds = preds.cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()
            
            #Metrics
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='macro')
            test_f1s.append(f1)
            test_accs.append(acc)

            y_predict.extend(preds)
            labelss.extend(labels)
        test_acc = sum(test_accs) / len(test_accs)
        test_f1 = sum(test_f1s) / len(test_f1s)
        print("---------------------------------------------end test---------------------------------------------")
        #Metrics
        all_acc = accuracy_score(labelss, y_predict)
        all_f1 = f1_score(labelss, y_predict, average='macro')
        print("all_acc:", all_acc,"all_f1:", all_f1)
        
        labelsss = []
        y_predicts = []
        for i in labelss:
            labelsss.append(cell_types[i])
        for i in y_predict:
            y_predicts.append(cell_types[i])
        
        
        #Storing predicted cell types and the CIForm
        log_dir = "log/"
        if (not os.path.isdir(log_dir)):
            os.makedirs(log_dir)

        np.save(log_dir  + 'y_tests.npy', labelsss)
        np.save(log_dir  + 'y_predicts.npy', y_predicts)

        torch.save(model.state_dict(), log_dir + 'CIForm.tar')
        with open(log_dir + "resilt.txt", "a") as f:
            f.writelines("acc:" + str(all_acc) + "\n")
            f.writelines('f1:' + str(all_f1) + "\n")



s = 1024

x_Traindata_path = 'Dataset/imu/Sun/'
Train_name = 'Sun'
referece_datapaths = [x_Traindata_path]
Train_names = [Train_name]

Testdata_path = 'Dataset/imu/pbmc_10k_v3/'
Testdata_name = 'pbmc_10k_v3'
##Datasets: The rows represent the cells and the columns represent the genes
ciForm(s,referece_datapaths,Train_names,Testdata_path,Testdata_name)

