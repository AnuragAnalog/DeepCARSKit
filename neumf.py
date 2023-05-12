import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error


data = pd.read_csv("./dataset/frappe/frappe.inter", sep = ',')

cols = ["user_id", "item_id", "rating"]
data = data.iloc[:, :3]
data = data.dropna(axis = 0).reset_index(drop = True)
data.columns = cols

user_encoder = {user_id: i for i, user_id in enumerate(data['user_id'].unique())}
item_encoder = {item_id: i for i, item_id in enumerate(data['item_id'].unique())}

data['user_id'] = data['user_id'].apply(lambda x: user_encoder[x])
data['item_id'] = data['item_id'].apply(lambda x: item_encoder[x])

class NCF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=8, mlp_layers=[16, 8], dropout=0.2):
        super(NCF, self).__init__()
        
        self.user_embedding_mf = nn.Embedding(num_users, mf_dim)
        self.item_embedding_mf = nn.Embedding(num_items, mf_dim)
        
        self.user_embedding_mlp = nn.Embedding(num_users, mlp_layers[0]//2)
        self.item_embedding_mlp = nn.Embedding(num_items, mlp_layers[0]//2)
        
        self.mlp_layers = nn.ModuleList()
        for i in range(len(mlp_layers) - 1):
            self.mlp_layers.append(nn.Linear(mlp_layers[i], mlp_layers[i+1]))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(p=dropout))
        
        self.output_layer = nn.Linear(mf_dim + mlp_layers[-1], 1)

        
    def forward(self, user_ids, item_ids):
        user_embedding_mf = self.user_embedding_mf(user_ids)
        item_embedding_mf = self.item_embedding_mf(item_ids)
        mf_output = torch.mul(user_embedding_mf, item_embedding_mf)
        
        user_embedding_mlp = self.user_embedding_mlp(user_ids)
        item_embedding_mlp = self.item_embedding_mlp(item_ids)
        mlp_output = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=1)
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)
        
        concat_output = torch.cat([mf_output, mlp_output], dim=1)
        
        output = self.output_layer(concat_output)
        return output.squeeze()

num_epochs = 10
batch_size = 256
learning_rate = 0.001
mf_dim = 8
mlp_layers = [64, 32, 16]
dropout = 0.2
n_splits = 10

model = NCF(num_users=len(user_encoder), num_items=len(item_encoder), mf_dim=mf_dim, mlp_layers=mlp_layers, dropout=dropout)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = criterion.to(device)

def train_test(num_epochs, train_data, val_data, test_data):
    for epoch in range(num_epochs):
        # shuffle the training data
        train_data = train_data.sample(frac=1)
        
        # iterate over batches
        for i in range(0, len(train_data), batch_size):
            # get batch data
            batch_data = train_data.iloc[i:i+batch_size]
            user_ids = torch.LongTensor(batch_data['user_id'].values).to(device)
            item_ids = torch.LongTensor(batch_data['item_id'].values).to(device)
            ratings = torch.FloatTensor(batch_data['rating'].values).to(device)
            
            # zero the gradients
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(user_ids, item_ids)
            loss = criterion(outputs, ratings)
            
            # backward pass
            loss.backward()
            optimizer.step()

    # evaluate the model on the validation set
    val_user_ids = torch.LongTensor(val_data['user_id'].values).to(device)
    val_item_ids = torch.LongTensor(val_data['item_id'].values).to(device)
    val_ratings = torch.FloatTensor(val_data['rating'].values)
    val_outputs = model(val_user_ids, val_item_ids)

    val_outputs = val_outputs.detach().cpu().numpy()

    if len(test_data) == 0:

        return mean_absolute_error(val_outputs, val_ratings), np.sqrt(mean_squared_error(val_outputs, val_ratings))

    else:

        test_user_ids = torch.LongTensor(test_data['user_id'].values).to(device)
        test_item_ids = torch.LongTensor(test_data['item_id'].values).to(device)
        test_ratings = torch.FloatTensor(test_data['rating'].values)
        test_outputs = model(test_user_ids, test_item_ids)

        test_outputs = test_outputs.detach().cpu().numpy()

        val_mae, val_rmse = mean_absolute_error(val_outputs, val_ratings), np.sqrt(mean_squared_error(val_outputs, val_ratings))
        test_mae, test_rmse = mean_absolute_error(test_outputs, test_ratings), np.sqrt(mean_squared_error(test_outputs, test_ratings))

        return val_mae, val_rmse, test_mae, test_rmse




def cross_validation(num_epochs, data, n_splits):

    kf = KFold(n_splits= n_splits, shuffle=True, random_state=42)

    mae_cv = 0
    rmse_cv = 0
    fold = 0
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        fold+=1 
        print("Fold: {}".format(fold))

        train_data = data.iloc[train_index].reset_index(drop = True)
        val_data = data.iloc[test_index].reset_index(drop = True)

        mae, rmse = train_test(num_epochs, train_data, val_data, '')
        mae_cv += mae
        rmse_cv += rmse


    return mae_cv, rmse_cv

    


        
    # print the epoch and validation loss

mae_cv, rmse_cv = cross_validation(num_epochs, data, n_splits)
print('MAE: {:.4f}, RMSE: {:.4f}'.format(mae_cv/n_splits, rmse_cv/n_splits))

# train_data, test_data = train_test_split(data, test_size = )
train_data, val_data, test_data = np.split(data.sample(frac=1), [int(.8*len(data)), int(.9*len(data))])


val_mae, val_rmse, test_mae, test_rmse = train_test(num_epochs, train_data, val_data, test_data)

print('val_mae: {:.4f}, val_rmse: {:.4f}'.format(val_mae, val_rmse))
print('test_mae: {:.4f}, test_rmse: {:.4f}'.format(test_mae, test_rmse))