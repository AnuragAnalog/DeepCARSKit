#!/usr/bin/python3

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FM(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_contexts = None, context_encoders = None):
        super(FM, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.linear = nn.Linear(embedding_dim, 1)

        # if num_contexts:
            # self.context_embedding = nn.Embedding(num_contexts, embedding_dim)
        
        self.num_contexts = num_contexts
        self.context_encoders = context_encoders
        if num_contexts > 0:
            self.v = nn.Parameter(torch.randn(embedding_dim, num_contexts))
            self.context_embedding = nn.ModuleList([nn.Embedding(len(encoder.classes_), embedding_dim) for _, encoder in self.context_encoders.items()])

    def forward(self, user_idx, item_idx, context_idx = None):
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)

        # if len(context_idx) > 0:
        #     print(context_idx)
        #     context_emb = self.context_embedding(context_idx.to(torch.int64))
        #     fm_term = (user_emb * item_emb * context_emb).sum(dim=1)
        #     linear_term = self.linear(user_emb) + self.linear(item_emb) + self.linear(context_emb).squeeze()
        
        if self.num_contexts > 0:
            context_embeddings = []
            for i, emb_layer in enumerate(self.context_embedding):
                context_idx_ = context_idx[:, i].unsqueeze(1)
                context_emb = emb_layer(context_idx_.to(torch.int64)).squeeze(1)
                context_embeddings.append(context_emb)
            context_embeddings = torch.stack(context_embeddings, dim=1)

            factor_1 = torch.matmul((user_emb.unsqueeze(1) + context_embeddings), self.v)
            factor_2 = torch.matmul((item_emb.unsqueeze(1) + context_embeddings), self.v)
            interaction_output = torch.sum(torch.mul(factor_1, factor_2), 1)

            linear_output = self.linear(user_emb.unsqueeze(1) * item_emb.unsqueeze(1) * context_embeddings).squeeze(1)

            output = interaction_output + linear_output.unsqueeze(2)
        else:
            fm_term = (user_emb * item_emb).sum(dim=1)
            linear_term = self.linear(user_emb + item_emb).squeeze()
            output = fm_term + linear_term
        
        return output

class Dataset(Dataset):
    def __init__(self, df, user_encoder, item_encoder, context_encoders = None):
        self.user_idx = user_encoder.transform(df['user_id'])
        self.item_idx = item_encoder.transform(df['item_id'])
        self.rating = df["rating"].astype(float).values
        

        self.context_encoders = context_encoders
        self.context_idx = np.empty((len(df), len(self.context_encoders)))
        if self.context_encoders:
            for i, (context, context_encoder) in enumerate(self.context_encoders.items()):
                self.context_idx[:, i] = context_encoder.transform(df[context])
        
    def __len__(self):
        return len(self.rating)
    
    def __getitem__(self, idx):
        
        if self.context_encoders:
            return self.user_idx[idx], self.item_idx[idx], self.rating[idx], self.context_idx[idx]
        
        else:
            return self.user_idx[idx], self.item_idx[idx], self.rating[idx]

if __name__ == "__main__":
    data = pd.read_csv('./dataset/frappe/frappe.inter')

    cols = ["user:token", "item:token", "cnt:float"]
    context_cols = ["daytime:token", "weekday:token", "isweekend:token", "homework:token", "cost:token", "weather:token", "country:token", "city:token"]

    data = data.iloc[:, :-2]
    data = data.dropna(axis = 0).reset_index(drop = True)
    data.columns = cols + context_cols

    users = data['user:token'].unique()
    items = data['item:token'].unique()

    user_to_idx = {user: idx for idx, user in enumerate(users)}
    item_to_idx = {item: idx for idx, item in enumerate(items)}

    data['user_id'] = data['user:token'].apply(lambda x: user_to_idx[x])
    data['item_id'] = data['item:token'].apply(lambda x: item_to_idx[x])
    data["rating"] = data["cnt:float"].astype(float)

    # Encode the user and item ids
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    data['user_id'] = user_encoder.fit_transform(data['user_id'])
    data['item_id'] = item_encoder.fit_transform(data['item_id'])

    context_encoders = {}
    for context in context_cols:
        context_encoder = LabelEncoder()
        contexts = data[context].unique()
        context_to_idx = {context: idx for idx, context in enumerate(contexts)}
        data[context] = data[context].apply(lambda x: context_to_idx[x])
        data[context] = context_encoder.fit_transform(data[context])
        context_encoders[context] = context_encoder


    def train(model, loss_function, optimizer, train_loader, epochs, device, contexts = False):
        for epoch in range(epochs):
            model.train()

            if contexts:

                for user_idx, item_idx, rating, context_idx in train_loader:
                    user_idx = user_idx.to(device)
                    item_idx = item_idx.to(device)
                    rating = rating.to(device)
                    context_idx = context_idx.to(device)

                    optimizer.zero_grad()

                    output = model(user_idx, item_idx, context_idx)

                    rating = rating.unsqueeze(1)
                    loss = loss_function(output, rating.float())

                    loss.backward()
                    optimizer.step()

            else:
            
                for user_idx, item_idx, rating in train_loader:
                    user_idx = user_idx.to(device)
                    item_idx = item_idx.to(device)
                    rating = rating.to(device)

                    optimizer.zero_grad()

                    output = model(user_idx, item_idx)

                    loss = loss_function(output, rating.float())

                    loss.backward()
                    optimizer.step()

            if (epoch + 1) % 10 == 0:
                print("Epoch: {}".format(epoch+1))


        return model


    def evaluate(model, test_loader, device, contexts = False):

        mae = 0
        rmse = 0

        if contexts:

            for user_idx, item_idx, rating, context_idx in test_loader:
                user_idx = user_idx.to(device)
                item_idx = item_idx.to(device)
                context_idx = context_idx.to(device)

                output = model(user_idx, item_idx, context_idx)

                output = output.detach().cpu().numpy()

                print(output.shape, rating.shape)

                mae += mean_absolute_error(output, rating)
                rmse += np.sqrt(mean_squared_error(output, rating))

        else:

            for user_idx, item_idx, rating in test_loader:
                user_idx = user_idx.to(device)
                item_idx = item_idx.to(device)

                output = model(user_idx, item_idx)

                output = output.detach().cpu().numpy()

                mae += mean_absolute_error(output, rating)
                rmse += np.sqrt(mean_squared_error(output, rating))


        return mae/len(test_loader), rmse/len(test_loader)


    def cross_validation(data, n_splits, epochs, contexts = False):

        kf = KFold(n_splits= n_splits, shuffle=True, random_state=42)

        mae_cv = 0
        rmse_cv = 0
        fold = 0
        for i, (train_index, test_index) in enumerate(kf.split(data)):

            fold+=1 
            print("Fold: {}".format(fold))

            X_train = data.iloc[train_index].reset_index(drop = True)
            X_test = data.iloc[test_index].reset_index(drop = True)

            loss_function = nn.MSELoss()
            loss_function = loss_function.to(device)


            if contexts:

                train_dataset = Dataset(X_train, user_encoder, item_encoder, context_encoders)
                test_dataset = Dataset(X_test, user_encoder, item_encoder, context_encoders)

                train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=128, shuffle = False)

                model = FM(num_users=len(user_encoder.classes_), num_items=len(item_encoder.classes_), embedding_dim=16, num_contexts = len(context_encoders), context_encoders = context_encoders)
                model = model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.01)
                # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
                model_trained = train(model, loss_function, optimizer, train_loader, epochs, device, contexts)
                mae , rmse = evaluate(model_trained, test_loader, device, contexts)



            else:

                train_dataset = Dataset(X_train, user_encoder, item_encoder)
                test_dataset = Dataset(X_test, user_encoder, item_encoder)

                train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=128, shuffle = False)

                model = FM(num_users=len(user_encoder.classes_), num_items=len(item_encoder.classes_), embedding_dim=16)
                model = model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.01)
                # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
                model_trained = train(model, loss_function, optimizer, train_loader, epochs, device)
                mae , rmse = evaluate(model_trained, test_loader, device)


            mae_cv += mae
            rmse_cv += rmse

        return mae_cv/n_splits, rmse_cv/n_splits


    epochs = 150
    n_splits = 2
    mae_cv, rmse_cv = cross_validation(data, n_splits, epochs, contexts = True)

    # from sklearn.model_selection import train_test_split
    # X_train, X_test = train_test_split(data, test_size = 0.2, random_state = 42)
    print(mae_cv, rmse_cv)