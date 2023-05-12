import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDatasetWithContext(Dataset):
    def __init__(self, users, items, context_features, targets):
        self.users = users
        self.items = items
        self.context_features = context_features
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        return self.users[index], self.items[index], self.context_features[index], self.targets[index]
        
class NCFWithContext(nn.Module):
    def __init__(self, num_users, num_items, num_context_features, embedding_dim, hidden_units, dropout_rate):
        super(NCFWithContext, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.context_embedding = nn.Linear(num_context_features, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, 1)
        )
        
    def forward(self, users, items, context_features):
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)
        context_embedding = self.context_embedding(context_features)
        
        user_item_concat = torch.cat([user_embedding, item_embedding], dim=1)
        user_item_context_concat = torch.cat([user_item_concat, context_embedding], dim=1)
        
        out = self.dropout(user_item_context_concat)
        out = self.mlp_layers(out)
        return out.squeeze()
    
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for users, items, context_features, targets in dataloader:
        users, items, context_features, targets = users.to(device), items.to(device), context_features.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(users, items, context_features)
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * users.shape[0]
    return total_loss / len(dataloader.dataset)
    
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for users, items, context_features, targets in dataloader:
            users, items, context_features, targets = users.to(device), items.to(device), context_features.to(device), targets.to(device)
            outputs = model(users, items, context_features).squeeze()
            loss = criterion(outputs, targets.float())
            total_loss += loss.item() * users.shape[0]
    return total_loss / len(dataloader.dataset)
    
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        val_loss = validate(model, val_dataloader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.5f} | Validation Loss: {val_loss:.5f}")
    return train_losses, val_losses

data = pd.read_csv("./dataset/frappe/frappe.inter", sep = ',')
# data = pd.read_csv('/scratch/mvongala/Deepcarskit/dataset/tripadvisor/tripadvisor.inter', sep = ',')

# cols = ["user_id", "item_id", "rating"]
# data = data.iloc[:, :3]
data = data.dropna(axis = 0).reset_index(drop = True)

user_encoder = {user_id: i for i, user_id in enumerate(data['user:token'].unique())}
item_encoder = {item_id: i for i, item_id in enumerate(data['item:token'].unique())}

data['user_id'] = data['user:token'].apply(lambda x: user_encoder[x])
data['item_id'] = data['item:token'].apply(lambda x: item_encoder[x])

context_cols = ["daytime:token","weekday:token","isweekend:token","homework:token","cost:token","weather:token","country:token","city:token"]
for col in context_cols:
    col_encoder = {col_id: i for i, col_id in enumerate(data[col].unique())}
    data[col] = data[col].apply(lambda x: col_encoder[x])

# Generate some dummy data
num_samples = data.shape[0]
num_users = len(user_encoder)
num_items = len(item_encoder)
num_context_features = 8
embedding_dim = 16
hidden_units = 32
dropout_rate = 0.5

users = data['user_id']
items = data['item_id']
context_features = data[context_cols]
targets = data['cnt:float']

# Convert above four variables into torch tensors
users = torch.tensor(users, dtype=torch.long)
items = torch.tensor(items, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.float)
context_features = torch.tensor(context_features.values, dtype=torch.float)

# Split data into train and validation sets
train_size = int(0.8 * num_samples)
train_users, val_users = torch.split(users, [train_size, num_samples-train_size])
train_items, val_items = torch.split(items, [train_size, num_samples-train_size])
train_context_features, val_context_features = torch.split(context_features, [train_size, num_samples-train_size])
train_targets, val_targets = torch.split(targets, [train_size, num_samples-train_size])

train_dataset = CustomDatasetWithContext(train_users, train_items, train_context_features, train_targets)
val_dataset = CustomDatasetWithContext(val_users, val_items, val_context_features, val_targets)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)

# Create the model, criterion, and optimizer
model = NCFWithContext(num_users, num_items, num_context_features, embedding_dim, hidden_units, dropout_rate)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
num_epochs = 10
train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs)