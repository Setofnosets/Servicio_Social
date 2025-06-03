import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MNISTSuperpixels
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

dataset = MNISTSuperpixels(root='.')

embedding_size = 64
class GCN(torch.nn.Module):
    def __init__(self):
        # Init
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        # GCN layers
        self.initial_conv = GCNConv(dataset.num_node_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        # Output layer
        self.out = Linear(embedding_size*2, dataset.num_classes)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)

        # Others
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)

        # Global Pooling
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)

        # Final
        out = self.out(hidden)
        return out, hidden

def get_vector(data):
    data.to(device)
    _, embedding = model(data.x, data.edge_index, data.batch)
    return embedding.detach().cpu().numpy()

model = GCN()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

data_size = len(dataset)
batch_size = 32

loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=batch_size, shuffle=False)

def train(data):
    for batch in loader:
        batch.to(device)
        optimizer.zero_grad()
        pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = torch.sqrt(loss_fn(pred, batch.y))
        loss.backward()
        optimizer.step()
    return loss, embedding

print("Comienzo del entrenamiento")
embeddings = []
losses = []
for epoch in range(100):
    loss, embedding = train(dataset)
    embeddings.append(embedding)
    losses.append(loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss {loss}")

embedding = embeddings[-1].detach().cpu().numpy()
print(embedding.shape)

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Load the model
model = GCN()
model.load_state_dict(torch.load('model.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

data_size = len(dataset)
batch_size = 32
loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=batch_size, shuffle=True)

#Get embedding
from tqdm.notebook import tqdm
train_results = []
labels = []
with torch.no_grad():
    for G in tqdm(loader):
        _, embedding = model(G.x.float(), G.edge_index, G.batch)
        train_results.append(embedding)
        labels.append(G.y)

import numpy as np
#train_results = np.array(train_results)

train_results = np.concatenate(train_results)
labels = np.concatenate(labels)
print(train_results.shape)
print(labels.shape)

# Save csv
import pandas as pd

df = pd.DataFrame(train_results)
df_tmp = pd.DataFrame(labels)
df.insert(0, "labels", labels)
print(df.shape)

df.to_csv("vectors.csv", index=False)
