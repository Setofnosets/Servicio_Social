import torch
import os
from rdkit import Chem

from torch_geometric.utils import from_networkx

import torch
from torch_geometric.data import Data
import networkx as nx

def atom_features(atom):
    return [
        atom.GetAtomicNum(),  # Atomic number
        atom.GetDegree(),      # Number of bonded neighbors
        atom.GetHybridization(),
        atom.GetIsAromatic()
    ]

# No supervisado
def nx_to_pyg(graph):
    # Extract node features (symbols)
    #node_features = torch.tensor([[ord(symbol)] for symbol in graph.graph['symbol']], dtype=torch.float)
    node_features = torch.tensor([atom_features(atom) for atom in Chem.MolFromSmiles(graph.graph['smile']).GetAtoms()], dtype=torch.float)
    # Extract edge indices
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    # Extract edge features (bond types)
    edge_attr = torch.tensor([bond_type for bond_type in graph.graph['bond_type']], dtype=torch.float).view(-1, 1)
    
    # Create PyG Data object
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


# Convert all graphs to PyG format
#pyg_data_list = [nx_to_pyg(nx.node_link_graph(graph)) for graph in graph_data]

from torch_geometric.data import Dataset

class CustomGraphDataset(Dataset):
    def __init__(self, data_list):
        super(CustomGraphDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

# Create the dataset
dataset = torch.load('dataset.pt')

# Print length of the dataset
print(len(dataset))

from torch_geometric.loader import DataLoader

dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

from torch_geometric.nn import GATConv
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import SAGPooling

class ContrastiveGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, pool_ratio=0.5):
        super(ContrastiveGNN, self).__init__()
        # Convolutional layers with skip connections
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        
        # Hierarchical pooling
        self.pool = SAGPooling(hidden_dim, ratio=pool_ratio)
        
        # Final layers
        self.fc = torch.nn.Linear(hidden_dim*2, embedding_dim)
    
    def forward(self, x, edge_index, batch):
        # Layer 1
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        
        # Layer 2 with skip connection
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2 + x1)  # Skip connection
        
        # Pooling
        x_pool, edge_index_pool, _, batch_pool, _, _ = self.pool(x2, edge_index, batch=batch)
        
        # Final embedding
        #x_embed = global_mean_pool(x_pool, batch_pool)
        x_embed = torch.cat([gmp(x_pool, batch_pool), gap(x_pool, batch_pool)], dim=1)
        out = self.fc(x_embed)
        return x_embed

# Contrastive loss function
def contrastive_loss(embeddings, temperature=0.1):
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    
    # Compute contrastive loss
    loss = -torch.log(torch.diag(sim_matrix).exp() / sim_matrix.exp().sum(dim=1))
    return loss.mean()

from tqdm import tqdm

# Training loop for contrastive learning
def train_contrastive(epochs):
    model.train()
    for epoch in range(epochs):
        # Initialize progress bar for the current epoch
        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=True,
            colour='green',
            dynamic_ncols=True 
        )

        total_loss = 0

        for batch_idx, batch in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()
            embeddings = model(batch.x, batch.edge_index, batch.batch)
            loss = contrastive_loss(embeddings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar description with current batch loss
            progress_bar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        # Close the progress bar for this epoch
        progress_bar.close()

        # Print epoch summary
        avg_epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Completed | Avg Loss: {avg_epoch_loss:.4f}\n")

# Initialize the model
model = ContrastiveGNN(input_dim=dataset.num_node_features, hidden_dim=256, embedding_dim=256)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, eps=1e-8, betas=(0.9, 0.999), weight_decay=1e-5) #Added 0.0001

print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Train the model
train_contrastive(epochs=15)

"""embeddings = []
for batch in dataloader:
    model.eval()
    batch = batch.to(device)
    with torch.no_grad():
        embeddings.append(model(batch.x, batch.edge_index, batch.batch))

embedding = embeddings[-1].detach().cpu().numpy()
print(embedding)
print(embedding.shape)
print(len(embedding))
"""
# Save the model
torch.save(model.state_dict(), 'model_contrastive_256.pth')
# Run distancias_contrastive.py

import os 

os.system('python distancias_contrastive.py')