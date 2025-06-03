from torch_geometric.nn import GATConv
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch
import networkx as nx
import os
from rdkit import Chem
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
import torch
from torch_geometric.data import Data
import networkx as nx
import json
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.nn import SAGPooling
import numpy as np

with open('grafos.json', 'r') as f:
    graph_data = json.load(f)

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

    
def get_vector(data):
    data.to(device)
    model.eval()
    embedding = model(data.x, data.edge_index, data.batch)
    return embedding.detach().cpu().numpy()
    
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

def get_graph(mol):
    adj = Chem.GetAdjacencyMatrix(mol)
    nodesym = [atom.GetSymbol() for atom in mol.GetAtoms()]
    bond_type = [bond.GetBondType() for bond in mol.GetBonds()]
    smile = Chem.MolToSmiles(mol)
    G = nx.Graph(adj, symbol=nodesym, bond_type=bond_type, smile=smile)
    return G

class CustomGraphDataset(Dataset):
    def __init__(self, data_list):
        super(CustomGraphDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

import pandas as pd
df = pd.DataFrame(columns=['distance', 'mahalanobis', 'graph', 'smile', 'vectors'])

print(nx.node_link_graph(graph_data[0]).graph['smile'])

# Lista: {distance, graph, smile}
pyg_data_list = []
for graph in graph_data:
    #pyg_data_list.append(nx_to_pyg(nx.node_link_graph(graph)))
    pyg_data_list.append(nx_to_pyg(nx.node_link_graph(graph)))
    #df.append({'distance': 0, 'graph': nx_to_pyg(nx.node_link_graph(graph)), 'smile': nx.node_link_graph(graph_data[0]).graph['smile']}, ignore_index=True)
    #pd.concat([df, pd.DataFrame({'distance': 0, 'graph': nx_to_pyg(nx.node_link_graph(graph)), 'smile': nx.node_link_graph(graph).graph['smile']})])

df = pd.DataFrame({'distance': [0]*len(pyg_data_list), 'mahalanobis': [0]*len(pyg_data_list), 'graph': pyg_data_list, 'smile': [nx.node_link_graph(graph).graph['smile'] for graph in graph_data], 'vectors': [0]*len(pyg_data_list)})

print(df.head())
print(df.count())

dataset = torch.load('dataset.pt')
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Cargar el modelo
model = ContrastiveGNN(input_dim=dataset.num_node_features, hidden_dim=256, embedding_dim=256)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(torch.load('model_contrastive_256.pth'))
model.eval()

# Sentence2vec
#target = Chem.MolFromSmiles('OC1=CNC=N1')
#target = Chem.MolFromSmiles('Oc1cnc[nH]1')
#target = Chem.MolFromSmiles('Cc1nn(O)c(=NN)[nH]1')
target = Chem.MolFromSmiles('C1=CC=C(C=C1)N') # Aniline
G = get_graph(target)

vector = get_vector(nx_to_pyg(G))
#print(vector)
#print(vector.shape)

# Calcular la distancia euclidiana
from scipy.spatial.distance import euclidean

from tqdm import tqdm
print("Obteniendo vectores")
# Exclude first four graphs
df = df.iloc[4:]
"""vectors = []
progress_bar = tqdm(total=len(df),
                    desc="Calculando vectores",
                    leave=True,
                    colour='green',
                    dynamic_ncols=True)
for index, row in df.iterrows():
    data = row['graph']
    vector2 = get_vector(data)
    vectors.append(vector2.flatten())
    progress_bar.set_postfix({"Index": index, "SMILES": row['smile']})
    progress_bar.update(1)
progress_bar.close()"""
    #vector2 = get_vector(row['graph'])
    #vectors.append(vector2.flatten())

batch_size = 128
dataset = [(row['graph'], row['smile']) for _, row in df.iterrows()]
vectors = []

progress_bar = tqdm(total=len(dataset),
                    desc="GPU Processing",
                    leave=True,
                    colour='green',
                    dynamic_ncols=True)

for batch in DataLoader(dataset, batch_size=batch_size):
    graph_batch, smiles_batch = batch
    batch_vectors = get_vector(graph_batch)
    vectors.extend([v.flatten() for v in batch_vectors])
    progress_bar.update(len(graph_batch))

progress_bar.close()
    
df['vectors'] = vectors

print("Calculando distancias euclidianas")
euclidean_dists = np.linalg.norm(vectors - vector, axis=1)
df['distance'] = euclidean_dists

#save to csv
#df.to_csv('distancias_1.csv')

import maha

print("Calculando distancias de Mahalanobis")
# Calcular la distancia de mahalanobis
matrix = vectors
#print(matrix)
cov_matrix = maha.cov_standard(matrix)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
diag_matrix = np.diag(eigenvalues)
VI = np.linalg.inv(diag_matrix)

embeddings = matrix

delta = embeddings - vector.flatten()
maha_dists = np.sqrt(np.einsum('ni,ij,nj->n', delta, VI, delta))

df['mahalanobis'] = maha_dists

print("Guardando resultados")

#save to csv
df.to_csv('distancias_contrastive_256.csv')

torch.save({
    'embeddings': vectors,
    'labels': df['smile']
}, 'embeddings_contrastive.pt')

