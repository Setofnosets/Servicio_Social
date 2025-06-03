import torch
import networkx as nx
import os
from rdkit import Chem
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import from_networkx
import torch
from torch_geometric.data import Data
import networkx as nx
import json
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import numpy as np

with open('grafos.json', 'r') as f:
    graph_data = json.load(f)

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

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)

        # Final
        out = self.out(hidden)
        return out, hidden

def get_vector(data):
    data.to(device)
    _, embedding = model(data.x, data.edge_index, data.batch)
    return embedding.detach().cpu().numpy()

def get_graph(mol):
    adj = Chem.GetAdjacencyMatrix(mol)
    nodesym = [atom.GetSymbol() for atom in mol.GetAtoms()]
    bond_type = [bond.GetBondType() for bond in mol.GetBonds()]
    G = nx.Graph(adj, symbol=nodesym, bond_type=bond_type)
    return G

def nx_to_pyg(graph):
    # Extract node features (symbols)
    node_features = torch.tensor([[ord(symbol)] for symbol in graph.graph['symbol']], dtype=torch.float)
    
    # Extract edge indices
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    # Extract edge features (bond types)
    edge_attr = torch.tensor([bond_type for bond_type in graph.graph['bond_type']], dtype=torch.float).view(-1, 1)
    
    # Create PyG Data object
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

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

dataset = torch.load('dataset_o.pt')
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Cargar el modelo
model = GCN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Sentence2vec
#target = Chem.MolFromSmiles('OC1=CNC=N1')
#target = Chem.MolFromSmiles('Oc1cnc[nH]1')
target = Chem.MolFromSmiles('C1=CC=C(C=C1)N')
G = get_graph(target)

vector = get_vector(nx_to_pyg(G))
#print(vector)
#print(vector.shape)

# Calcular la distancia euclidiana
from scipy.spatial.distance import euclidean
import maha

print("Obteniendo vectores")

from tqdm import tqdm
# Exclude first four graphs
df = df.iloc[4:]
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
    batch_vectors = get_vector(graph_batch)  # Batched GPU implementation
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

# load csv
#df = pd.read_csv('distancias.csv')

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

"""for index, row in df.iterrows():
    #vec2 = row['vectors']
    vec2 = vectors[index - 4] # El df empieza en 4
    vec2 = np.array(vec2)
    vec2_1d = vec2.reshape(-1)
    #print(vec2_1d)
    dist = maha.mahalanobis(vector.flatten(), vec2_1d, np.linalg.inv(cov_matrix))
    df.at[index, 'mahalanobis'] = dist"""

#save to csv
df.to_csv('distancias_1.csv')
