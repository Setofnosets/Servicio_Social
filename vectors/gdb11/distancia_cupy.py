from mol2vec import features
from mol2vec.helpers import depict_identifier
from rdkit import Chem
import os
import pandas as pd
import gensim as gs
import numpy as np

df = pd.read_csv('distancias.csv')
print(df.head())
print(df.count())
print(df.dtypes)

# get vectors
matrix = df['vectors'].values
# Remove newline characters
for i in range(len(matrix)):
    matrix[i] = matrix[i].replace('\n', '')
    matrix[i] = matrix[i].replace('[', '')
    matrix[i] = matrix[i].replace(']', '')
    matrix[i] = matrix[i].split()
    matrix[i] = [float(x) for x in matrix[i]]
print(matrix)

cov_matrix = np.cov(matrix.T)
#print(cov_matrix)

# Standarizar
mean = np.mean(matrix, axis=0)
std = np.std(matrix, axis=0)
matrix = (matrix - mean) / std
cov_matrix = np.cov(matrix.T)
#print(cov_matrix)

# Calcular eigenvectores y eigenvalores
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
#print(eig_vals)

# Diagonal matrix
diag_matrix = np.diag(eig_vals)
print(diag_matrix)

# Reconstruct the matrix
reconstructed_matrix = eig_vecs.dot(diag_matrix).dot(eig_vecs.T)
#print(reconstructed_matrix)

molecula = Chem.MolFromSmiles('OC1=CNC=N1')
#molecula = Chem.MolFromSmiles('CC1CC(=O)N1')
sentence = features.mol2alt_sentence(molecula, 1)
model = gs.models.Word2Vec.load('model.plk')
vec = features.sentences2vec([sentence], model) #NOTA: mol2vec utiliza la suma de los vectores de las moleculas, no el promedio
vec = np.array(vec)
vec_1d = vec.reshape(-1)
print(vec_1d.shape)
print(vec_1d.dtype)

#Implementación de scipy.spatial.distance.mahalanobis con Cupy
import cupy as cp
def mahalanobis_CU(u, v, VI):
    u = cp.asarray(u)
    v = cp.asarray(v)
    VI = np.atleast_2d(VI)
    VI = cp.asarray(VI)
    delta = cp.subtract(u, v)
    m = cp.dot(cp.dot(delta, VI), delta)
    return cp.sqrt(m).get()

"""from scipy.spatial.distance import mahalanobis
for index, row in df.iterrows():
    vec2 = row[3:]
    vec2 = np.array(vec2)
    vec2_1d = vec2.reshape(-1)

    #Distancia de mahalanobis
    dist = mahalanobis(vec_1d, vec2_1d, np.linalg.inv(cov_matrix))
    df.at[index, 'mahalanobis'] = dist

    #Distancia euclidiana
    dist = np.linalg.norm(vec - vec2)
    df.at[index, 'euclidean'] = dist

    #Distancia de mahanobis diagonal
    dist = mahalanobis(vec_1d, vec2_1d, np.linalg.inv(diag_matrix))
    df.at[index, 'mahalanobis_diag'] = dist"""

for index, row in df.iterrows():
    vec2 = row[3:]
    vec2 = np.array(vec2)
    vec2_1d = vec2.reshape(-1)
    vec2_1d = vec2_1d.astype(float)
    #print(vec2_1d.dtype)
    #print(vec2_1d.shape)

    #Distancia de mahalanobis
    dist = mahalanobis_CU(vec_1d, vec2_1d, np.linalg.inv(cov_matrix))
    df.at[index, 'mahalanobis'] = dist

    #Distancia euclidiana
    dist = np.linalg.norm(vec - vec2)
    df.at[index, 'euclidean'] = dist

    #Distancia de mahanobis diagonal
    dist = mahalanobis_CU(vec_1d, vec2_1d, np.linalg.inv(diag_matrix))
    df.at[index, 'mahalanobis_diag'] = dist

#Crear archivo de salida
file = open('out2_df.csv', 'w')
file.write(df.to_csv(index=False))
file.close()