from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import Chem

def calculate_tanimoto_similarities(target_smiles, smiles_list, radius=2, n_bits=1024):
    # Convert target SMILES to molecule and generate fingerprint
    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is None:
        raise ValueError("Invalid target SMILES string.")
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, radius, nBits=n_bits)
    
    similarities = []
    for smiles in smiles_list:
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            similarities.append(None)
            continue
        # Generate fingerprint and calculate similarity
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        similarity = DataStructs.TanimotoSimilarity(target_fp, fp)
        similarities.append(similarity)
    return similarities

# Calculate Tanimoto similarities
target_smiles = 'C1=CC=C(C=C1)N'  # Aniline

import torch
import pandas as pd
data = torch.load('embeddings_contrastive.pt')
#print(data.keys())
smiles_list = data['labels']

similarities = calculate_tanimoto_similarities(target_smiles, smiles_list)

# Create DataFrame
df = pd.DataFrame({'smile': smiles_list, 'similarity': similarities})
df = df.sort_values('similarity', ascending=False)
print(df.head())

# Save DataFrame to CSV file

df.to_csv('tanimoto_similarities.csv', index=False)