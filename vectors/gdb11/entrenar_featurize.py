from mol2vec import features
from mol2vec.helpers import depict_identifier
from rdkit import Chem
from openbabel import openbabel
import os
import pandas as pd
import gensim as gs

# Generar corpus
def corpus():
    archivos = os.listdir('mol2vec\gdb11\gdb11')
    corpus = []

    # Leer archivos y generar corpus
    for archivo in archivos:
        print(archivo)
        if archivo != "gdb11_size11.smi":
            os.system(f"mol2vec corpus -i mol2vec\gdb11\gdb11\{archivo} -o mol2vec\gdb11\out.cp -r 1 -j 24 --uncommon UNK --threshold 3")     
            f = open("mol2vec\gdb11\out.cp_UNK", 'r')
        else:
            # Corpus generado previamente por su tamaño
            f = open("mol2vec\gdb11\out_o.cp_UNK", 'r')        
        corpus.append(f.read())

    # Unir corpus
    f = open("mol2vec\gdb11\corpus.cp", 'x')
    f.write("\n".join(corpus))
    f.close()

# Entrenar modelo
def train():
    os.system(f"mol2vec train -i mol2vec\gdb11\corpus.cp -o mol2vec\gdb11\model.plk -d 300 -w 10 -m skip-gram --threshold 3 -j 24")

# Featurizar
def featurize():
    # Featurizar (10 y 11 es demasiado grande)
    for i in range(1, 9):
        os.system(f"mol2vec featurize -i mol2vec\gdb11\gdb11\gdb11_size{i:02d}.smi -o mol2vec\gdb11\out{i:02d}.csv -m mol2vec\gdb11\model.plk -r 1")