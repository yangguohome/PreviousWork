import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MolStandardize
from rdkit.Chem.Draw import IPythonConsole

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
# Assume data has a processed structure like ['SMILES', 'IC50', 'LD50']
df_dir = ''
df = pd.read_csv(df_dir)

# Molecule Standardization
def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = MolStandardize.rdMolStandardize.Cleanup(mol)
        mol = MolStandardize.rdMolStandardize.StandardizeMol(mol)
        return Chem.MolToSmiles(mol)
    return None

df['standard_smiles'] = df['SMILES'].apply(standardize_smiles)
df.dropna(subset=['standard_smiles'], inplace=True)

# Convert molecules to graph data for GNN
def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    edge_index = [[], []]
    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum()])  

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index[0] += [i, j]
        edge_index[1] += [j, i]

    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

graph_data = [mol_to_graph(smi) for smi in df['standard_smiles']]
graph_data = [g for g in graph_data if g is not None]

# 4. GNN Encoder (basic example)
class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(a, b)
        self.conv2 = GCNConv(b, c)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return x

# Self-supervised representation learning
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN().to(device)
model.eval()

embeddings = []
for data in graph_data:
    data = data.to(device)
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(device)
    with torch.no_grad():
        emb = model(data.x, data.edge_index, data.batch)
    embeddings.append(emb.cpu().numpy().squeeze())

# UMAP Visualization
embedding_matrix = np.vstack(embeddings)
reducer = umap.UMAP()
embedding_2d = reducer.fit_transform(embedding_matrix)

plt.figure(figsize=(8,6))
plt.scatter(embedding_2d[:,0], embedding_2d[:,1], c=df['IC50'], cmap='viridis')
plt.colorbar(label='IC50')
plt.title("UMAP Projection of GNN Embeddings")
plt.show()

# Clustering
kmeans = KMeans(n_clusters=)
clusters = kmeans.fit_predict(embedding_matrix)
df['cluster'] = clusters

# Regression per cluster (IC50 and LD50)
results = {}
for c in df['cluster'].unique():
    idx = df['cluster'] == c
    X = embedding_matrix[idx]
    y_ic50 = df.loc[idx, 'IC50']
    y_ld50 = df.loc[idx, 'LD50']

    model_ic50 = LinearRegression().fit(X, y_ic50)
    model_ld50 = LinearRegression().fit(X, y_ld50)

    y_ic50_pred = model_ic50.predict(X)
    y_ld50_pred = model_ld50.predict(X)

    results[c] = {
        'IC50_MSE': mean_squared_error(y_ic50, y_ic50_pred),
        'LD50_MSE': mean_squared_error(y_ld50, y_ld50_pred)
    }

# Visualize performance
plt.figure(figsize=(10,5))
for i, metric in enumerate(['IC50_MSE', 'LD50_MSE']):
    plt.subplot(1,2,i+1)
    plt.bar(results.keys(), [v[metric] for v in results.values()])
    plt.title(metric + ' per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('MSE')

plt.tight_layout()
plt.show()
