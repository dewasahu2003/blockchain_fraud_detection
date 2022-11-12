import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# in data 1->invalid,2->valid,unknown->unknown
classes = pd.read_csv("./elliptic_bitcoin_dataset/elliptic_txs_classes.csv")
edges = pd.read_csv("./elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")
features = pd.read_csv(
    "./elliptic_bitcoin_dataset/elliptic_txs_features.csv", header=None
)

# STEP0:preparing node==================================================

# 0->valid 1->invalid 2->unknown
classes["class"] = classes["class"].map({"unknown": 2, "1": 1, "2": 0})
df_merge = pd.merge(left=features, right=classes, left_on=0, right_on="txId")
df_merge: pd.DataFrame = df_merge.sort_values(0).reset_index(drop=True)


# classified node -> class!=2
classified_node = df_merge[df_merge["class"] != 2].drop("txId", axis=1)
unclassified_node = df_merge[df_merge["class"] == 2].drop("txId", axis=1)


# classified edges -> class!=2
classified_edges = edges[
    edges["txId1"].isin(classified_node[0]) & edges["txId2"].isin(classified_node[0])
]

unclassified_edges = edges[
    edges["txId1"].isin(unclassified_node[0])
    & edges["txId2"].isin(unclassified_node[0])
]

# STEP1:preparing edges=================================================
node_id = df_merge[0].values
map_id = {j: i for i, j in enumerate(node_id)}
# mapping id to index

edges["txId1"] = edges["txId1"].map(map_id)
edges["txId2"] = edges["txId2"].map(map_id)

edge_index = np.array(edges).T
edge_index = torch.tensor(edge_index, dtype=torch.long)
weights = torch.tensor([1] * edge_index.shape[1], dtype=torch.long)

# STEP2:preparing node-feature===========================================

node_feature = df_merge.drop("txId", axis=1)
node_feature[0] = node_feature[0].map(map_id)

# classified node_feature
classifed_idx = node_feature[node_feature["class"] != 2].index
unclassified_idx = node_feature[node_feature["class"] == 2].index

# ....🤔
node_feature["class"] = node_feature["class"].replace(2, 0)

labels = torch.tensor(node_feature["class"].values,dtype=torch.float32)

node_feature = torch.tensor(
    np.array(node_feature.drop([0, "class", 1], axis=1)), dtype=torch.float32
)

# STEP3:prepare the dataset================================================

elliptical_data = Data(
    x=node_feature, edge_index=edge_index, edge_attr=weights, y=labels
)


y_train = labels[classifed_idx]

_, _, _, _, train_idx, val_idx = train_test_split(
    node_feature[classifed_idx],
    y_train,
    classifed_idx,
    test_size=0.15,
    random_state=0,
    stratify=y_train,
)

elliptical_data.train_idx = torch.tensor(train_idx, dtype=torch.long)
elliptical_data.val_idx = torch.tensor(val_idx, dtype=torch.long)
elliptical_data.test_idx = torch.tensor(unclassified_idx, dtype=torch.long)
