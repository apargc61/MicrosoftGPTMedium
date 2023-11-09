import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Define a simple GCN model
class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Create a simple graph dataset
edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
print(edges)
x = torch.randn(3, 4)  # Node features (3 nodes, 4-dimensional features)
print(x)
y = torch.tensor([0, 1, 2], dtype=torch.long)  # Node labels (3 nodes, 3 classes)
print(y)
data = Data(x=x, edge_index=edges, y=y)
print(data)
# Create a train mask (randomly selecting nodes for training)
train_mask = torch.tensor([True, False, True])  # Example: Node 0 and Node 2 for training
train_mask

# Initialize and train the GCN model
num_features = x.shape[1]
hidden_dim = 16
num_classes = torch.unique(y).size(0)
model = GCN(num_features, hidden_dim, num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    logits = model(data)
    predicted = logits.argmax(dim=1)
    test_mask = ~train_mask  # Use the remaining nodes for testing
    acc = (predicted[test_mask] == data.y[test_mask]).sum().item() / test_mask.sum().item()
    print(f'Test accuracy: {acc:.4f}')
