import torch
from torch_geometric.data import Data

#first list has src nodes and 2nd has dest nodes
edge_index = torch.Tensor([[0,1,1,2],[1,0,2,1]], dtype=torch.long)
x = torch.tensor([[-1],[0],[1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

print(data)

#if you want to write it in a list of src dest,
#call contiguous and transpose

edge_index = torch.Tensor([[0,1],[1,0],[1,2],[2,1]], dtype= torch.long)
data = Data(x=x,edge_index=edge_index)

print(data)
print(data.keys) #x, edge_index
print(data['x']) #tensor

for key,item in data:
    print(key)

print(data.num_nodes, data.num_edges, data.num_features)
print(data.contains_isolated_nodes(), data.contains_self_loops(), data.is_directed())

#transfer data object to GPU
device = torch.device('cuda')
data = data.to(device)


from torch_geometric.datasets import TUDataset














