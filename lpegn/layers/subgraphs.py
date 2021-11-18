import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.utils.subgraph import k_hop_subgraph, subgraph
from torch_geometric.data import Data, Batch
from torch_geometric.nn.norm import GraphNorm
from torch_scatter import gather_csr, scatter, segment_csr

def get_subgraphs(data, edge_attr, x1, subgraph_degrees, k):
    ## Get the sub graphs
    edge_index_src = data.edge_index_orig[0]
    edge_index_dst = data.edge_index_orig[1]

    edge_indices = {}
    node_indices = {}
    sub_graphs = {}
    batch_save = {}
    sub_nodes = {}
    for deg in subgraph_degrees:
        edge_indices[f'{deg}'] = []
        sub_graphs[f'{deg}'] = []
        batch_save[f'{deg}'] = []
        node_indices[f'{deg}'] = []
        sub_nodes[f'{deg}'] = []

    for i in range(data.num_nodes):
        ## Graph level subgraph - matrix features - rho_2
        if k == 1:
            node_indices_temp = edge_index_dst[edge_index_src==i]
        elif k==2:
            node_indices_temp_1 = edge_index_dst[edge_index_src==i]
            node_indices_temp = []
            for ni1 in node_indices_temp_1:
                node_indices_temp.append(edge_index_dst[edge_index_src==ni1])
            node_indices_temp = torch.cat(node_indices_temp, 0)
            node_indices_temp = torch.unique(node_indices_temp)

        edge_index_new, edge_attr_new = subgraph(node_indices_temp, data.edge_index, 
                                                 edge_attr=edge_attr, relabel_nodes=True)

        sub_graphs[f'{len(node_indices_temp)}'].append(Data(edge_index=edge_index_new, edge_attr=edge_attr_new, 
                                                       num_nodes=len(node_indices_temp)))

        edge_index_k = []
        node_index_k = []
        for idxi in node_indices_temp:
            for idxj in node_indices_temp:
                edge_index_k.append([idxi,idxj])
                if idxi==idxj:
                    node_index_k.append(idxi)
        edge_index_k = torch.LongTensor(edge_index_k).t().to(data.edge_index_orig.device)
        node_index_k = torch.LongTensor(node_index_k).to(data.edge_index_orig.device)
        edge_indices[f'{len(node_indices)}'].append(edge_index_k)
        node_indices[f'{len(node_indices)}'].append(node_index_k)
        batch_save[f'{len(node_indices)}'].append(data.batch[i])

        ## Node level subgraph - vector features - rho_1
        if x1 is not None:
            sub_nodes[f'{len(node_indices_temp)}'].append(x1[node_indices_temp])

    adj_out = []
    for i in subgraph_degrees:
        if not sub_graphs[f'{i}']:
            adj_out.append(None)
        else:
            data_temp = Batch.from_data_list(sub_graphs[f'{i}'])
            adj_temp = to_dense_adj(data_temp.edge_index, edge_attr=data_temp.edge_attr, batch=data_temp.batch)
            adj_out.append(adj_temp.permute(0,3,1,2))


    x_out = []
    for i in subgraph_degrees:
        if not sub_nodes[f'{i}']:
            x_out.append(None)
        else:
            x_out.append(torch.stack(sub_nodes[f'{i}'], dim=0).permute(0,2,1))

    return adj_out, edge_indices, x_out, node_indices, data, batch_save


def subgraphs_to_graph(data, adj_list, edge_indices_orig, subgraph_degrees):
    ## Put back into 1 sparse graph

    edge_index_list = []
    for i in subgraph_degrees:
        if edge_indices_orig[f'{i}']:
            edge_index_list.append(torch.cat(edge_indices_orig[f'{i}'], dim=-1))
        else:
            edge_index_list.append(torch.LongTensor().to(data.dev))


    edge_index = torch.cat(edge_index_list, dim=-1)

    index = edge_index[0] * data.num_nodes + edge_index[1]
    index_unique = torch.unique(index)

    i = 0
    for idx_unique in index_unique:
        index[index==idx_unique] = i
        i+=1

    adj_out = []
    for adj in adj_list:
        if adj is not None:
            adj_out.append(adj.permute(0,2,3,1).reshape(-1, adj.shape[1]))
        else:
            adj_out.append(torch.FloatTensor().to(data.dev))


    edge_attr = torch.cat(adj_out, dim=0)
    edge_attr = scatter(edge_attr, index, dim=0, dim_size=len(index_unique), reduce="mean")

    edge_index = scatter(edge_index, index, dim=1, dim_size=len(index_unique), reduce="max")

    return edge_index, edge_attr

def subnodes_to_nodes(data, x_list, node_indices_orig, subgraph_degrees):
    ## Put back into 1 sparse graph

    node_indices = []
    for i in subgraph_degrees:
        if node_indices_orig[f'{i}']:
            node_indices.append(torch.cat(node_indices_orig[f'{i}'], dim=-1))
        else:
            node_indices.append(torch.LongTensor().to(data.dev))
    node_indices = torch.cat(node_indices, dim=-1)

    index = node_indices
    index_unique = torch.unique(index)

    x_out = []
    for x in x_list:
        if x is not None:
            x_out.append(x.reshape(-1, x.shape[1]))
        else:
            x_out.append(torch.FloatTensor().to(data.dev))

    x = torch.cat(x_out, dim=0)

    x = scatter(x, node_indices, dim=0, dim_size=len(index_unique), reduce="mean")

    node_indices = scatter(node_indices, node_indices, dim=0, dim_size=len(index_unique), reduce="max")
    ## End put back into 1 sparse graph

    return node_indices, x