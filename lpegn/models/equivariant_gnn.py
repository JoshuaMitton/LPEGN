from models.base_model import BaseModel
import layers.equivariant_linear as eq
import layers.layers as layers
import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.utils.subgraph import k_hop_subgraph, subgraph
from torch_geometric.data import Data, Batch
from torch_geometric.nn.norm import GraphNorm
from torch_scatter import gather_csr, scatter, segment_csr

class equivariant_gnn(BaseModel):
    def __init__(self, config, weight_degrees, subgraph_degrees, degree_mapping):
        super(equivariant_gnn, self).__init__(config)
        print(f'model build for dataset : {config.dataset_name}')

        self.architecture = config.architecture
        self.BN = config.BN
        self.DO = config.DO
        self.residual = config.residual

        self.is_training = torch.autograd.Variable(torch.ones(1, dtype=torch.bool))
        
        self.weight_degrees = weight_degrees
        self.subgraph_degrees = subgraph_degrees
        self.degree_mapping = degree_mapping
        
        if len(self.architecture) > 0:
            eqblock1_p2p2_list = []
            for i in self.weight_degrees:
                eqblock1_p2p2_list.append([f'{i}', eq.equi_2_to_2(config.num_features_in, 2*self.architecture[0], 'cuda')])
            self.eqblock1_p2p2 = torch.nn.ModuleDict(eqblock1_p2p2_list)
            eqblock1_p2p1_list = []
            for i in self.weight_degrees:
                eqblock1_p2p1_list.append([f'{i}', eq.equi_2_to_1(config.num_features_in, 2*self.architecture[0], 'cuda')])
            self.eqblock1_p2p1 = torch.nn.ModuleDict(eqblock1_p2p1_list)
            eqblock1_p1p2_list = []
            for i in self.weight_degrees:
                eqblock1_p1p2_list.append([f'{i}', eq.equi_1_to_2(config.num_features_in, 2*self.architecture[0], 'cuda')])
            self.eqblock1_p1p2 = torch.nn.ModuleDict(eqblock1_p1p2_list)
            eqblock1_p1p1_list = []
            for i in self.weight_degrees:
                eqblock1_p1p1_list.append([f'{i}', eq.equi_1_to_1(config.num_features_in, 2*self.architecture[0], 'cuda')])
            self.eqblock1_p1p1 = torch.nn.ModuleDict(eqblock1_p1p1_list)
       
        if len(self.architecture) > 1:
            eqblock2_p2p2_list = []
            for i in self.weight_degrees:
                eqblock2_p2p2_list.append([f'{i}', eq.equi_2_to_2(2*self.architecture[0], self.architecture[1], 'cuda')])
            self.eqblock2_p2p2 = torch.nn.ModuleDict(eqblock2_p2p2_list)
            eqblock2_p2p1_list = []
            for i in self.weight_degrees:
                eqblock2_p2p1_list.append([f'{i}', eq.equi_2_to_1(2*self.architecture[0], self.architecture[1], 'cuda')])
            self.eqblock2_p2p1 = torch.nn.ModuleDict(eqblock2_p2p1_list)
            eqblock2_p1p2_list = []
            for i in self.weight_degrees:
                eqblock2_p1p2_list.append([f'{i}', eq.equi_1_to_2(2*self.architecture[0], self.architecture[1], 'cuda')])
            self.eqblock2_p1p2 = torch.nn.ModuleDict(eqblock2_p1p2_list)
            eqblock2_p1p1_list = []
            for i in self.weight_degrees:
                eqblock2_p1p1_list.append([f'{i}', eq.equi_1_to_1(2*self.architecture[0], self.architecture[1], 'cuda')])
            self.eqblock2_p1p1 = torch.nn.ModuleDict(eqblock2_p1p1_list)
           
            self.elu1 = torch.nn.ELU()
           
            if self.BN:
                self.gn1p1 = GraphNorm(2*self.architecture[0])
                self.gn1p2 = GraphNorm(2*self.architecture[0])
               
            if self.DO:
                self.do1 = torch.nn.Dropout(p=0.01)

        if len(self.architecture) > 2:
            eqblock3_p2p2_list = []
            for i in self.weight_degrees:
                eqblock3_p2p2_list.append([f'{i}', eq.equi_2_to_2(2*self.architecture[1], self.architecture[2], 'cuda')])
            self.eqblock3_p2p2 = torch.nn.ModuleDict(eqblock3_p2p2_list)
            eqblock3_p2p1_list = []
            for i in self.weight_degrees:
                eqblock3_p2p1_list.append([f'{i}', eq.equi_2_to_1(2*self.architecture[1], self.architecture[2], 'cuda')])
            self.eqblock3_p2p1 = torch.nn.ModuleDict(eqblock3_p2p1_list)
            eqblock3_p1p2_list = []
            for i in self.weight_degrees:
                eqblock3_p1p2_list.append([f'{i}', eq.equi_1_to_2(2*self.architecture[1], self.architecture[2], 'cuda')])
            self.eqblock3_p1p2 = torch.nn.ModuleDict(eqblock3_p1p2_list)
            eqblock3_p1p1_list = []
            for i in self.weight_degrees:
                eqblock3_p1p1_list.append([f'{i}', eq.equi_1_to_1(2*self.architecture[1], self.architecture[2], 'cuda')])
            self.eqblock3_p1p1 = torch.nn.ModuleDict(eqblock3_p1p1_list)
           
            self.elu2 = torch.nn.ELU()
           
            if self.BN:
                self.gn2p1 = GraphNorm(2*self.architecture[1])
                self.gn2p2 = GraphNorm(2*self.architecture[1])

            if self.DO:
                self.do2 = torch.nn.Dropout(p=0.01)
           
        if len(self.architecture) > 3:
            eqblock4_p2p2_list = []
            for i in self.weight_degrees:
                eqblock4_p2p2_list.append([f'{i}', eq.equi_2_to_2(2*self.architecture[2], self.architecture[3], 'cuda')])
            self.eqblock4_p2p2 = torch.nn.ModuleDict(eqblock4_p2p2_list)
            eqblock4_p2p1_list = []
            for i in self.weight_degrees:
                eqblock4_p2p1_list.append([f'{i}', eq.equi_2_to_1(2*self.architecture[2], self.architecture[3], 'cuda')])
            self.eqblock4_p2p1 = torch.nn.ModuleDict(eqblock4_p2p1_list)
            eqblock4_p1p2_list = []
            for i in self.weight_degrees:
                eqblock4_p1p2_list.append([f'{i}', eq.equi_1_to_2(2*self.architecture[2], self.architecture[3], 'cuda')])
            self.eqblock4_p1p2 = torch.nn.ModuleDict(eqblock4_p1p2_list)
            eqblock4_p1p1_list = []
            for i in self.weight_degrees:
                eqblock4_p1p1_list.append([f'{i}', eq.equi_1_to_1(2*self.architecture[2], self.architecture[3], 'cuda')])
            self.eqblock4_p1p1 = torch.nn.ModuleDict(eqblock4_p1p1_list)
           
            self.elu3 = torch.nn.ELU()
           
            if self.BN:
                self.gn3p1 = GraphNorm(2*self.architecture[2])
                self.gn3p2 = GraphNorm(2*self.architecture[2])

            if self.DO:
                self.do3 = torch.nn.Dropout(p=0.01)

        if len(self.architecture) > 4:
            eqblock5_p2p2_list = []
            for i in self.weight_degrees:
                eqblock5_p2p2_list.append([f'{i}', eq.equi_2_to_2(2*self.architecture[3], self.architecture[4], 'cuda')])
            self.eqblock5_p2p2 = torch.nn.ModuleDict(eqblock5_p2p2_list)
            eqblock5_p2p1_list = []
            for i in self.weight_degrees:
                eqblock5_p2p1_list.append([f'{i}', eq.equi_2_to_1(2*self.architecture[3], self.architecture[4], 'cuda')])
            self.eqblock5_p2p1 = torch.nn.ModuleDict(eqblock5_p2p1_list)
            eqblock5_p1p2_list = []
            for i in self.weight_degrees:
                eqblock5_p1p2_list.append([f'{i}', eq.equi_1_to_2(2*self.architecture[3], self.architecture[4], 'cuda')])
            self.eqblock5_p1p2 = torch.nn.ModuleDict(eqblock5_p1p2_list)
            eqblock5_p1p1_list = []
            for i in self.weight_degrees:
                eqblock5_p1p1_list.append([f'{i}', eq.equi_1_to_1(2*self.architecture[3], self.architecture[4], 'cuda')])
            self.eqblock5_p1p1 = torch.nn.ModuleDict(eqblock5_p1p1_list)
           
            self.elu4 = torch.nn.ELU()
           
            if self.BN:
                self.gn4p1 = GraphNorm(2*self.architecture[3])
                self.gn4p2 = GraphNorm(2*self.architecture[3])

            if self.DO:
                self.do4 = torch.nn.Dropout(p=0.01)
                
        if len(self.architecture) > 6:
            eqblock51_p2p2_list = []
            for i in self.weight_degrees:
                eqblock51_p2p2_list.append([f'{i}', eq.equi_2_to_2(2*self.architecture[3], self.architecture[4], 'cuda')])
            self.eqblock51_p2p2 = torch.nn.ModuleDict(eqblock51_p2p2_list)
            eqblock51_p2p1_list = []
            for i in self.weight_degrees:
                eqblock51_p2p1_list.append([f'{i}', eq.equi_2_to_1(2*self.architecture[3], self.architecture[4], 'cuda')])
            self.eqblock51_p2p1 = torch.nn.ModuleDict(eqblock51_p2p1_list)
            eqblock51_p1p2_list = []
            for i in self.weight_degrees:
                eqblock51_p1p2_list.append([f'{i}', eq.equi_1_to_2(2*self.architecture[3], self.architecture[4], 'cuda')])
            self.eqblock51_p1p2 = torch.nn.ModuleDict(eqblock51_p1p2_list)
            eqblock51_p1p1_list = []
            for i in self.weight_degrees:
                eqblock51_p1p1_list.append([f'{i}', eq.equi_1_to_1(2*self.architecture[3], self.architecture[4], 'cuda')])
            self.eqblock51_p1p1 = torch.nn.ModuleDict(eqblock51_p1p1_list)
           
            self.elu41 = torch.nn.ELU()
           
            if self.BN:
                self.gn41p1 = GraphNorm(2*self.architecture[3])
                self.gn41p2 = GraphNorm(2*self.architecture[3])

            if self.DO:
                self.do41 = torch.nn.Dropout(p=0.01)
                
        if len(self.architecture) > 7:
            eqblock52_p2p2_list = []
            for i in self.weight_degrees:
                eqblock52_p2p2_list.append([f'{i}', eq.equi_2_to_2(2*self.architecture[3], self.architecture[4], 'cuda')])
            self.eqblock52_p2p2 = torch.nn.ModuleDict(eqblock52_p2p2_list)
            eqblock52_p2p1_list = []
            for i in self.weight_degrees:
                eqblock52_p2p1_list.append([f'{i}', eq.equi_2_to_1(2*self.architecture[3], self.architecture[4], 'cuda')])
            self.eqblock52_p2p1 = torch.nn.ModuleDict(eqblock52_p2p1_list)
            eqblock52_p1p2_list = []
            for i in self.weight_degrees:
                eqblock52_p1p2_list.append([f'{i}', eq.equi_1_to_2(2*self.architecture[3], self.architecture[4], 'cuda')])
            self.eqblock52_p1p2 = torch.nn.ModuleDict(eqblock52_p1p2_list)
            eqblock52_p1p1_list = []
            for i in self.weight_degrees:
                eqblock52_p1p1_list.append([f'{i}', eq.equi_1_to_1(2*self.architecture[3], self.architecture[4], 'cuda')])
            self.eqblock52_p1p1 = torch.nn.ModuleDict(eqblock52_p1p1_list)
           
            self.elu42 = torch.nn.ELU()
           
            if self.BN:
                self.gn42p1 = GraphNorm(2*self.architecture[3])
                self.gn42p2 = GraphNorm(2*self.architecture[3])

            if self.DO:
                self.do42 = torch.nn.Dropout(p=0.01)
               
        eqblock6_p2p0_list = []
        for i in self.weight_degrees:
            eqblock6_p2p0_list.append([f'{i}', eq.equi_2_to_0(2*self.architecture[-2], self.architecture[-1], 'cuda')])
        self.eqblock6_p2p0 = torch.nn.ModuleDict(eqblock6_p2p0_list)
        eqblock6_p1p0_list = []
        for i in self.weight_degrees:
            eqblock6_p1p0_list.append([f'{i}', eq.equi_1_to_0(2*self.architecture[-2], self.architecture[-1], 'cuda')])
        self.eqblock6_p1p0 = torch.nn.ModuleDict(eqblock6_p1p0_list)
       
        self.elu5 = torch.nn.ELU()
       
        if self.BN:
            self.gn5p1 = GraphNorm(2*self.architecture[-2])
            self.gn5p2 = GraphNorm(2*self.architecture[-2])
       
        if self.DO:
            self.do5 = torch.nn.Dropout(p=0.01)
            
        fc_list = []

        fc_list.append(layers.fully_connected(2*self.architecture[-1], 256, do_prob=0.5))
        fc_list.append(layers.fully_connected(256, 128, do_prob=0.5))
#         fc_list.append(layers.fully_connected(2*self.architecture[-1], 32, do_prob=0.5))
        fc_list.append(layers.fully_connected(128, self.config.num_classes, activation_fn=None))
#         fc_list.append(layers.fully_connected(32, self.config.num_classes, activation_fn=None))

        self.net_fc = torch.nn.ModuleList(fc_list)
        
        
    def get_subgraphs(self, num_nodes, edge_index_input, edge_index, edge_attr, batch, x1, k=1):
        ## Get the sub graphs
#         print(f'edge_index get_subgraphs : {edge_index}')

        edge_index_src = edge_index_input[0]
        edge_index_dst = edge_index_input[1]
        
#         sub_graph_nodes_orig = {'2':[],'3':[],'4':[],'5':[]}
        edge_indices_orig = {}
        sub_graphs = {}
        batch_save = {}
        node_indices_orig = {}
        sub_nodes = {}
        for deg in self.subgraph_degrees:
            edge_indices_orig[f'{deg}'] = []
            sub_graphs[f'{deg}'] = []
            batch_save[f'{deg}'] = []
            node_indices_orig[f'{deg}'] = []
            sub_nodes[f'{deg}'] = []
        
#         print(f'Num of nodes in batch : {num_nodes}')
        for i in range(num_nodes):
            ## Graph level subgraph - matrix features - rho_2
            if k == 1:
                node_indices = edge_index_dst[edge_index_src==i]
            elif k==2:
#                 print(edge_index_src)
                node_indices_1 = edge_index_dst[edge_index_src==i]
#                 print(f'node_indices_1: {node_indices_1}')
                node_indices = []
                for ni1 in node_indices_1:
                    node_indices.append(edge_index_dst[edge_index_src==ni1])
                node_indices = torch.cat(node_indices, 0)
                node_indices = torch.unique(node_indices)

#             print(f'k : {k} - node_indices: {node_indices}')

            edge_index_new, edge_attr_new = subgraph(node_indices, edge_index, edge_attr=edge_attr, relabel_nodes=True)
            print(f'edge_index_new shape : {edge_index_new.shape}')

            sub_graphs[f'{len(node_indices)}'].append(Data(edge_index=edge_index_new, edge_attr=edge_attr_new, num_nodes=len(node_indices)))
#             batch_save[f'{len(node_indices)}'].append(batch[i])
#             sub_graph_nodes_orig[f'{len(node_indices)}'].append(node_indices)

            edge_index_k = []
            node_index_k = []
            for idxi in node_indices:
                for idxj in node_indices:
                    edge_index_k.append([idxi,idxj])
                    if idxi==idxj:
                        node_index_k.append(idxi)
            edge_index_k = torch.LongTensor(edge_index_k).t().to(edge_index.device)
            node_index_k = torch.LongTensor(node_index_k).to(edge_index.device)
            edge_indices_orig[f'{len(node_indices)}'].append(edge_index_k)
            node_indices_orig[f'{len(node_indices)}'].append(node_index_k)
            batch_save[f'{len(node_indices)}'].append(batch[i])
            
            ## Node level subgraph - vector features - rho_1
            if x1 is not None:
                x1_new = x1[node_indices]
#                 print(x1_new.shape)
                sub_nodes[f'{len(node_indices)}'].append(x1_new)
            
        adj_out = []
#         for i in range(1,7):
        for i in self.subgraph_degrees:
            if not sub_graphs[f'{i}']:
                adj_out.append(None)
            else:
                data_2 = Batch.from_data_list(sub_graphs[f'{i}'])
                adj_2 = to_dense_adj(data_2.edge_index, edge_attr=data_2.edge_attr, batch=data_2.batch)
                adj_out.append(adj_2.permute(0,3,1,2))


        x_out = []
#         for i in range(1,7):
        for i in self.subgraph_degrees:
            if not sub_nodes[f'{i}']:
                x_out.append(None)
            else:
                x_2 = torch.stack(sub_nodes[f'{i}'], dim=0)
                x_out.append(x_2.permute(0,2,1))


        ## End get the sub graphs
        
        return adj_out, edge_indices_orig, batch_save, node_indices_orig, x_out
#         return [adj_2, adj_3, adj_4, adj_5], edge_indices_orig, batch_save, node_indices_orig, [x_2, x_3, x_4, x_5]
#         return [adj_2, adj_3, adj_4], edge_indices_orig
    
    def subgraphs_to_graph(self, dev, num_nodes, adj_list, edge_indices_orig):
        ## Put back into 1 sparse graph
        
#         adj_2, adj_3, adj_4, adj_5 = adj_list
        
        edge_index_list = []
#         for i in range(1,7):
        for i in self.subgraph_degrees:
            if edge_indices_orig[f'{i}']:
                edge_index_list.append(torch.cat(edge_indices_orig[f'{i}'], dim=-1))
            else:
                edge_index_list.append(torch.LongTensor().to(dev))

        
        edge_index = torch.cat(edge_index_list, dim=-1)
#         edge_index = torch.cat((edge_index_2, edge_index_3, edge_index_4, edge_index_5), dim=-1)
        
#         print(f'edge_indices == 1 : {edge_index[:, edge_index[0]==1]}')
#         print(f'unique edge_indices : {torch.unique(edge_index)}')
#         print(f'edge_indices shape : {edge_index.shape}')
#         edge_index = torch.cat(edge_indices, dim=-1)
        
#         print(f'max edge_index : {num_nodes}')
        index = edge_index[0] * num_nodes + edge_index[1]
#         print(f'index shape : {index.shape}')
#         print(f'index : {index}')
        index_unique = torch.unique(index)
# #         print(f'edge_index shape : {edge_index.shape}')
# #         print(f'edge_index : {edge_index}')
#         print(f'index_unique shape : {index_unique.shape}')
#         print(f'index_unique : {index_unique}')
        
        i = 0
        for idx_unique in index_unique:
            index[index==idx_unique] = i
            i+=1
            
#         print(f'index shape : {index.shape}')
#         print(f'index : {index}')
        
        adj_out = []
        for adj in adj_list:
            if adj is not None:
                adj_out.append(adj.permute(0,2,3,1).reshape(-1, adj.shape[1]))
            else:
                adj_out.append(torch.FloatTensor().to(dev))


        edge_attr = torch.cat(adj_out, dim=0)
#         edge_attr = torch.cat((adj_2, adj_3, adj_4, adj_5), dim=0)
    
        print(f'edge_attr shape : {edge_attr.shape}')
        print(f'index shape : {index.shape}')
#         print(f'edge_attr : {edge_attr}')
        edge_attr = scatter(edge_attr, index, dim=0, dim_size=len(index_unique), reduce="mean")
#         print(f'edge_attr shape : {edge_attr.shape}')
        
        edge_index = scatter(edge_index, index, dim=1, dim_size=len(index_unique), reduce="max")
#         print(f'edge_indices == 1 : {edge_index[:, edge_index[0]==1]}')
#         edge_index = edge_index[:,index_unique]
#         print(f'edge_index shape : {edge_index.shape}')
#         print(f'edge_index dtype : {edge_index.dtype}')

        ## End put back into 1 sparse graph
        
        return edge_index, edge_attr
    
    def subnodes_to_nodes(self, dev, num_nodes, x_list, node_indices_orig):
        ## Put back into 1 sparse graph
        
#         x_2, x_3, x_4, x_5 = x_list
        
        node_indices = []
#         for i in range(1,7):
        for i in self.subgraph_degrees:
            if node_indices_orig[f'{i}']:
                node_indices.append(torch.cat(node_indices_orig[f'{i}'], dim=-1))
            else:
                node_indices.append(torch.LongTensor().to(dev))

        
        node_indices = torch.cat(node_indices, dim=-1)
#         node_indices = torch.cat((node_indices_2, node_indices_3, node_indices_4, node_indices_5), dim=-1)
        
#         print(f'node_indices shape : {node_indices.shape}')
#         print(f'node_indices : {node_indices}')
        
        index = node_indices # edge_index[0] * num_nodes + edge_index[1]
        index_unique = torch.unique(index)
        
        x_out = []
        for x in x_list:
            if x is not None:
                x_out.append(x.reshape(-1, x.shape[1]))
            else:
                x_out.append(torch.FloatTensor().to(dev))

        x = torch.cat(x_out, dim=0)
#         x = torch.cat((x_2, x_3, x_4, x_5), dim=0)

        x = scatter(x, node_indices, dim=0, dim_size=len(index_unique), reduce="mean")
#         print(f'x shape : {x.shape}')
        
        node_indices = scatter(node_indices, node_indices, dim=0, dim_size=len(index_unique), reduce="max")
#         print(f'node_indices shape : {node_indices.shape}')
#         print(f'node_indices : {node_indices}')

        ## End put back into 1 sparse graph
        
        return node_indices, x
    
    def forward(self, data):

        x2, x1, x0, edge_index, edge_index_dense, edge_attr, batch, y = data.x2, data.x1, data.x0, data.edge_index, data.edge_index_dense, data.edge_attr, data.batch, data.y
        print(f'edge_attr shape : {edge_attr.shape}')
        print(f'edge_index shape : {edge_index.shape}')
        num_nodes = data.num_nodes
        num_graphs = len(data.y)
        dev = edge_index.device

        if len(self.architecture) > 0:
       
            adj_list, edge_indices_orig, batch_orig, node_indices_orig, _ = self.get_subgraphs(num_nodes, edge_index, edge_index, edge_attr, batch, x1, k=1)

            ## HERE WE WOULD DO THE CONVOLUTION WITH DENSE self.degree_mapping
            adj_out = []
            x1_out = []
            for adj_in in adj_list:
                if adj_in is not None:
                    deg_w = self.degree_mapping[f'{adj_in.shape[-1]}']
                    adj_out.append(self.eqblock1_p2p2[deg_w](adj_in))
                    x1_out.append(self.eqblock1_p2p1[deg_w](adj_in))
                else:
                    adj_out.append(None)
                    x1_out.append(None)

        if len(self.architecture) > 1:

            edge_index_new, edge_attr = self.subgraphs_to_graph(dev, num_nodes, adj_out, edge_indices_orig)
            node_indices_orig, x1 = self.subnodes_to_nodes(dev, num_nodes, x1_out, node_indices_orig)
            
            edge_attr_res = edge_attr
            x1_res = x1
                       
            x1 = self.elu1(x1)
            edge_attr = self.elu1(edge_attr)
           
            if self.BN:
                x1 = self.gn1p1(x1, batch)
                edge_attr = self.gn1p2(edge_attr, batch[edge_index_new[0,:]])

            if self.DO:
                x1 = self.do1(x1)
                edge_attr = self.do1(edge_attr)
               
    #         ## Layer 2
            adj_list, edge_indices_orig, _, node_indices_orig, x1_list = self.get_subgraphs(num_nodes, edge_index, edge_index_new, edge_attr, batch, x1, k=1)

    #         ## HERE WE WOULD DO THE CONVOLUTION WITH DENSE
            adj_out = []
            x1_out = []
            for adj_in, x1_in in zip(adj_list, x1_list):
                if (adj_in is not None) and (x1_in is not None):
                    deg_w = self.degree_mapping[f'{adj_in.shape[-1]}']
                    adj_out.append(torch.cat((self.eqblock2_p2p2[deg_w](adj_in),self.eqblock2_p1p2[deg_w](x1_in)), dim=1))
                    x1_out.append(torch.cat((self.eqblock2_p2p1[deg_w](adj_in),self.eqblock2_p1p1[deg_w](x1_in)), dim=1))
                else:
                    adj_out.append(None)
                    x1_out.append(None)

        if len(self.architecture) > 2:

            edge_index_new, edge_attr = self.subgraphs_to_graph(dev, num_nodes, adj_out, edge_indices_orig)
            node_indices_orig, x1 = self.subnodes_to_nodes(dev, num_nodes, x1_out, node_indices_orig)
            
            if self.residual:
                edge_attr = edge_attr + edge_attr_res
                x1 = x1 + x1_res
                
            edge_attr_res = edge_attr
            x1_res = x1

            x1 = self.elu2(x1)
            edge_attr = self.elu2(edge_attr)
           
            if self.BN:
                x1 = self.gn2p1(x1, batch)
                edge_attr = self.gn2p2(edge_attr, batch[edge_index_new[0,:]])
               
            if self.DO:
                x1 = self.do2(x1)
                edge_attr = self.do2(edge_attr)
           
    #         ## Layer 3
            adj_list, edge_indices_orig, _, node_indices_orig, x1_list = self.get_subgraphs(num_nodes, edge_index, edge_index_new, edge_attr, batch, x1, k=1)

    #         ## HERE WE WOULD DO THE CONVOLUTION WITH DENSE
            adj_out = []
            x1_out = []
            for adj_in, x1_in in zip(adj_list, x1_list):
                if (adj_in is not None) and (x1_in is not None):
                    deg_w = self.degree_mapping[f'{adj_in.shape[-1]}']
                    adj_out.append(torch.cat((self.eqblock3_p2p2[deg_w](adj_in),self.eqblock3_p1p2[deg_w](x1_in)), dim=1))
                    x1_out.append(torch.cat((self.eqblock3_p2p1[deg_w](adj_in),self.eqblock3_p1p1[deg_w](x1_in)), dim=1))
                else:
                    adj_out.append(None)
                    x1_out.append(None)
                   
        if len(self.architecture) > 3:

            edge_index_new, edge_attr = self.subgraphs_to_graph(dev, num_nodes, adj_out, edge_indices_orig)
            node_indices_orig, x1 = self.subnodes_to_nodes(dev, num_nodes, x1_out, node_indices_orig)
            
            if self.residual:
                edge_attr = edge_attr + edge_attr_res
                x1 = x1 + x1_res
                
            edge_attr_res = edge_attr
            x1_res = x1
           
            x1 = self.elu3(x1)
            edge_attr = self.elu3(edge_attr)
           
            if self.BN:
                x1 = self.gn3p1(x1, batch)
                edge_attr = self.gn3p2(edge_attr, batch[edge_index_new[0,:]])
               
            if self.DO:
                x1 = self.do3(x1)
                edge_attr = self.do3(edge_attr)

    #         ## Layer 4
            adj_list, edge_indices_orig, _, node_indices_orig, x1_list = self.get_subgraphs(num_nodes, edge_index, edge_index_new, edge_attr, batch, x1, k=1)

    #         ## HERE WE WOULD DO THE CONVOLUTION WITH DENSE
            adj_out = []
            x1_out = []
            for adj_in, x1_in in zip(adj_list, x1_list):
                if (adj_in is not None) and (x1_in is not None):
                    deg_w = self.degree_mapping[f'{adj_in.shape[-1]}']
                    adj_out.append(torch.cat((self.eqblock4_p2p2[deg_w](adj_in),self.eqblock4_p1p2[deg_w](x1_in)), dim=1))
                    x1_out.append(torch.cat((self.eqblock4_p2p1[deg_w](adj_in),self.eqblock4_p1p1[deg_w](x1_in)), dim=1))
                else:
                    adj_out.append(None)
                    x1_out.append(None)
                   
        if len(self.architecture) > 4:

            edge_index_new, edge_attr = self.subgraphs_to_graph(dev, num_nodes, adj_out, edge_indices_orig)
            node_indices_orig, x1 = self.subnodes_to_nodes(dev, num_nodes, x1_out, node_indices_orig)
            
            if self.residual:
                edge_attr = edge_attr + edge_attr_res
                x1 = x1 + x1_res
                
            edge_attr_res = edge_attr
            x1_res = x1
           
            x1 = self.elu4(x1)
            edge_attr = self.elu4(edge_attr)
           
            if self.BN:
                x1 = self.gn4p1(x1, batch)
                edge_attr = self.gn4p2(edge_attr, batch[edge_index_new[0,:]])
               
            if self.DO:
                x1 = self.do4(x1)
                edge_attr = self.do4(edge_attr)

    #         ## Layer 5
            adj_list, edge_indices_orig, _, node_indices_orig, x1_list = self.get_subgraphs(num_nodes, edge_index, edge_index_new, edge_attr, batch, x1, k=1)

    #         ## HERE WE WOULD DO THE CONVOLUTION WITH DENSE
            adj_out = []
            x1_out = []
            for adj_in, x1_in in zip(adj_list, x1_list):
                if (adj_in is not None) and (x1_in is not None):
                    deg_w = self.degree_mapping[f'{adj_in.shape[-1]}']
                    adj_out.append(torch.cat((self.eqblock5_p2p2[deg_w](adj_in),self.eqblock5_p1p2[deg_w](x1_in)), dim=1))
                    x1_out.append(torch.cat((self.eqblock5_p2p1[deg_w](adj_in),self.eqblock5_p1p1[deg_w](x1_in)), dim=1))
                else:
                    adj_out.append(None)
                    x1_out.append(None)
                    
        if len(self.architecture) > 6:

            edge_index_new, edge_attr = self.subgraphs_to_graph(dev, num_nodes, adj_out, edge_indices_orig)
            node_indices_orig, x1 = self.subnodes_to_nodes(dev, num_nodes, x1_out, node_indices_orig)
            
            if self.residual:
                edge_attr = edge_attr + edge_attr_res
                x1 = x1 + x1_res
                
            edge_attr_res = edge_attr
            x1_res = x1
           
            x1 = self.elu41(x1)
            edge_attr = self.elu41(edge_attr)
           
            if self.BN:
                x1 = self.gn41p1(x1, batch)
                edge_attr = self.gn41p2(edge_attr, batch[edge_index_new[0,:]])
               
            if self.DO:
                x1 = self.do41(x1)
                edge_attr = self.do41(edge_attr)

    #         ## Layer 51
            adj_list, edge_indices_orig, _, node_indices_orig, x1_list = self.get_subgraphs(num_nodes, edge_index, edge_index_new, edge_attr, batch, x1, k=1)

    #         ## HERE WE WOULD DO THE CONVOLUTION WITH DENSE
            adj_out = []
            x1_out = []
            for adj_in, x1_in in zip(adj_list, x1_list):
                if (adj_in is not None) and (x1_in is not None):
                    deg_w = self.degree_mapping[f'{adj_in.shape[-1]}']
                    adj_out.append(torch.cat((self.eqblock51_p2p2[deg_w](adj_in),self.eqblock51_p1p2[deg_w](x1_in)), dim=1))
                    x1_out.append(torch.cat((self.eqblock51_p2p1[deg_w](adj_in),self.eqblock51_p1p1[deg_w](x1_in)), dim=1))
                else:
                    adj_out.append(None)
                    x1_out.append(None)
                    
        if len(self.architecture) > 7:

            edge_index_new, edge_attr = self.subgraphs_to_graph(dev, num_nodes, adj_out, edge_indices_orig)
            node_indices_orig, x1 = self.subnodes_to_nodes(dev, num_nodes, x1_out, node_indices_orig)
            
            if self.residual:
                edge_attr = edge_attr + edge_attr_res
                x1 = x1 + x1_res
                
            edge_attr_res = edge_attr
            x1_res = x1
           
            x1 = self.elu42(x1)
            edge_attr = self.elu42(edge_attr)
           
            if self.BN:
                x1 = self.gn42p1(x1, batch)
                edge_attr = self.gn42p2(edge_attr, batch[edge_index_new[0,:]])
               
            if self.DO:
                x1 = self.do42(x1)
                edge_attr = self.do42(edge_attr)

    #         ## Layer 52
            adj_list, edge_indices_orig, _, node_indices_orig, x1_list = self.get_subgraphs(num_nodes, edge_index, edge_index_new, edge_attr, batch, x1, k=1)

    #         ## HERE WE WOULD DO THE CONVOLUTION WITH DENSE
            adj_out = []
            x1_out = []
            for adj_in, x1_in in zip(adj_list, x1_list):
                if (adj_in is not None) and (x1_in is not None):
                    deg_w = self.degree_mapping[f'{adj_in.shape[-1]}']
                    adj_out.append(torch.cat((self.eqblock52_p2p2[deg_w](adj_in),self.eqblock52_p1p2[deg_w](x1_in)), dim=1))
                    x1_out.append(torch.cat((self.eqblock52_p2p1[deg_w](adj_in),self.eqblock52_p1p1[deg_w](x1_in)), dim=1))
                else:
                    adj_out.append(None)
                    x1_out.append(None)

        edge_index_new, edge_attr = self.subgraphs_to_graph(dev, num_nodes, adj_out, edge_indices_orig)
        node_indices_orig, x1 = self.subnodes_to_nodes(dev, num_nodes, x1_out, node_indices_orig)
        
        if self.residual:
            edge_attr = edge_attr + edge_attr_res
            x1 = x1 + x1_res

        edge_attr_res = edge_attr
        x1_res = x1
       
        x1 = self.elu5(x1)
        edge_attr = self.elu5(edge_attr)
           
        if self.BN:
            x1 = self.gn5p1(x1, batch)
            edge_attr = self.gn5p2(edge_attr, batch[edge_index_new[0,:]])
           
        if self.DO:
            x1 = self.do5(x1)
            edge_attr = self.do5(edge_attr)

#         ## Layer 6
        adj_list, edge_indices_orig, _, node_indices_orig, x1_list = self.get_subgraphs(num_nodes, edge_index, edge_index_new, edge_attr, batch, x1, k=1)

#         ## HERE WE WOULD DO THE CONVOLUTION WITH DENSE
        adj_out = []
        for adj_in, x1_in in zip(adj_list, x1_list):
            if (adj_in is not None) and (x1_in is not None):
                deg_w = self.degree_mapping[f'{adj_in.shape[-1]}']
                adj_out.append(torch.cat((self.eqblock6_p2p0[deg_w](adj_in), self.eqblock6_p1p0[deg_w](x1_in)), dim=1))
            else:
                adj_out.append(torch.FloatTensor().to(dev))


        output = torch.cat(adj_out, dim=0)
       
        output = self.elu5(output)

        batch_out = []
        for i in self.subgraph_degrees:
            if batch_orig[f'{i}']:
                batch_out.append(torch.stack(batch_orig[f'{i}']))
            else:
                batch_out.append(torch.LongTensor().to(dev))
       
        batch_output = torch.cat(batch_out, dim=0)

        output = scatter(output, batch_output, dim=0, dim_size=num_graphs, reduce="mean")
        for i, l in enumerate(self.net_fc):
            output = l(output)    
        return output
    
    
class perm_eq_linear(torch.nn.Module):
    def __init__(self, input_features, output_features, BN=True, DO=False):
        super(perm_eq_linear, self).__init__()
        
        self.BN = BN
        self.DO = DO
        
        p2p2_list = []
        for i in self.weight_degrees:
            p2p2_list.append([f'{i}', eq.equi_2_to_2(input_features, output_features, 'cuda')])
        self.eq_p2p2 = torch.nn.ModuleDict(p2p2_list)
        p2p1_list = []
        for i in self.weight_degrees:
            p2p1_list.append([f'{i}', eq.equi_2_to_1(input_features, output_features, 'cuda')])
        self.eq_p2p1 = torch.nn.ModuleDict(p2p1_list)
        p1p2_list = []
        for i in self.weight_degrees:
            p1p2_list.append([f'{i}', eq.equi_1_to_2(input_features, output_features, 'cuda')])
        self.eq_p1p2 = torch.nn.ModuleDict(p1p2_list)
        p1p1_list = []
        for i in self.weight_degrees:
            p1p1_list.append([f'{i}', eq.equi_1_to_1(input_features, output_features, 'cuda')])
        self.eq_p1p1 = torch.nn.ModuleDict(p1p1_list)
        
        self.elu = torch.nn.ELU()
           
        if self.BN:
            self.gnp1 = GraphNorm(self.architecture[0])
            self.gnp2 = GraphNorm(self.architecture[0])

        if self.DO:
            self.do = torch.nn.Dropout(p=0.01)
            
    def get_subgraphs(self, num_nodes, edge_index_input, edge_index, edge_attr, batch, x1):
        edge_index_src = edge_index_input[0]
        edge_index_dst = edge_index_input[1]
        
        edge_indices_orig = {}
        sub_graphs = {}
        batch_save = {}
        node_indices_orig = {}
        sub_nodes = {}
        for deg in self.subgraph_degrees:
            edge_indices_orig[f'{deg}'] = []
            sub_graphs[f'{deg}'] = []
            batch_save[f'{deg}'] = []
            node_indices_orig[f'{deg}'] = []
            sub_nodes[f'{deg}'] = []

        for i in range(num_nodes):
            node_indices = edge_index_dst[edge_index_src==i]

            edge_index_new, edge_attr_new = subgraph(node_indices, edge_index, edge_attr=edge_attr, relabel_nodes=True)

            sub_graphs[f'{len(node_indices)}'].append(Data(edge_index=edge_index_new, edge_attr=edge_attr_new, num_nodes=len(node_indices)))

            edge_index_k = []
            node_index_k = []
            for idxi in node_indices:
                for idxj in node_indices:
                    edge_index_k.append([idxi,idxj])
                    if idxi==idxj:
                        node_index_k.append(idxi)
            edge_index_k = torch.LongTensor(edge_index_k).t().to(edge_index.device)
            node_index_k = torch.LongTensor(node_index_k).to(edge_index.device)
            edge_indices_orig[f'{len(node_indices)}'].append(edge_index_k)
            node_indices_orig[f'{len(node_indices)}'].append(node_index_k)
            batch_save[f'{len(node_indices)}'].append(batch[i])
            
            ## Node level subgraph - vector features - rho_1
            if x1 is not None:
                x1_new = x1[node_indices]
                sub_nodes[f'{len(node_indices)}'].append(x1_new)
            
        adj_out = []
        for i in self.subgraph_degrees:
            if not sub_graphs[f'{i}']:
                adj_out.append(None)
            else:
                data_2 = Batch.from_data_list(sub_graphs[f'{i}'])
                adj_2 = to_dense_adj(data_2.edge_index, edge_attr=data_2.edge_attr, batch=data_2.batch)
                adj_out.append(adj_2.permute(0,3,1,2))

        x_out = []
        for i in self.subgraph_degrees:
            if not sub_nodes[f'{i}']:
                x_out.append(None)
            else:
                x_2 = torch.stack(sub_nodes[f'{i}'], dim=0)
                x_out.append(x_2.permute(0,2,1))

        return adj_out, edge_indices_orig, batch_save, node_indices_orig, x_out
    
    def subgraphs_to_graph(self, dev, num_nodes, adj_list, edge_indices_orig):
        edge_index_list = []
        for i in self.subgraph_degrees:
            if edge_indices_orig[f'{i}']:
                edge_index_list.append(torch.cat(edge_indices_orig[f'{i}'], dim=-1))
            else:
                edge_index_list.append(torch.LongTensor().to(dev))

        edge_index = torch.cat(edge_index_list, dim=-1)
        index = edge_index[0] * num_nodes + edge_index[1]
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
                adj_out.append(torch.FloatTensor().to(dev))

        edge_attr = torch.cat(adj_out, dim=0)
        edge_attr = scatter(edge_attr, index, dim=0, dim_size=len(index_unique), reduce="mean")
        edge_index = scatter(edge_index, index, dim=1, dim_size=len(index_unique), reduce="max")
        return edge_index, edge_attr
    
    def subnodes_to_nodes(self, dev, num_nodes, x_list, node_indices_orig):
        node_indices = []
        for i in self.subgraph_degrees:
            if node_indices_orig[f'{i}']:
                node_indices.append(torch.cat(node_indices_orig[f'{i}'], dim=-1))
            else:
                node_indices.append(torch.LongTensor().to(dev))

        node_indices = torch.cat(node_indices, dim=-1)
        
        index = node_indices # edge_index[0] * num_nodes + edge_index[1]
        index_unique = torch.unique(index)
        
        x_out = []
        for x in x_list:
            if x is not None:
                x_out.append(x.reshape(-1, x.shape[1]))
            else:
                x_out.append(torch.FloatTensor().to(dev))

        x = torch.cat(x_out, dim=0)
        x = scatter(x, node_indices, dim=0, dim_size=len(index_unique), reduce="mean")
        node_indices = scatter(node_indices, node_indices, dim=0, dim_size=len(index_unique), reduce="max")
        return node_indices, x
        
    def forward(num_nodes, edge_index, edge_index_new, edge_attr, batch, x1):
        adj_list, edge_indices_orig, _, node_indices_orig, x1_list = self.get_subgraphs(num_nodes, edge_index, edge_index_new, edge_attr, batch, x1)

#         ## HERE WE WOULD DO THE CONVOLUTION WITH DENSE
        adj_out = []
        x1_out = []
        for adj_in, x1_in in zip(adj_list, x1_list):
            if (adj_in is not None) and (x1_in is not None):
                deg_w = self.degree_mapping[f'{adj_in.shape[-1]}']
                adj_out.append(torch.cat((self.eqblock2_p2p2[deg_w](adj_in),self.eqblock2_p1p2[deg_w](x1_in)), dim=1))
                x1_out.append(torch.cat((self.eqblock2_p2p1[deg_w](adj_in),self.eqblock2_p1p1[deg_w](x1_in)), dim=1))
            else:
                adj_out.append(None)
                x1_out.append(None)

        edge_index_new, edge_attr = self.subgraphs_to_graph(dev, num_nodes, adj_out, edge_indices_orig)
        node_indices_orig, x1 = self.subnodes_to_nodes(dev, num_nodes, x1_out, node_indices_orig)

        x1 = self.elu(x1)
        edge_attr = self.elu(edge_attr)

        if self.BN:
            x1 = self.gnp1(x1, batch)
            edge_attr = self.gnp2(edge_attr, batch[edge_index_new[0,:]])

        if self.DO:
            x1 = self.do(x1)
            edge_attr = self.do(edge_attr)
            
        return edge_index_new, edge_attr, x1
                
                
    
class invariant_mp_v2(BaseModel):
    def __init__(self, config, weight_degrees, subgraph_degrees, degree_mapping):
        super(invariant_mp_v2, self).__init__(config)
        print(f'model build for dataset : {config.dataset_name}')

        self.architecture = config.architecture
        self.BN = config.BN
        self.DO = config.DO

        self.is_training = torch.autograd.Variable(torch.ones(1, dtype=torch.bool))
        
        self.weight_degrees = weight_degrees
        self.subgraph_degrees = subgraph_degrees
        self.degree_mapping = degree_mapping
        
        self.eq1 = perm_eq_linear(config.num_features_in, self.architecture[0], BN=True, DO=False)
        self.eq2 = perm_eq_linear(self.architecture[0], self.architecture[1], BN=True, DO=False)
        self.eq3 = perm_eq_linear(self.architecture[1], self.architecture[2], BN=True, DO=False)
            
        fc_list = []

        fc_list.append(layers.fully_connected(2*self.architecture[-1], 256, do_prob=0.5))
        fc_list.append(layers.fully_connected(256, 128, do_prob=0.5))
        fc_list.append(layers.fully_connected(128, self.config.num_classes, activation_fn=None))

        self.net_fc = torch.nn.ModuleList(fc_list)
    
    def forward(self, data):

        x2, x1, x0, edge_index, edge_index_dense, edge_attr, batch, y = data.x2, data.x1, data.x0, data.edge_index, data.edge_index_dense, data.edge_attr, data.batch, data.y
        num_nodes = data.num_nodes
        num_graphs = len(data.y)
        dev = edge_index.device

        edge_index_new, edge_attr, x1 = self.eq1(num_nodes, edge_index, edge_index, edge_attr, batch, x1)
        edge_index_new, edge_attr, x1 = self.eq2(num_nodes, edge_index, edge_index_new, edge_attr, batch, x1)
        edge_index_new, edge_attr, x1 = self.eq3(num_nodes, edge_index, edge_index_new, edge_attr, batch, x1)

#         ## HERE WE WOULD DO THE CONVOLUTION WITH DENSE
        adj_out = []
        for adj_in, x1_in in zip(adj_list, x1_list):
            if (adj_in is not None) and (x1_in is not None):
                deg_w = self.degree_mapping[f'{adj_in.shape[-1]}']
                adj_out.append(torch.cat((self.eqblock6_p2p0[deg_w](adj_in), self.eqblock6_p1p0[deg_w](x1_in)), dim=1))
            else:
                adj_out.append(torch.FloatTensor().to(dev))


        output = torch.cat(adj_out, dim=0)
       
        output = self.elu5(output)

        batch_out = []
        for i in self.subgraph_degrees:
            if batch_orig[f'{i}']:
                batch_out.append(torch.stack(batch_orig[f'{i}']))
            else:
                batch_out.append(torch.LongTensor().to(dev))
       
        batch_output = torch.cat(batch_out, dim=0)

        output = scatter(output, batch_output, dim=0, dim_size=num_graphs, reduce="mean")
        for i, l in enumerate(self.net_fc):
            output = l(output)    
        return output

        
    
class invariant_mp_seq4_e2(BaseModel):
    def __init__(self, config, weight_degrees, subgraph_degrees, degree_mapping):
        super(invariant_mp_seq4_e2, self).__init__(config)
        print(f'model build for dataset : {config.dataset_name}')

        self.hid_dim1 = config.architecture[0]
        self.hid_dim2 = config.architecture[1]
#         self.hid_dim3 = config.architecture[2]
#         self.hid_dim4 = config.architecture[3]
        self.hid_dim5 = config.architecture[2]
        self.hid_dim6 = config.architecture[3]

        self.is_training = torch.autograd.Variable(torch.ones(1, dtype=torch.bool))
        
        self.weight_degrees = weight_degrees
        self.subgraph_degrees = subgraph_degrees
        self.degree_mapping = degree_mapping
        
        eqblock1_p2p2_list = []
        for i in self.weight_degrees:
            eqblock1_p2p2_list.append([f'{i}', eq.equi_2_to_2(config.num_features_in, self.hid_dim1, 'cuda')])
        self.eqblock1_p2p2 = torch.nn.ModuleDict(eqblock1_p2p2_list)
        eqblock1_p2p1_list = []
        for i in self.weight_degrees:
            eqblock1_p2p1_list.append([f'{i}', eq.equi_2_to_1(config.num_features_in, self.hid_dim1, 'cuda')])
        self.eqblock1_p2p1 = torch.nn.ModuleDict(eqblock1_p2p1_list)
        eqblock1_p1p2_list = []
        for i in self.weight_degrees:
            eqblock1_p1p2_list.append([f'{i}', eq.equi_1_to_2(config.num_features_in, self.hid_dim1, 'cuda')])
        self.eqblock1_p1p2 = torch.nn.ModuleDict(eqblock1_p1p2_list)
        eqblock1_p1p1_list = []
        for i in self.weight_degrees:
            eqblock1_p1p1_list.append([f'{i}', eq.equi_1_to_1(config.num_features_in, self.hid_dim1, 'cuda')])
        self.eqblock1_p1p1 = torch.nn.ModuleDict(eqblock1_p1p1_list)
        
        self.gn1p1 = GraphNorm(self.hid_dim1)
        self.gn1p2 = GraphNorm(self.hid_dim1)
        
        eqblock2_p2p2_list = []
        for i in self.weight_degrees:
            eqblock2_p2p2_list.append([f'{i}', eq.equi_2_to_2(self.hid_dim1, self.hid_dim2, 'cuda')])
        self.eqblock2_p2p2 = torch.nn.ModuleDict(eqblock2_p2p2_list)
        eqblock2_p2p1_list = []
        for i in self.weight_degrees:
            eqblock2_p2p1_list.append([f'{i}', eq.equi_2_to_1(self.hid_dim1, self.hid_dim2, 'cuda')])
        self.eqblock2_p2p1 = torch.nn.ModuleDict(eqblock2_p2p1_list)
        eqblock2_p1p2_list = []
        for i in self.weight_degrees:
            eqblock2_p1p2_list.append([f'{i}', eq.equi_1_to_2(self.hid_dim1, self.hid_dim2, 'cuda')])
        self.eqblock2_p1p2 = torch.nn.ModuleDict(eqblock2_p1p2_list)
        eqblock2_p1p1_list = []
        for i in self.weight_degrees:
            eqblock2_p1p1_list.append([f'{i}', eq.equi_1_to_1(self.hid_dim1, self.hid_dim2, 'cuda')])
        self.eqblock2_p1p1 = torch.nn.ModuleDict(eqblock2_p1p1_list)
        
        self.gn2p1 = GraphNorm(2*self.hid_dim2)
        self.gn2p2 = GraphNorm(2*self.hid_dim2)

#         eqblock3_p2p2_list = []
#         for i in self.weight_degrees:
#             eqblock3_p2p2_list.append([f'{i}', eq.equi_2_to_2(2*self.hid_dim2, self.hid_dim3, 'cuda')])
#         self.eqblock3_p2p2 = torch.nn.ModuleDict(eqblock3_p2p2_list)
#         eqblock3_p2p1_list = []
#         for i in self.weight_degrees:
#             eqblock3_p2p1_list.append([f'{i}', eq.equi_2_to_1(2*self.hid_dim2, self.hid_dim3, 'cuda')])
#         self.eqblock3_p2p1 = torch.nn.ModuleDict(eqblock3_p2p1_list)
#         eqblock3_p1p2_list = []
#         for i in self.weight_degrees:
#             eqblock3_p1p2_list.append([f'{i}', eq.equi_1_to_2(2*self.hid_dim2, self.hid_dim3, 'cuda')])
#         self.eqblock3_p1p2 = torch.nn.ModuleDict(eqblock3_p1p2_list)
#         eqblock3_p1p1_list = []
#         for i in self.weight_degrees:
#             eqblock3_p1p1_list.append([f'{i}', eq.equi_1_to_1(2*self.hid_dim2, self.hid_dim3, 'cuda')])
#         self.eqblock3_p1p1 = torch.nn.ModuleDict(eqblock3_p1p1_list)
        
#         self.gn3p1 = GraphNorm(2*self.hid_dim3)
#         self.gn3p2 = GraphNorm(2*self.hid_dim3)

#         eqblock4_p2p2_list = []
#         for i in self.weight_degrees:
#             eqblock4_p2p2_list.append([f'{i}', eq.equi_2_to_2(2*self.hid_dim3, self.hid_dim4, 'cuda')])
#         self.eqblock4_p2p2 = torch.nn.ModuleDict(eqblock4_p2p2_list)
#         eqblock4_p2p1_list = []
#         for i in self.weight_degrees:
#             eqblock4_p2p1_list.append([f'{i}', eq.equi_2_to_1(2*self.hid_dim3, self.hid_dim4, 'cuda')])
#         self.eqblock4_p2p1 = torch.nn.ModuleDict(eqblock4_p2p1_list)
#         eqblock4_p1p2_list = []
#         for i in self.weight_degrees:
#             eqblock4_p1p2_list.append([f'{i}', eq.equi_1_to_2(2*self.hid_dim3, self.hid_dim4, 'cuda')])
#         self.eqblock4_p1p2 = torch.nn.ModuleDict(eqblock4_p1p2_list)
#         eqblock4_p1p1_list = []
#         for i in self.weight_degrees:
#             eqblock4_p1p1_list.append([f'{i}', eq.equi_1_to_1(2*self.hid_dim3, self.hid_dim4, 'cuda')])
#         self.eqblock4_p1p1 = torch.nn.ModuleDict(eqblock4_p1p1_list)
        
#         self.gn4p1 = GraphNorm(2*self.hid_dim4)
#         self.gn4p2 = GraphNorm(2*self.hid_dim4)

        self.eqblock5_p2p2 = eq.equi_2_to_2(2*self.hid_dim2, self.hid_dim5, 'cuda')
        self.eqblock5_p2p1 = eq.equi_2_to_1(2*self.hid_dim2, self.hid_dim5, 'cuda')
        self.eqblock5_p1p2 = eq.equi_1_to_2(2*self.hid_dim2, self.hid_dim5, 'cuda')
        self.eqblock5_p1p1 = eq.equi_1_to_1(2*self.hid_dim2, self.hid_dim5, 'cuda')
        
        self.gn5p1 = GraphNorm(2*self.hid_dim5)
        self.gn5p2 = GraphNorm(2*self.hid_dim5)

        self.eqblock6_p2p0 = eq.equi_2_to_0(2*self.hid_dim5, self.hid_dim6, 'cuda')
        self.eqblock6_p1p0 = eq.equi_1_to_0(2*self.hid_dim5, self.hid_dim6, 'cuda')

        fc_list = []

        fc_list.append(layers.fully_connected(2*self.hid_dim6, 256, do_prob=0.5))
        fc_list.append(layers.fully_connected(256, self.config.num_classes, activation_fn=None))

        self.net_fc = torch.nn.ModuleList(fc_list)
        
    def get_subgraphs(self, num_nodes, edge_index_input, edge_index, edge_attr, batch, x1):
        ## Get the sub graphs
        edge_index_src = edge_index_input[0]
        edge_index_dst = edge_index_input[1]
        
        edge_indices_orig = {}
        sub_graphs = {}
        batch_save = {}
        node_indices_orig = {}
        sub_nodes = {}
        for deg in self.subgraph_degrees:
            edge_indices_orig[f'{deg}'] = []
            sub_graphs[f'{deg}'] = []
            batch_save[f'{deg}'] = []
            node_indices_orig[f'{deg}'] = []
            sub_nodes[f'{deg}'] = []

        for i in range(num_nodes):
            node_indices = edge_index_dst[edge_index_src==i]

            edge_index_new, edge_attr_new = subgraph(node_indices, edge_index, edge_attr=edge_attr, relabel_nodes=True)

            sub_graphs[f'{len(node_indices)}'].append(Data(edge_index=edge_index_new, edge_attr=edge_attr_new, num_nodes=len(node_indices)))

            edge_index_k = []
            node_index_k = []
            for idxi in node_indices:
                for idxj in node_indices:
                    edge_index_k.append([idxi,idxj])
                    if idxi==idxj:
                        node_index_k.append(idxi)
            edge_index_k = torch.LongTensor(edge_index_k).t().to(edge_index.device)
            node_index_k = torch.LongTensor(node_index_k).to(edge_index.device)
            edge_indices_orig[f'{len(node_indices)}'].append(edge_index_k)
            node_indices_orig[f'{len(node_indices)}'].append(node_index_k)
            batch_save[f'{len(node_indices)}'].append(batch[i])
            
            ## Node level subgraph - vector features - rho_1
            if x1 is not None:
                x1_new = x1[node_indices]
                sub_nodes[f'{len(node_indices)}'].append(x1_new)
            
        adj_out = []
        for i in self.subgraph_degrees:
            if not sub_graphs[f'{i}']:
                adj_out.append(None)
            else:
                data_2 = Batch.from_data_list(sub_graphs[f'{i}'])
                adj_2 = to_dense_adj(data_2.edge_index, edge_attr=data_2.edge_attr, batch=data_2.batch)
                adj_out.append(adj_2.permute(0,3,1,2))

        x_out = []
        for i in self.subgraph_degrees:
            if not sub_nodes[f'{i}']:
                x_out.append(None)
            else:
                x_2 = torch.stack(sub_nodes[f'{i}'], dim=0)
                x_out.append(x_2.permute(0,2,1))

        ## End get the sub graphs
        
        return adj_out, edge_indices_orig, batch_save, node_indices_orig, x_out
    
    def subgraphs_to_graph(self, dev, num_nodes, adj_list, edge_indices_orig):
        ## Put back into 1 sparse graph

        edge_index_list = []
        for i in self.subgraph_degrees:
            if edge_indices_orig[f'{i}']:
                edge_index_list.append(torch.cat(edge_indices_orig[f'{i}'], dim=-1))
            else:
                edge_index_list.append(torch.LongTensor().to(dev))

        edge_index = torch.cat(edge_index_list, dim=-1)

        index = edge_index[0] * num_nodes + edge_index[1]
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
                adj_out.append(torch.FloatTensor().to(dev))

        edge_attr = torch.cat(adj_out, dim=0)
    
        edge_attr = scatter(edge_attr, index, dim=0, dim_size=len(index_unique), reduce="mean")
        
        edge_index = scatter(edge_index, index, dim=1, dim_size=len(index_unique), reduce="max")

        ## End put back into 1 sparse graph
        
        return edge_index, edge_attr
    
    def subnodes_to_nodes(self, dev, num_nodes, x_list, node_indices_orig):
        ## Put back into 1 sparse graph

        node_indices = []
        for i in self.subgraph_degrees:
            if node_indices_orig[f'{i}']:
                node_indices.append(torch.cat(node_indices_orig[f'{i}'], dim=-1))
            else:
                node_indices.append(torch.LongTensor().to(dev))

        node_indices = torch.cat(node_indices, dim=-1)
        index = node_indices # edge_index[0] * num_nodes + edge_index[1]
        index_unique = torch.unique(index)
        
        x_out = []
        for x in x_list:
            if x is not None:
                x_out.append(x.reshape(-1, x.shape[1]))
            else:
                x_out.append(torch.FloatTensor().to(dev))

        x = torch.cat(x_out, dim=0)

        x = scatter(x, node_indices, dim=0, dim_size=len(index_unique), reduce="mean")
        
        node_indices = scatter(node_indices, node_indices, dim=0, dim_size=len(index_unique), reduce="max")

        ## End put back into 1 sparse graph
        
        return node_indices, x
    
    def forward(self, data):

        x2, x1, x0, edge_index, edge_index_dense, edge_attr, batch, y = data.x2, data.x1, data.x0, data.edge_index, data.edge_index_dense, data.edge_attr, data.batch, data.y
        num_nodes = data.num_nodes
        num_graphs = len(data.y)
        dev = edge_index.device

        
        adj_list, edge_indices_orig, batch_orig, node_indices_orig, _ = self.get_subgraphs(num_nodes, edge_index, edge_index, edge_attr, batch, x1)

        ## HERE WE WOULD DO THE CONVOLUTION WITH DENSE self.degree_mapping
        adj_out = []
        x1_out = []
        for adj_in in adj_list:
            if adj_in is not None:
                deg_w = self.degree_mapping[f'{adj_in.shape[-1]}']
                adj_out.append(self.eqblock1_p2p2[deg_w](adj_in))
                x1_out.append(self.eqblock1_p2p1[deg_w](adj_in))
            else:
                adj_out.append(None)
                x1_out.append(None)
    
        edge_index_new, edge_attr = self.subgraphs_to_graph(dev, num_nodes, adj_out, edge_indices_orig)
        node_indices_orig, x1 = self.subnodes_to_nodes(dev, num_nodes, x1_out, node_indices_orig)

        x1 = self.gn1p1(x1, batch)
        edge_attr = self.gn1p2(edge_attr, batch[edge_index_new[0,:]])

#         ## Layer 2
        adj_list, edge_indices_orig, _, node_indices_orig, x1_list = self.get_subgraphs(num_nodes, edge_index, edge_index_new, edge_attr, batch, x1)

#         ## HERE WE WOULD DO THE CONVOLUTION WITH DENSE
        adj_out = []
        x1_out = []
        for adj_in, x1_in in zip(adj_list, x1_list):
            if (adj_in is not None) and (x1_in is not None):
                deg_w = self.degree_mapping[f'{adj_in.shape[-1]}']
                adj_out.append(torch.cat((self.eqblock2_p2p2[deg_w](adj_in),self.eqblock2_p1p2[deg_w](x1_in)), dim=1))
                x1_out.append(torch.cat((self.eqblock2_p2p1[deg_w](adj_in),self.eqblock2_p1p1[deg_w](x1_in)), dim=1))
            else:
                adj_out.append(None)
                x1_out.append(None)

        edge_index_new, edge_attr = self.subgraphs_to_graph(dev, num_nodes, adj_out, edge_indices_orig)
        node_indices_orig, x1 = self.subnodes_to_nodes(dev, num_nodes, x1_out, node_indices_orig)

        x1 = self.gn2p1(x1, batch)
        edge_attr = self.gn2p2(edge_attr, batch[edge_index_new[0,:]])

# #         ## Layer 3
#         adj_list, edge_indices_orig, _, node_indices_orig, x1_list = self.get_subgraphs(num_nodes, edge_index, edge_index_new, edge_attr, batch, x1)

# #         ## HERE WE WOULD DO THE CONVOLUTION WITH DENSE
#         adj_out = []
#         x1_out = []
#         for adj_in, x1_in in zip(adj_list, x1_list):
#             if (adj_in is not None) and (x1_in is not None):
#                 deg_w = self.degree_mapping[f'{adj_in.shape[-1]}']
#                 adj_out.append(torch.cat((self.eqblock3_p2p2[deg_w](adj_in),self.eqblock3_p1p2[deg_w](x1_in)), dim=1))
#                 x1_out.append(torch.cat((self.eqblock3_p2p1[deg_w](adj_in),self.eqblock3_p1p1[deg_w](x1_in)), dim=1))
#             else:
#                 adj_out.append(None)
#                 x1_out.append(None)

#         edge_index_new, edge_attr = self.subgraphs_to_graph(dev, num_nodes, adj_out, edge_indices_orig)
#         node_indices_orig, x1 = self.subnodes_to_nodes(dev, num_nodes, x1_out, node_indices_orig)

#         x1 = self.gn3p1(x1, batch)
#         edge_attr = self.gn3p2(edge_attr, batch[edge_index_new[0,:]])

# #         print(f'edge_attr : {edge_attr.shape}')
        
# #         ## Layer 4
#         adj_list, edge_indices_orig, _, node_indices_orig, x1_list = self.get_subgraphs(num_nodes, edge_index, edge_index_new, edge_attr, batch, x1)

# #         ## HERE WE WOULD DO THE CONVOLUTION WITH DENSE
#         adj_out = []
#         x1_out = []
#         for adj_in, x1_in in zip(adj_list, x1_list):
#             if (adj_in is not None) and (x1_in is not None):
# #                 print(f'adj_in : {adj_in.shape}')
#                 deg_w = self.degree_mapping[f'{adj_in.shape[-1]}']
#                 adj_out.append(torch.cat((self.eqblock4_p2p2[deg_w](adj_in),self.eqblock4_p1p2[deg_w](x1_in)), dim=1))
#                 x1_out.append(torch.cat((self.eqblock4_p2p1[deg_w](adj_in),self.eqblock4_p1p1[deg_w](x1_in)), dim=1))
#             else:
#                 adj_out.append(None)
#                 x1_out.append(None)

#         edge_index_new, edge_attr = self.subgraphs_to_graph(dev, num_nodes, adj_out, edge_indices_orig)
#         node_indices_orig, x1 = self.subnodes_to_nodes(dev, num_nodes, x1_out, node_indices_orig)

# #         print(f'edge_attr : {edge_attr.shape}')
        
#         x1 = self.gn4p1(x1, batch)
#         edge_attr = self.gn4p2(edge_attr, batch[edge_index_new[0,:]])
        
#         print(f'batch : {batch.shape}')
#         print(f'edge_index_new : {edge_index_new.shape}')
#         print(f'edge_attr : {edge_attr.shape}')
#         print(f'edge_attr : {edge_index_new}')
#         print(f'num_nodes : {num_nodes}')
        
#         edge_index_new, edge_attr_new = subgraph(torch.arange(num_nodes), edge_index=edge_index_new, edge_attr=edge_attr, relabel_nodes=True)
#         print(f'edge_index_new : {edge_index_new.shape}')
#         print(f'edge_attr_new : {edge_attr_new.shape}')
#         data_2 = Data(edge_index=edge_index_new, edge_attr=edge_attr_new, num_nodes=num_nodes, batch=batch[edge_index_new[:,0]])
        edge_attr = to_dense_adj(edge_index=edge_index_new, edge_attr=edge_attr, batch=batch)
        edge_attr = edge_attr.permute(0,3,1,2)
        x1, x1_mask = to_dense_batch(x=x1, batch=batch)
        x1 = x1.permute(0,2,1)
#         print(f'edge_attr : {edge_attr.shape}')
#         print(f'x1 : {x1.shape}')


#         ## Layer 5

#         ## HERE WE WOULD DO THE CONVOLUTION WITH DENSE

        edge_attr = torch.cat((self.eqblock5_p2p2(edge_attr),self.eqblock5_p1p2(x1)), dim=1)
        x1 = torch.cat((self.eqblock5_p2p1(edge_attr),self.eqblock5_p1p1(x1)), dim=1)
        
#         print(f'edge_attr : {edge_attr.shape}')
#         print(f'x1 : {x1.shape}')

#         edge_index_new, edge_attr = self.subgraphs_to_graph(dev, num_nodes, adj_out, edge_indices_orig)
#         node_indices_orig, x1 = self.subnodes_to_nodes(dev, num_nodes, x1_out, node_indices_orig)

#         x1 = self.gn5p1(x1, batch)
#         edge_attr = self.gn5p2(edge_attr, batch[edge_index_new[0,:]])

#         ## Layer 6
        
#         ## HERE WE WOULD DO THE CONVOLUTION WITH DENSE

        output = torch.cat((self.eqblock6_p2p0(edge_attr), self.eqblock6_p1p0(x1)), dim=1)
#         print(f'output : {output.shape}')
        
#         batch_out = []
# #         for i in range(1,7):
#         for i in self.subgraph_degrees:
#             if batch_orig[f'{i}']:
#                 batch_out.append(torch.stack(batch_orig[f'{i}']))
#             else:
#                 batch_out.append(torch.LongTensor().to(dev))

        
#         batch_output = torch.cat(batch_out, dim=0)

#         output = scatter(output, batch_output, dim=0, dim_size=num_graphs, reduce="mean")
        for i, l in enumerate(self.net_fc):
            output = l(output)    
        return output
        

#     def build_model(self):
#         # here you build the tensorflow graph of any model you want and define the loss.
#         self.is_training = torch.autograd.Variable(torch.ones(1, dtype=torch.bool))
# #         self.is_training = tf.placeholder(tf.bool)

# #         self.graphs = tf.placeholder(tf.float32, shape=[None, self.config.node_labels + 1, None, None])
# #         self.labels = tf.placeholder(tf.int32, shape=[None])

#         # build network architecture using config file
#         net = eq.equi_2_to_2('equi0', self.data.train_graphs[0].shape[0], self.config.architecture[0], self.graphs)
#         net = tf.nn.relu(net, name='relu0')
#         for layer in range(1, len(self.config.architecture)):
#             net = eq.equi_2_to_2('equi%d' %layer, self.config.architecture[layer-1], self.config.architecture[layer], net)
#             net = tf.nn.relu(net, name='relu%d'%layer)

#         net = layers.diag_offdiag_maxpool(net)

#         net = layers.fully_connected(net, 512, "fully1")
#         net = layers.fully_connected(net, 256, "fully2")
#         net = layers.fully_connected(net, self.config.num_classes, "fully3", activation_fn=None)

#         # define loss function
#         with tf.name_scope("loss"):
#             self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=net))
#             self.correct_predictions = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(net, 1, output_type=tf.int32), self.labels), tf.int32))

#         # get learning rate with decay every 20 epochs
#         learning_rate = self.get_learning_rate(self.global_step_tensor, self.data.train_size*20)

#         # choose optimizer
#         if self.config.optimizer == 'momentum':
#             self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=self.config.momentum)
#         elif self.config.optimizer == 'adam':
#             self.optimizer = tf.train.AdamOptimizer(learning_rate)

#         # define train step
#         self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)


#     def init_saver(self):
#         # here you initialize the tensorflow saver that will be used in saving the checkpoints.
#         self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

#     def get_learning_rate(self, global_step, decay_step):
#         """
#         helper method to fit learning rat
#         :param global_step: current index into dataset, int
#         :param decay_step: decay step, float
#         :return: output: N x S x m x m tensor
#         """
#         learning_rate = tf.train.exponential_decay(
#             self.config.learning_rate,  # Base learning rate.
#             global_step*self.config.batch_size,
#             decay_step,
#             self.config.decay_rate,  # Decay rate.
#             staircase=True)
#         learning_rate = tf.maximum(learning_rate, 0.00001)
#         return learning_rate
