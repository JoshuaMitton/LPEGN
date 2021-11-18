from models.base_model import BaseModel
import layers.equivariant_linear as eq
from layers.subgraphs import get_subgraphs, subgraphs_to_graph, subnodes_to_nodes
import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.utils.subgraph import k_hop_subgraph, subgraph
from torch_geometric.data import Data, Batch
from torch_geometric.nn.norm import GraphNorm
from torch_scatter import gather_csr, scatter, segment_csr

class eq_gconv(torch.nn.Module):
    def __init__(self, repin, repout, hid_dim1, hid_dim2, weight_degrees, subgraph_degrees, degree_mapping, k=1):
        super(eq_gconv, self).__init__()
        self.hid_dim1 = hid_dim1
        self.hid_dim2 = hid_dim2        
        self.weight_degrees = weight_degrees
        self.subgraph_degrees = subgraph_degrees
        self.degree_mapping = degree_mapping
        self.k = k
        self.repin = repin
        self.repout = repout
        
        if (repin==21) and (repout==21):
            eqblock_p2p2_list = []
            for i in self.weight_degrees:
                eqblock_p2p2_list.append([f'{i}', eq.equi_2_to_2(self.hid_dim1, self.hid_dim2, 'cuda')])
            self.eqblock_p2p2 = torch.nn.ModuleDict(eqblock_p2p2_list)
            eqblock_p2p1_list = []
            for i in self.weight_degrees:
                eqblock_p2p1_list.append([f'{i}', eq.equi_2_to_1(self.hid_dim1, self.hid_dim2, 'cuda')])
            self.eqblock_p2p1 = torch.nn.ModuleDict(eqblock_p2p1_list)
            eqblock_p1p2_list = []
            for i in self.weight_degrees:
                eqblock_p1p2_list.append([f'{i}', eq.equi_1_to_2(self.hid_dim1, self.hid_dim2, 'cuda')])
            self.eqblock_p1p2 = torch.nn.ModuleDict(eqblock_p1p2_list)
            eqblock_p1p1_list = []
            for i in self.weight_degrees:
                eqblock_p1p1_list.append([f'{i}', eq.equi_1_to_1(self.hid_dim1, self.hid_dim2, 'cuda')])
            self.eqblock_p1p1 = torch.nn.ModuleDict(eqblock_p1p1_list)
            
            self.gnp1 = GraphNorm(2*self.hid_dim2)
            self.gnp2 = GraphNorm(2*self.hid_dim2)
        
        if (repin==2) and (repout==21):
            eqblock_p2p2_list = []
            for i in self.weight_degrees:
                eqblock_p2p2_list.append([f'{i}', eq.equi_2_to_2(self.hid_dim1, self.hid_dim2, 'cuda')])
            self.eqblock_p2p2 = torch.nn.ModuleDict(eqblock_p2p2_list)
            eqblock_p2p1_list = []
            for i in self.weight_degrees:
                eqblock_p2p1_list.append([f'{i}', eq.equi_2_to_1(self.hid_dim1, self.hid_dim2, 'cuda')])
            self.eqblock_p2p1 = torch.nn.ModuleDict(eqblock_p2p1_list)
            
            self.gnp1 = GraphNorm(self.hid_dim2)
            self.gnp2 = GraphNorm(self.hid_dim2)
            
        if (repin==21) and (repout==0):
            eqblock_p2p0_list = []
            for i in self.weight_degrees:
                eqblock_p2p0_list.append([f'{i}', eq.equi_2_to_0(self.hid_dim1, self.hid_dim2, 'cuda')])
            self.eqblock_p2p0 = torch.nn.ModuleDict(eqblock_p2p0_list)
            eqblock_p1p0_list = []
            for i in self.weight_degrees:
                eqblock_p1p0_list.append([f'{i}', eq.equi_1_to_0(self.hid_dim1, self.hid_dim2, 'cuda')])
            self.eqblock_p1p0 = torch.nn.ModuleDict(eqblock_p1p0_list)
            
        self.elu = torch.nn.ELU()
    
    def forward(self, data, T2, T1):
        T2_list, edge_indices_orig, T1_list, node_indices_orig, data, _ = get_subgraphs(data, T2, T1, self.subgraph_degrees, k=self.k)

        T2_out = []
        T1_out = []
        T0_out = []
        if (self.repin==21) and (self.repout==21):
            for T2_in, T1_in in zip(T2_list, T1_list):
                if (T2_in is not None) and (T1_in is not None):
                    deg_w = self.degree_mapping[f'{T2_in.shape[-1]}']
                    T2_out.append(torch.cat((self.eqblock_p2p2[deg_w](T2_in),self.eqblock_p1p2[deg_w](T1_in)), dim=1))
                    T1_out.append(torch.cat((self.eqblock_p2p1[deg_w](T2_in),self.eqblock_p1p1[deg_w](T1_in)), dim=1))
                else:
                    T2_out.append(None)
                    T1_out.append(None)
        if (self.repin==2) and (self.repout==21):
            for T2_in in T2_list:
                if (T2_in is not None):
                    deg_w = self.degree_mapping[f'{T2_in.shape[-1]}']
                    T2_out.append(self.eqblock_p2p2[deg_w](T2_in))
                    T1_out.append(self.eqblock_p2p1[deg_w](T2_in))
                else:
                    T2_out.append(None)
                    T1_out.append(None)
        if (self.repin==21) and (self.repout==0):
            for T2_in, T1_in in zip(T2_list, T1_list):
                if (T2_in is not None) and (T1_in is not None):
                    deg_w = self.degree_mapping[f'{T2_in.shape[-1]}']
                    T0_out.append(torch.cat((self.eqblock_p2p0[deg_w](T2_in),self.eqblock_p1p0[deg_w](T1_in)), dim=1))
                else:
                    T0_out.append(torch.FloatTensor().to(data.dev))
                
        if T2_out:
            edge_index_new, T2 = subgraphs_to_graph(data, T2_out, edge_indices_orig, self.subgraph_degrees)
            data.edge_index = edge_index_new
            T2 = self.elu(T2)
            T2 = self.gnp2(T2, data.batch[edge_index_new[0,:]])
        else:
            T2 = None
            
        if T1_out:
            node_indices_orig, T1 = subnodes_to_nodes(data, T1_out, node_indices_orig, self.subgraph_degrees)
            T1 = self.elu(T1)
            T1 = self.gnp1(T1, data.batch)
        else:
            T1 = None

        if T0_out:
            T0 = torch.cat(T0_out, dim=0)
            T0 = self.elu(T0)
        else:
            T0 = None
            
        return data, T2, T1, T0
        