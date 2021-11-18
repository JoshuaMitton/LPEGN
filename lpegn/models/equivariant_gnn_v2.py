from models.base_model import BaseModel
import layers.equivariant_linear as eq
from layers.subgraphs import get_subgraphs
import layers.layers as layers
import layers.eq_gconv as eq_gconv
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
        self.subgraph_degrees = subgraph_degrees
        
                
        eq_list = []
        eq_list.append(eq_gconv.eq_gconv(repin=2, repout=21, hid_dim1=config.num_features_in, hid_dim2=self.architecture[0], 
                                     weight_degrees=weight_degrees, subgraph_degrees=subgraph_degrees, 
                                     degree_mapping=degree_mapping, k=1))
        eq_list.append(eq_gconv.eq_gconv(repin=21, repout=21, hid_dim1=self.architecture[0], hid_dim2=self.architecture[1], 
                                     weight_degrees=weight_degrees, subgraph_degrees=subgraph_degrees, 
                                     degree_mapping=degree_mapping, k=1))
        eq_list.append(eq_gconv.eq_gconv(repin=21, repout=21, hid_dim1=2*self.architecture[1], hid_dim2=self.architecture[2], 
                                     weight_degrees=weight_degrees, subgraph_degrees=subgraph_degrees, 
                                     degree_mapping=degree_mapping, k=1))
        eq_list.append(eq_gconv.eq_gconv(repin=21, repout=21, hid_dim1=2*self.architecture[2], hid_dim2=self.architecture[3], 
                                     weight_degrees=weight_degrees, subgraph_degrees=subgraph_degrees, 
                                     degree_mapping=degree_mapping, k=1))
        eq_list.append(eq_gconv.eq_gconv(repin=21, repout=0, hid_dim1=2*self.architecture[3], hid_dim2=self.architecture[4], 
                                     weight_degrees=weight_degrees, subgraph_degrees=subgraph_degrees, 
                                     degree_mapping=degree_mapping, k=1))
        self.net_eq = torch.nn.ModuleList(eq_list)

        fc_list = []
        fc_list.append(layers.fully_connected(2*self.architecture[-1], 256, do_prob=0.5))
        fc_list.append(layers.fully_connected(256, 128, do_prob=0.5))
        fc_list.append(layers.fully_connected(128, self.config.num_classes, activation_fn=None))
        self.net_fc = torch.nn.ModuleList(fc_list)
        

    def forward(self, data):
        data.edge_index_orig = data.edge_index.detach()
        data.dev = data.edge_index.get_device()
        
        T2, T1 = data.T2, data.T1
        _, _, _, _, _, batch_orig = get_subgraphs(data, T2, T1, self.subgraph_degrees, k=1)
        num_graphs = len(data.y)

        for i, l in enumerate(self.net_eq):
            data, T2, T1, T0 = l(data, T2, T1)
        output = T0

        batch_out = []
        for i in self.subgraph_degrees:
            if batch_orig[f'{i}']:
                batch_out.append(torch.stack(batch_orig[f'{i}']))
            else:
                batch_out.append(torch.LongTensor().to(data.dev))
        batch_output = torch.cat(batch_out, dim=0)

        output = scatter(output, batch_output, dim=0, dim_size=num_graphs, reduce="mean")
        for i, l in enumerate(self.net_fc):
            output = l(output)    
        return output
