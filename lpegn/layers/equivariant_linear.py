# import tensorflow as tf
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import avg_pool_x
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import add_self_loops, degree

import os
import re
import inspect
import os.path as osp
from uuid import uuid1
from itertools import chain
from inspect import Parameter
from typing import List, Optional, Set
from torch_geometric.typing import Adj, Size

import torch
from torch import Tensor
from jinja2 import Template
from torch_sparse import SparseTensor
from torch_scatter import gather_csr, scatter, segment_csr

from torch_geometric.nn.conv.utils.helpers import expand_left
from torch_geometric.nn.conv.utils.jit import class_from_module_repr
from torch_geometric.nn.conv.utils.typing import (sanitize, split_types_repr, parse_types,
                           resolve_types)
from torch_geometric.nn.conv.utils.inspector import Inspector, func_header_repr, func_body_repr


class equi_2_to_2(torch.nn.Module):
    """equivariant nn layer."""

    def __init__(self, input_depth, output_depth, device):
        super(equi_2_to_2, self).__init__()
        self.basis_dimension = 15
        self.device = device
#         self.coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
        self.coeffs_values = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.cuda()

        self.coeffs = torch.nn.Parameter(self.coeffs_values, requires_grad=True)
        self.diag_bias = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
        self.all_bias = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
       
    def ops_2_to_2(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
#         print(f'input shape : {inputs.shape}')
        diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
#         print(f'diag_part shape : {diag_part.shape}')
        sum_diag_part = torch.sum(diag_part, dim=2, keepdim=True)  # N x D x 1
#         print(f'sum_diag_part shape : {sum_diag_part.shape}')
        sum_of_rows = torch.sum(inputs, dim=3)  # N x D x m
#         print(f'sum_of_rows shape : {sum_of_rows.shape}')
        sum_of_cols = torch.sum(inputs, dim=2)  # N x D x m
#         print(f'sum_of_cols shape : {sum_of_cols.shape}')
        sum_all = torch.sum(sum_of_rows, dim=2)  # N x D
#         print(f'sum_all shape : {sum_all.shape}')

        # op1 - (1234) - extract diag
        op1 = torch.diag_embed(diag_part)  # N x D x m x m

        # op2 - (1234) + (12)(34) - place sum of diag on diag
        op2 = torch.diag_embed(sum_diag_part.repeat(1, 1, dim))  # N x D x m x m

        # op3 - (1234) + (123)(4) - place sum of row i on diag ii
        op3 = torch.diag_embed(sum_of_rows)  # N x D x m x m

        # op4 - (1234) + (124)(3) - place sum of col i on diag ii
        op4 = torch.diag_embed(sum_of_cols)  # N x D x m x m

        # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
        op5 = torch.diag_embed(torch.unsqueeze(sum_all, dim=2).repeat(1, 1, dim))  # N x D x m x m

        # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
        op6 = torch.unsqueeze(sum_of_cols, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

        # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
        op7 = torch.unsqueeze(sum_of_rows, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

        # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
        op8 = torch.unsqueeze(sum_of_cols, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

        # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
        op9 = torch.unsqueeze(sum_of_rows, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

        # op10 - (1234) + (14)(23) - identity
        op10 = inputs  # N x D x m x m

        # op11 - (1234) + (13)(24) - transpose
        op11 = inputs.permute(0, 1, 3, 2)  # N x D x m x m

        # op12 - (1234) + (234)(1) - place ii element in row i
        op12 = torch.unsqueeze(diag_part, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

        # op13 - (1234) + (134)(2) - place ii element in col i
        op13 = torch.unsqueeze(diag_part, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

        # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
        op14 = torch.unsqueeze(sum_diag_part, dim=3).repeat(1, 1, dim, dim)   # N x D x m x m

        # op15 - sum of all ops - place sum of all entries in all entries
        op15 = torch.unsqueeze(torch.unsqueeze(sum_all, dim=2), dim=3).repeat(1, 1, dim, dim)  # N x D x m x m

        if normalization is not None:
            float_dim = dim.type(torch.FloatTensor)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)
                op3 = torch.div(op3, float_dim)
                op4 = torch.div(op4, float_dim)
                op5 = torch.div(op5, float_dim**2)
                op6 = torch.div(op6, float_dim)
                op7 = torch.div(op7, float_dim)
                op8 = torch.div(op8, float_dim)
                op9 = torch.div(op9, float_dim)
                op14 = torch.div(op14, float_dim)
                op15 = torch.div(op15, float_dim**2)

        return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]

    def forward(self, inputs, normalization='inf'):
        m = torch.tensor(inputs.shape[3], dtype=torch.int32, device=self.device)  # extract dimension

#         print(f'inputs device : {inputs.device}')
        ops_out = self.ops_2_to_2(inputs=inputs, dim=m, normalization=normalization)
#         for idx, op in enumerate(ops_out):
#             print(f'ops_out{idx} : {op.shape}')
        ops_out = torch.stack(ops_out, dim=2)

#         print(f'self.coeffs device : {self.coeffs.device}')
#         print(f'ops_out : {ops_out}')
        output = torch.einsum('dsb,ndbij->nsij', self.coeffs, ops_out)  # N x S x m x m

        # bias
#         print(f'diag_bias shape : {self.diag_bias.shape}')
#         print(f'eye shape : {torch.eye(torch.tensor(inputs.shape[3], dtype=torch.int32, device=self.device), device=self.device).shape}')
#         mat_diag_bias = torch.mul(torch.unsqueeze(torch.unsqueeze(torch.eye(torch.tensor(inputs.shape[3], dtype=torch.int32, device=self.device), device=self.device), 0), 0), self.diag_bias)
        mat_diag_bias = self.diag_bias.expand(-1,-1,inputs.shape[3],inputs.shape[3])
        mat_diag_bias = torch.mul(mat_diag_bias, torch.eye(inputs.shape[3], device=self.device))
        output = output + self.all_bias + mat_diag_bias
#         print(f'mat_diag_bias shape : {mat_diag_bias.shape}')

        return output

class equi_2_to_1(torch.nn.Module):
    """equivariant nn layer."""

    def __init__(self, input_depth, output_depth, device):
        super(equi_2_to_1, self).__init__()
        self.basis_dimension = 5
        self.device = device
#         self.coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
        self.coeffs_values = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.cuda()

        self.coeffs = torch.nn.Parameter(self.coeffs_values, requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
       

    def ops_2_to_1(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
        diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
        sum_diag_part = torch.sum(diag_part, dim=2, keepdim=True)  # N x D x 1
        sum_of_rows = torch.sum(inputs, dim=3)  # N x D x m
        sum_of_cols = torch.sum(inputs, dim=2)  # N x D x m
        sum_all = torch.sum(sum_of_rows, dim=2)  # N x D

        # op1 - (123) - extract diag
        op1 = diag_part  # N x D x m

        # op2 - (123) + (12)(3) - tile sum of diag part
        op2 = sum_diag_part.repeat(1, 1, dim)  # N x D x m

        # op3 - (123) + (13)(2) - place sum of row i in element i
        op3 = sum_of_rows  # N x D x m

        # op4 - (123) + (23)(1) - place sum of col i in element i
        op4 = sum_of_cols  # N x D x m

        # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
        op5 = torch.unsqueeze(sum_all, dim=2).repeat(1, 1, dim)  # N x D x m


        if normalization is not None:
            float_dim = dim.type(torch.FloatTensor)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)
                op3 = torch.div(op3, float_dim)
                op4 = torch.div(op4, float_dim)
                op5 = torch.div(op5, float_dim ** 2)

        return [op1, op2, op3, op4, op5]

    def forward(self, inputs, normalization='inf'):
        m = torch.tensor(inputs.shape[3], dtype=torch.int32, device=self.device)  # extract dimension

        ops_out = self.ops_2_to_1(inputs=inputs, dim=m, normalization=normalization)
        ops_out = torch.stack(ops_out, dim=2)

        output = torch.einsum('dsb,ndbi->nsi', self.coeffs, ops_out)  # N x S x m x m

        output = output + self.bias

        return output
    
class equi_1_to_2(torch.nn.Module):
    """equivariant nn layer."""

    def __init__(self, input_depth, output_depth, device):
        super(equi_1_to_2, self).__init__()
        self.basis_dimension = 5
        self.device = device
#         self.coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
        self.coeffs_values = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.cuda()

        self.coeffs = torch.nn.Parameter(self.coeffs_values, requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
       
    def ops_1_to_2(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m
        sum_all = torch.sum(inputs, dim=2, keepdims=True)  # N x D x 1

        # op1 - (123) - place on diag
        op1 = torch.diag_embed(inputs)  # N x D x m x m

        # op2 - (123) + (12)(3) - tile sum on diag
        op2 = torch.diag_embed(sum_all.repeat(1, 1, dim))  # N x D x m x m

        # op3 - (123) + (13)(2) - tile element i in row i
        op3 = torch.unsqueeze(inputs, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

        # op4 - (123) + (23)(1) - tile element i in col i
        op4 = torch.unsqueeze(inputs, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

        # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
        op5 = torch.unsqueeze(sum_all, dim=3).repeat(1, 1, dim, dim)  # N x D x m x m

        if normalization is not None:
            float_dim = dim.type(torch.float32)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)
                op5 = torch.div(op5, float_dim)

        return [op1, op2, op3, op4, op5]

    

    def forward(self, inputs, normalization='inf'):
        m = torch.tensor(inputs.shape[2], dtype=torch.int32, device=self.device)  # extract dimension

        ops_out = self.ops_1_to_2(inputs=inputs, dim=m, normalization=normalization)
        ops_out = torch.stack(ops_out, dim=2)

        output = torch.einsum('dsb,ndbij->nsij', self.coeffs, ops_out)  # N x S x m x m

        output = output + self.bias

        return output
    
class equi_1_to_1(torch.nn.Module):
    """equivariant nn layer."""

    def __init__(self, input_depth, output_depth, device):
        super(equi_1_to_1, self).__init__()
        self.basis_dimension = 2
        self.device = device
#         self.coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
        self.coeffs_values = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.cuda()

        self.coeffs = torch.nn.Parameter(self.coeffs_values, requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
       

    def ops_1_to_1(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m
        sum_all = torch.sum(inputs, dim=2, keepdims=True)  # N x D x 1

        # op1 - (12) - identity
        op1 = inputs  # N x D x m

        # op2 - (1)(2) - tile sum of all
        op2 = sum_all.repeat(1, 1, dim)  # N x D x m

        if normalization is not None:
            float_dim = dim.type(torch.float32)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)

        return [op1, op2]

    def forward(self, inputs, normalization='inf'):
        m = torch.tensor(inputs.shape[2], dtype=torch.int32, device=self.device)  # extract dimension

        ops_out = self.ops_1_to_1(inputs=inputs, dim=m, normalization=normalization)
        ops_out = torch.stack(ops_out, dim=2)

        output = torch.einsum('dsb,ndbi->nsi', self.coeffs, ops_out)  # N x S x m x m

        output = output + self.bias

        return output
    
class equi_2_to_0(torch.nn.Module):
    """equivariant nn layer."""

    def __init__(self, input_depth, output_depth, device):
        super(equi_2_to_0, self).__init__()
        self.basis_dimension = 2
        self.device = device
#         self.coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
        self.coeffs_values = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.cuda()

        self.coeffs = torch.nn.Parameter(self.coeffs_values, requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros((1, output_depth), dtype=torch.float32), requires_grad=True)
       

    def ops_2_to_0(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
        diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
        sum_diag_part = torch.sum(diag_part, dim=2)  # N x D
        sum_of_rows = torch.sum(inputs, dim=3)  # N x D x m
        sum_all = torch.sum(sum_of_rows, dim=2)  # N x D

        # op1 -
        op1 = sum_diag_part  # N x D

        # op2 -
        op2 = sum_all  # N x D


        if normalization is not None:
            float_dim = dim.type(torch.float32)
            if normalization is 'inf':
                op1 = torch.div(op1, float_dim)
                op2 = torch.div(op2, float_dim ** 2)

        return [op1, op2]


    def forward(self, inputs, normalization='inf'):
        m = torch.tensor(inputs.shape[3], dtype=torch.int32, device=self.device)  # extract dimension

        ops_out = self.ops_2_to_0(inputs=inputs, dim=m, normalization=normalization)
        ops_out = torch.stack(ops_out, dim=2)

#         print(f'self.coeffs shape : {self.coeffs.shape}')
#         print(f'ops_out shape : {ops_out.shape}')
        output = torch.einsum('dsb,ndb->ns', self.coeffs, ops_out)  # N x S x m x m

        output = output + self.bias

        return output
    
class equi_1_to_0(torch.nn.Module):
    """equivariant nn layer."""

    def __init__(self, input_depth, output_depth, device):
        super(equi_1_to_0, self).__init__()
        self.basis_dimension = 1
        self.device = device
#         self.coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
        self.coeffs_values = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.cuda()

        self.coeffs = torch.nn.Parameter(self.coeffs_values, requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros((1, output_depth), dtype=torch.float32), requires_grad=True)
       

    def ops_1_to_0(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m
        sum_all = torch.sum(inputs, dim=2)  # N x D

        # op1 - (12) - identity
        op1 = sum_all  # N x D

        if normalization is not None:
            float_dim = dim.type(torch.float32)
            if normalization is 'inf':
                op1 = torch.div(op1, float_dim)

        return [op1]


    def forward(self, inputs, normalization='inf'):
        m = torch.tensor(inputs.shape[2], dtype=torch.int32, device=self.device)  # extract dimension

        ops_out = self.ops_1_to_0(inputs=inputs, dim=m, normalization=normalization)
        ops_out = torch.stack(ops_out, dim=2)

        output = torch.einsum('dsb,ndb->ns', self.coeffs, ops_out)  # N x S x m x m

        output = output + self.bias

        return output


class MessagePassing2(torch.nn.Module):
    r"""Base class for creating message passing layers of the form
    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right),
    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_gnn.html>`__ for the accompanying tutorial.
    Args:
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"` or :obj:`None`).
            (default: :obj:`"add"`)
        flow (string, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`-2`)
    """

    special_args: Set[str] = {
        'edge_index', 'adj_t', 'edge_index_i', 'edge_index_j', 'size',
        'size_i', 'size_j', 'ptr', 'index', 'dim_size'
    }

    def __init__(self, aggr: Optional[str] = "add",
                 flow: str = "source_to_target", node_dim: int = -2):

        super(MessagePassing2, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max', None]

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.node_dim = node_dim

        self.inspector = Inspector(self)
        self.inspector.inspect(self.message)
        self.inspector.inspect(self.aggregate, pop_first=True)
        self.inspector.inspect(self.message_and_aggregate, pop_first=True)
        self.inspector.inspect(self.update, pop_first=True)

        self.__user_args__ = self.inspector.keys(
            ['message', 'aggregate', 'update']).difference(self.special_args)
        self.__fused_user_args__ = self.inspector.keys(
            ['message_and_aggregate', 'update']).difference(self.special_args)

        # Support for "fused" message passing.
        self.fuse = self.inspector.implements('message_and_aggregate')

        # Support for GNNExplainer.
        self.__explain__ = False
        self.__edge_mask__ = None

    def __check_input__(self, edge_index, size):
        the_size: List[Optional[int]] = [None, None]

        if isinstance(edge_index, Tensor):
            assert edge_index.dtype == torch.long
            assert edge_index.dim() == 2
            assert edge_index.size(0) == 2
            if size is not None:
                the_size[0] = size[0]
                the_size[1] = size[1]
            return the_size

        elif isinstance(edge_index, SparseTensor):
            if self.flow == 'target_to_source':
                raise ValueError(
                    ('Flow direction "target_to_source" is invalid for '
                     'message propagation via `torch_sparse.SparseTensor`. If '
                     'you really want to make use of a reverse message '
                     'passing flow, pass in the transposed sparse tensor to '
                     'the message passing module, e.g., `adj_t.t()`.'))
            the_size[0] = edge_index.sparse_size(1)
            the_size[1] = edge_index.sparse_size(0)
            return the_size

        raise ValueError(
            ('`MessagePassing.propagate` only supports `torch.LongTensor` of '
             'shape `[2, num_messages]` or `torch_sparse.SparseTensor` for '
             'argument `edge_index`.'))

    def __set_size__(self, size: List[Optional[int]], dim: int, src: Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim):
            raise ValueError(
                (f'Encountered tensor with size {src.size(self.node_dim)} in '
                 f'dimension {self.node_dim}, but expected size {the_size}.'))

    def __lift__(self, src, edge_index, dim):
        if isinstance(edge_index, Tensor):
            index = edge_index[dim]
            return src.index_select(self.node_dim, index)
        elif isinstance(edge_index, SparseTensor):
            if dim == 1:
                rowptr = edge_index.storage.rowptr()
                rowptr = expand_left(rowptr, dim=self.node_dim, dims=src.dim())
                return gather_csr(src, rowptr)
            elif dim == 0:
                col = edge_index.storage.col()
                return src.index_select(self.node_dim, col)
        raise ValueError

    def __collect__(self, args, edge_index, size, kwargs):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        out = {}
        for arg in args:
            if arg[-2:] not in ['_i', '_j']:
                out[arg] = kwargs.get(arg, Parameter.empty)
            else:
                dim = 0 if arg[-2:] == '_j' else 1
                data = kwargs.get(arg[:-2], Parameter.empty)

                if isinstance(data, (tuple, list)):
                    assert len(data) == 2
                    if isinstance(data[1 - dim], Tensor):
                        self.__set_size__(size, 1 - dim, data[1 - dim])
                    data = data[dim]

                if isinstance(data, Tensor):
                    self.__set_size__(size, dim, data)
                    data = self.__lift__(data, edge_index,
                                         j if arg[-2:] == '_j' else i)

                out[arg] = data

        if isinstance(edge_index, Tensor):
            out['adj_t'] = None
            out['edge_index'] = edge_index
            out['edge_index_i'] = edge_index[i]
            out['edge_index_j'] = edge_index[j]
            out['ptr'] = None
        elif isinstance(edge_index, SparseTensor):
            out['adj_t'] = edge_index
            out['edge_index'] = None
            out['edge_index_i'] = edge_index.storage.row()
            out['edge_index_j'] = edge_index.storage.col()
            out['ptr'] = edge_index.storage.rowptr()
            out['edge_weight'] = edge_index.storage.value()
            out['edge_attr'] = edge_index.storage.value()
            out['edge_type'] = edge_index.storage.value()

        out['index'] = out['edge_index_i']
        out['size'] = size
        out['size_i'] = size[1] or size[0]
        out['size_j'] = size[0] or size[1]
        out['dim_size'] = out['size_i']

        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.
        Args:
            edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self.__explain__):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                         kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            out = self.message(**msg_kwargs)
#             print(f'message out shape : {out.shape}')

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self.__explain__:
                edge_mask = self.__edge_mask__.sigmoid()
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)
            print(f'aggregate out shape : {out.shape}')

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

    def message(self, x_j: Tensor) -> Tensor:
        r"""Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :obj:`edge_index`.
        This function can take any argument as input which was initially
        passed to :meth:`propagate`.
        Furthermore, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        return x_j

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.
        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.
        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        if ptr is not None:
            ptr = expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            print(f'aggregate function inputs : {inputs.shape}')
            print(f'aggregate function index : {index.shape}')
            print(f'aggregate function index : {index}')
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        r"""Fuses computations of :func:`message` and :func:`aggregate` into a
        single function.
        If applicable, this saves both time and memory since messages do not
        explicitly need to be materialized.
        This function will only gets called in case it is implemented and
        propagation takes place based on a :obj:`torch_sparse.SparseTensor`.
        """
        raise NotImplementedError

    def update(self, inputs: Tensor) -> Tensor:
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """
        return inputs

    @torch.jit.unused
    def jittable(self, typing: Optional[str] = None):
        r"""Analyzes the :class:`MessagePassing` instance and produces a new
        jittable module.
        Args:
            typing (string, optional): If given, will generate a concrete
                instance with :meth:`forward` types based on :obj:`typing`,
                *e.g.*: :obj:`"(Tensor, Optional[Tensor]) -> Tensor"`.
        """
        # Find and parse `propagate()` types to format `{arg1: type1, ...}`.
        if hasattr(self, 'propagate_type'):
            prop_types = {
                k: sanitize(str(v))
                for k, v in self.propagate_type.items()
            }
        else:
            source = inspect.getsource(self.__class__)
            match = re.search(r'#\s*propagate_type:\s*\((.*)\)', source)
            if match is None:
                raise TypeError(
                    'TorchScript support requires the definition of the types '
                    'passed to `propagate()`. Please specificy them via\n\n'
                    'propagate_type = {"arg1": type1, "arg2": type2, ... }\n\n'
                    'or via\n\n'
                    '# propagate_type: (arg1: type1, arg2: type2, ...)\n\n'
                    'inside the `MessagePassing` module.')
            prop_types = split_types_repr(match.group(1))
            prop_types = dict([re.split(r'\s*:\s*', t) for t in prop_types])

        # Parse `__collect__()` types to format `{arg:1, type1, ...}`.
        collect_types = self.inspector.types(
            ['message', 'aggregate', 'update'])

        # Collect `forward()` header, body and @overload types.
        forward_types = parse_types(self.forward)
        forward_types = [resolve_types(*types) for types in forward_types]
        forward_types = list(chain.from_iterable(forward_types))

        keep_annotation = len(forward_types) < 2
        forward_header = func_header_repr(self.forward, keep_annotation)
        forward_body = func_body_repr(self.forward, keep_annotation)

        if keep_annotation:
            forward_types = []
        elif typing is not None:
            forward_types = []
            forward_body = 8 * ' ' + f'# type: {typing}\n{forward_body}'

        root = os.path.dirname(osp.realpath(__file__))
        with open(osp.join(root, 'message_passing.jinja'), 'r') as f:
            template = Template(f.read())

        uid = uuid1().hex[:6]
        cls_name = f'{self.__class__.__name__}Jittable_{uid}'
        jit_module_repr = template.render(
            uid=uid,
            module=str(self.__class__.__module__),
            cls_name=cls_name,
            parent_cls_name=self.__class__.__name__,
            prop_types=prop_types,
            collect_types=collect_types,
            user_args=self.__user_args__,
            forward_header=forward_header,
            forward_types=forward_types,
            forward_body=forward_body,
            msg_args=self.inspector.keys(['message']),
            aggr_args=self.inspector.keys(['aggregate']),
            msg_and_aggr_args=self.inspector.keys(['message_and_aggregate']),
            update_args=self.inspector.keys(['update']),
            check_input=inspect.getsource(self.__check_input__)[:-1],
            lift=inspect.getsource(self.__lift__)[:-1],
        )

        # Instantiate a class from the rendered JIT module representation.
        cls = class_from_module_repr(cls_name, jit_module_repr)
        module = cls.__new__(cls)
        module.__dict__ = self.__dict__.copy()
        module.jittable = None

        return module

# class eq_mp(MessagePassing2):
#     def __init__(self, in_channels, out_channels):
#         super(eq_mp, self).__init__(aggr='add', flow='target_to_source')
#         self.lin = torch.nn.Linear(in_channels, out_channels)

#     def forward(self, x, edge_index, edge_attr):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]
#         # edge_attr has shape [E, in_channels]

#         # Step 4-5: Start propagating messages.
#         return self.propagate(edge_index, x=x, edge_attr=edge_attr)

#     def message(self, x_i, x_j, edge_attr):
#         print(f'x_i shape : {x_i.shape}')
#         print(f'x_j shape : {x_j.shape}')
#         print(f'edge_attr shape : {edge_attr.shape}')
        
# #         ## basis 1
# #         out1 = x_i
        
# #         ## basis 2
# #         out2 = x_j
        
# #         ## basis 3
# #         out3 = edge_attr
        
# #         ## basis 4
# #         out4 = edge_attr

# #         ## basis 5
# #         out5 = edge_attr
        
#         # Step 4: Return necessary node and edge features.
#         return x_i, x_j, edge_attr
    
#     def aggregate(self, inputs, index, ptr = None, dim_size = None):

#         if ptr is not None:
#             ptr = expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
#             return segment_csr(inputs, ptr, reduce=self.aggr)
#         else:
#             x_i = inputs[0]
#             x_j = inputs[1]
#             edge_attr = inputs[2]
#             print(f'aggregate function x_i : {x_i.shape}')
#             print(f'aggregate function x_j : {x_j.shape}')
#             print(f'aggregate function edge_attr : {edge_attr.shape}')
#             print(f'aggregate function x_i : {x_i}')
#             print(f'aggregate function x_j : {x_j}')
#             print(f'aggregate function edge_attr : {edge_attr}')
#             print(f'aggregate function index : {index.shape}')
#             print(f'aggregate function index : {index}')
#             print(f'aggregate reduce : {self.aggr}')
#             print(f'aggregate dim_size : {dim_size}')
            
#             ops1 = scatter(x_i, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
#             print(f'ops1 : {ops1.shape}')
#             ops2 = scatter(x_j, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
#             print(f'ops2 : {ops2.shape}')
#             ops3 = scatter(edge_attr, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
#             print(f'ops3 : {ops3.shape}')
#             ops4 = scatter(edge_attr, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
#             print(f'ops4 : {ops4.shape}')
            
#             return ops1
        
        
class eq_mp_22(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(eq_mp_22, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)
        self.lin3 = torch.nn.Linear(in_channels, out_channels)
        self.lin4 = torch.nn.Linear(in_channels, out_channels)
        self.lin5 = torch.nn.Linear(in_channels, out_channels)
        self.lin6 = torch.nn.Linear(in_channels, out_channels)
        self.lin7 = torch.nn.Linear(in_channels, out_channels)
        self.lin8 = torch.nn.Linear(in_channels, out_channels)
        self.lin9 = torch.nn.Linear(in_channels, out_channels)
        self.lin10 = torch.nn.Linear(in_channels, out_channels)
        self.lin11 = torch.nn.Linear(in_channels, out_channels)
        self.lin12 = torch.nn.Linear(in_channels, out_channels)
        self.lin13 = torch.nn.Linear(in_channels, out_channels)
        self.lin14 = torch.nn.Linear(in_channels, out_channels)
        self.lin15 = torch.nn.Linear(in_channels, out_channels)
        
    def op5(self, x, edge_index, edge_attr, batch, batch_edge):
        hyper_edge_attr = []
        hyper_edge_index = []
        for i in range(x.shape[0]):
            mask = (edge_index[0] != i) & (edge_index[1] != i) & (batch_edge == batch[i])
            hyper_edge_attr.append(torch.mean(edge_attr[mask], dim=0))
#             hyper_edge_index.append(torch.ones(edge_attr[mask].shape[0], dtype=torch.int64).to(x.device)*i)
        hyper_edge_attr = torch.stack(hyper_edge_attr, dim=0)
#         hyper_edge_index = torch.cat(hyper_edge_index, dim=0)
        return hyper_edge_attr#, hyper_edge_index

    def op8(self, edge_index, edge_attr, batch, batch_edge):
        edge_attr_ij = []
        edge_index_ij = []
        for i, j in zip(edge_index[0], edge_index[1]):
            mask = (edge_index[0] == i) & (edge_index[1] != j) & (batch_edge == batch[i]) & (batch_edge == batch[j])
            edge_attr_ij.append(torch.mean(edge_attr[mask], dim=0))
#             edge_index_ij.append(torch.ones(edge_attr[mask].shape[0], dtype=torch.int64).to(edge_attr.device)*i)
        edge_attr_ij = torch.stack(edge_attr_ij, dim=0)
#         edge_index_ij = torch.cat(edge_index_ij, dim=0)
        return edge_attr_ij#, edge_index_ij

    def op9(self, edge_index, edge_attr, batch, batch_edge):
        edge_attr_ij = []
        edge_index_ij = []
        for i, j in zip(edge_index[0], edge_index[1]):
            mask = (edge_index[0] != j) & (edge_index[1] == i) & (batch_edge == batch[i]) & (batch_edge == batch[j])
            edge_attr_ij.append(torch.mean(edge_attr[mask], dim=0))
#             edge_index_ij.append(torch.ones(edge_attr[mask].shape[0], dtype=torch.int64).to(edge_attr.device)*i)
        edge_attr_ij = torch.stack(edge_attr_ij, dim=0)
#         edge_index_ij = torch.cat(edge_index_ij, dim=0)
        return edge_attr_ij#, edge_index_ij

    def op10(self, x, edge_index, edge_attr, batch, batch_edge):
        x_ij = []
        x_index_ij = []
        idx = torch.arange(x.shape[0]).to(x.device)
#         print(len(batch[batch==0]))
#         print(len(batch[batch==1]))
        if len(batch[batch==0])==2 or len(batch[batch==1])==2:
            printmask = True
        else:
            printmask = False
        for i, j in zip(edge_index[0], edge_index[1]):
            mask = (idx != i) & (idx != j) & (batch == batch[i]) & (batch == batch[j])
            if printmask:
                print(f'op10 count nonzero mask : {torch.count_nonzero(mask)}')
            x_ij.append(torch.mean(x[mask],dim=0))
#             x_index_ij.append(torch.ones(x[mask].shape[0], dtype=torch.int64).to(edge_attr.device)*i)
        x_ij = torch.stack(x_ij, dim=0)
#         x_index_ij = torch.cat(x_index_ij, dim=0)
        return x_ij#, x_index_ij

    def op12(self, edge_index, edge_attr, batch, batch_edge):
        edge_attr_ij = []
        edge_index_ij = []
        for i, j in zip(edge_index[0], edge_index[1]):
            mask = (edge_index[0] == j) & (edge_index[1] == i) & (batch_edge == batch[i]) & (batch_edge == batch[j])
            edge_attr_ij.append(torch.mean(edge_attr[mask], dim=0))
#             edge_index_ij.append(torch.ones(edge_attr[mask].shape[0], dtype=torch.int64).to(edge_attr.device)*i)
        edge_attr_ij = torch.stack(edge_attr_ij, dim=0)
#         edge_index_ij = torch.cat(edge_index_ij, dim=0)
        return edge_attr_ij#, edge_index_ij

    def op13(self, edge_index, edge_attr, batch, batch_edge):
        edge_attr_ij = []
        edge_index_ij = []
        for i, j in zip(edge_index[0], edge_index[1]):
            mask = (edge_index[0] == j) & (edge_index[1] != i) & (batch_edge == batch[i]) & (batch_edge == batch[j])
            edge_attr_ij.append(torch.mean(edge_attr[mask], dim=0))
#             edge_index_ij.append(torch.ones(edge_attr[mask].shape[0], dtype=torch.int64).to(edge_attr.device)*i)
        edge_attr_ij = torch.stack(edge_attr_ij, dim=0)
#         edge_index_ij = torch.cat(edge_index_ij, dim=0)
        return edge_attr_ij#, edge_index_ij

    def op14(self, edge_index, edge_attr, batch, batch_edge):
        edge_attr_ij = []
        edge_index_ij = []
        for i, j in zip(edge_index[0], edge_index[1]):
            mask = (edge_index[0] != i) & (edge_index[1] == j) & (batch_edge == batch[i]) & (batch_edge == batch[j])
            edge_attr_ij.append(torch.mean(edge_attr[mask], dim=0))
#             edge_index_ij.append(torch.ones(edge_attr[mask].shape[0], dtype=torch.int64).to(edge_attr.device)*i)
        edge_attr_ij = torch.stack(edge_attr_ij, dim=0)
#         edge_index_ij = torch.cat(edge_index_ij, dim=0)
        return edge_attr_ij#, edge_index_ij

    def op15(self, edge_index, edge_attr, batch, batch_edge):
        edge_attr_ij = []
        edge_index_ij = []
        for i, j in zip(edge_index[0], edge_index[1]):
            mask = (edge_index[0] != i) & (edge_index[1] != j) & (batch_edge == batch[i]) & (batch_edge == batch[j])
            edge_attr_ij.append(torch.mean(edge_attr[mask], dim=0))
#             edge_index_ij.append(torch.ones(edge_attr[mask].shape[0], dtype=torch.int64).to(edge_attr.device)*i)
        edge_attr_ij = torch.stack(edge_attr_ij, dim=0)
#         edge_index_ij = torch.cat(edge_index_ij, dim=0)
        return edge_attr_ij#, edge_index_ij
        

    def forward(self, x, edge_index, edge_attr, batch):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, in_channels]
                
        x_src = x[edge_index[0]]
        x_dst = x[edge_index[1]]
        
#         print(f'x shape : {x.shape}')
#         print(f'batch : {batch}')
#         print(f'x_src shape : {x_src.shape}')
#         print(f'x_dst shape : {x_dst.shape}')
#         print(f'edge_index shape : {edge_index.shape}')
#         print(f'edge_index shape : {edge_index}')
#         print(f'edge_attr shape : {edge_attr.shape}')
#         print(f'edge_index[0] : {edge_index[0]}')
#         print(f'edge_index[1] : {edge_index[1]}')
        batch_edge = batch[edge_index[0]]

        
        ## Node update ops
        ops1 = x
#         print(f'ops1 : {ops1.shape}')
#         print(f'ops1 : {ops1}')
        ops2 = scatter(x_dst, edge_index[0], dim=-2, dim_size=x.shape[0], reduce='mean')
#         print(f'ops2 : {ops2.shape}')
#         print(f'ops2 : {ops2}')
        ops3 = scatter(edge_attr, edge_index[0], dim=-2, dim_size=x.shape[0], reduce='mean')
#         print(f'ops3 : {ops3.shape}')
#         print(f'ops3 : {ops3}')
        ops4 = scatter(edge_attr, edge_index[1], dim=-2, dim_size=x.shape[0], reduce='mean')
#         print(f'ops4 : {ops4.shape}')
#         print(f'ops4 : {ops4}')
#         hyper_edge_attr, hyper_edge_index = self.op5(x, edge_index, edge_attr, batch, batch_edge)
#         ops5 = scatter(hyper_edge_attr, hyper_edge_index, dim=-2, dim_size=x.shape[0], reduce='mean')
        ops5 = self.op5(x, edge_index, edge_attr, batch, batch_edge)
#         print(f'ops5 : {ops5.shape}')
#         print(f'ops5 : {ops5}')
        
        ## Edge update ops
        ops6 = x_src
#         print(f'ops6 : {ops6.shape}')
#         print(f'ops6 : {ops6}')
        ops7 = edge_attr
#         print(f'ops7 : {ops7.shape}')
#         print(f'ops7 : {ops7}')
#         edge_attr_ij, edge_index_ij = self.op8(edge_index, edge_attr, batch, batch_edge)
#         ops8 = scatter(edge_attr_ij, edge_index_ij, dim=-2, dim_size=edge_attr.shape[0], reduce='mean')
        ops8 = self.op8(edge_index, edge_attr, batch, batch_edge)
#         print(f'ops8 : {ops8.shape}')
#         print(f'ops8 : {ops8}')
#         edge_attr_ij, edge_index_ij = self.op9(edge_index, edge_attr, batch, batch_edge)
#         ops9 = scatter(edge_attr_ij, edge_index_ij, dim=-2, dim_size=edge_attr.shape[0], reduce='mean')
        ops9 = self.op9(edge_index, edge_attr, batch, batch_edge)
#         print(f'ops9 : {ops9.shape}')
#         print(f'ops9 : {ops9}')
#         x_ij, x_index_ij = self.op10(x, edge_index, edge_attr, batch, batch_edge)
#         ops10 = scatter(x_ij, x_index_ij, dim=-2, dim_size=edge_attr.shape[0], reduce='mean')
        ops10 = self.op10(x, edge_index, edge_attr, batch, batch_edge)
#         print(f'ops10 : {ops10.shape}')
#         print(f'ops10 : {ops10}')
        ops11 = x_dst
#         print(f'ops11 : {ops11.shape}')
#         print(f'ops11 : {ops11}')
#         edge_attr_ij, edge_index_ij = self.op12(edge_index, edge_attr, batch, batch_edge)
#         ops12 = scatter(edge_attr_ij, edge_index_ij, dim=-2, dim_size=edge_attr.shape[0], reduce='mean')
        ops12 = self.op12(edge_index, edge_attr, batch, batch_edge)
#         print(f'ops12 : {ops12.shape}')
#         print(f'ops12 : {ops12}')
#         edge_attr_ij, edge_index_ij = self.op13(edge_index, edge_attr, batch, batch_edge)
#         ops13 = scatter(edge_attr_ij, edge_index_ij, dim=-2, dim_size=edge_attr.shape[0], reduce='mean')
        ops13 = self.op13(edge_index, edge_attr, batch, batch_edge)
#         print(f'ops13 : {ops13.shape}')
#         print(f'ops13 : {ops13}')
#         edge_attr_ij, edge_index_ij = self.op14(edge_index, edge_attr, batch, batch_edge)
#         ops14 = scatter(edge_attr_ij, edge_index_ij, dim=-2, dim_size=edge_attr.shape[0], reduce='mean')
        ops14 = self.op14(edge_index, edge_attr, batch, batch_edge)
#         print(f'ops14 : {ops14.shape}')
#         print(f'ops14 : {ops14}')
#         edge_attr_ij, edge_index_ij = self.op15(edge_index, edge_attr, batch, batch_edge)
#         ops15 = scatter(edge_attr_ij, edge_index_ij, dim=-2, dim_size=edge_attr.shape[0], reduce='mean')
        ops15 = self.op15(edge_index, edge_attr, batch, batch_edge)
#         print(f'ops15 : {ops15.shape}')
#         print(f'ops15 : {ops15}')
        
        x = torch.mean(torch.stack([self.lin1(ops1), self.lin2(ops2), self.lin3(ops3), self.lin4(ops4), self.lin5(ops5)], dim=-1), dim=-1)
#         print(f'x : {x.shape}')
#         print(f'x : {x}')
        edge_attr = torch.mean(torch.stack([self.lin6(ops6), self.lin7(ops7), self.lin8(ops8), self.lin9(ops9), self.lin10(ops10), self.lin11(ops11), self.lin12(ops12), self.lin13(ops13), self.lin14(ops14), self.lin15(ops15)], dim=-1), dim=-1)
#         print(f'edge_attr : {edge_attr.shape}')
#         print(f'edge_attr : {edge_attr}')
                
        return x, edge_index, edge_attr
    
class eq_mp_20(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(eq_mp_20, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch, num_graphs):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, in_channels]
                
#         print(f'x shape : {x.shape}')
#         print(f'edge_index shape : {edge_index.shape}')
        x_src = x[edge_index[0]]
        x_dst = x[edge_index[1]]
        
#         print(f'x shape : {x[:60,:]}')
#         print(f'x_src shape : {x_src.shape}')
#         print(f'x_dst shape : {x_dst.shape}')
#         print(f'edge_index shape : {edge_index.shape}')
#         print(f'edge_attr shape : {edge_attr}')
#         print(f'edge_index[0] : {edge_index[0]}')
#         print(f'edge_index[1] : {edge_index[1]}')
        batch_edge = batch[edge_index[0]]
#         print(f'batch_edge shape : {batch_edge.shape}')
#         print(f'batch_edge : {batch_edge}')
        
        ## Node update ops
        ops1 = scatter(x, batch, dim=-2, dim_size=num_graphs, reduce='mean')
#         print(f'ops1 : {ops1}')
        ops2 = scatter(edge_attr, batch_edge, dim=-2, dim_size=num_graphs, reduce='mean')
#         print(f'ops2 : {ops2}')
        
        g = torch.mean(torch.stack([self.lin1(ops1), self.lin2(ops2)], dim=-1), dim=-1)
        
        return g

        
        
# class equi_2_to_2(torch.nn.Module):
#     """equivariant nn layer."""

#     def __init__(self, input_depth, output_depth, device):
#         super(equi_2_to_2, self).__init__()
#         self.basis_dimension = 15
#         self.device = device
# #         self.coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
#         self.coeffs_values = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.cuda()

#         self.coeffs = torch.nn.Parameter(self.coeffs_values, requires_grad=True)
#         self.diag_bias = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
#         self.all_bias = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
        
#     def ops_2_to_2(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
# #         print(f'input shape : {inputs.shape}')
#         diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
# #         print(f'diag_part shape : {diag_part.shape}')
#         sum_diag_part = torch.sum(diag_part, dim=2, keepdim=True)  # N x D x 1
# #         print(f'sum_diag_part shape : {sum_diag_part.shape}')
#         sum_of_rows = torch.sum(inputs, dim=3)  # N x D x m
# #         print(f'sum_of_rows shape : {sum_of_rows.shape}')
#         sum_of_cols = torch.sum(inputs, dim=2)  # N x D x m
# #         print(f'sum_of_cols shape : {sum_of_cols.shape}')
#         sum_all = torch.sum(sum_of_rows, dim=2)  # N x D
# #         print(f'sum_all shape : {sum_all.shape}')

#         # op1 - (1234) - extract diag
#         op1 = torch.diag_embed(diag_part)  # N x D x m x m

#         # op2 - (1234) + (12)(34) - place sum of diag on diag
#         op2 = torch.diag_embed(sum_diag_part.repeat(1, 1, dim))  # N x D x m x m

#         # op3 - (1234) + (123)(4) - place sum of row i on diag ii
#         op3 = torch.diag_embed(sum_of_rows)  # N x D x m x m

#         # op4 - (1234) + (124)(3) - place sum of col i on diag ii
#         op4 = torch.diag_embed(sum_of_cols)  # N x D x m x m

#         # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
#         op5 = torch.diag_embed(torch.unsqueeze(sum_all, dim=2).repeat(1, 1, dim))  # N x D x m x m

#         # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
#         op6 = torch.unsqueeze(sum_of_cols, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

#         # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
#         op7 = torch.unsqueeze(sum_of_rows, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

#         # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
#         op8 = torch.unsqueeze(sum_of_cols, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

#         # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
#         op9 = torch.unsqueeze(sum_of_rows, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

#         # op10 - (1234) + (14)(23) - identity
#         op10 = inputs  # N x D x m x m

#         # op11 - (1234) + (13)(24) - transpose
#         op11 = inputs.permute(0, 1, 3, 2)  # N x D x m x m

#         # op12 - (1234) + (234)(1) - place ii element in row i
#         op12 = torch.unsqueeze(diag_part, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

#         # op13 - (1234) + (134)(2) - place ii element in col i
#         op13 = torch.unsqueeze(diag_part, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

#         # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
#         op14 = torch.unsqueeze(sum_diag_part, dim=3).repeat(1, 1, dim, dim)   # N x D x m x m

#         # op15 - sum of all ops - place sum of all entries in all entries
#         op15 = torch.unsqueeze(torch.unsqueeze(sum_all, dim=2), dim=3).repeat(1, 1, dim, dim)  # N x D x m x m

#         if normalization is not None:
#             float_dim = dim.type(torch.FloatTensor)
#             if normalization is 'inf':
#                 op2 = torch.div(op2, float_dim)
#                 op3 = torch.div(op3, float_dim)
#                 op4 = torch.div(op4, float_dim)
#                 op5 = torch.div(op5, float_dim**2)
#                 op6 = torch.div(op6, float_dim)
#                 op7 = torch.div(op7, float_dim)
#                 op8 = torch.div(op8, float_dim)
#                 op9 = torch.div(op9, float_dim)
#                 op14 = torch.div(op14, float_dim)
#                 op15 = torch.div(op15, float_dim**2)

#         return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]

#     def forward(self, inputs, normalization='inf'):
#         m = torch.tensor(inputs.shape[3], dtype=torch.int32, device=self.device)  # extract dimension

# #         print(f'inputs device : {inputs.device}')
#         ops_out = self.ops_2_to_2(inputs=inputs, dim=m, normalization=normalization)
# #         for idx, op in enumerate(ops_out):
# #             print(f'ops_out{idx} : {op.shape}')
#         ops_out = torch.stack(ops_out, dim=2)

# #         print(f'self.coeffs device : {self.coeffs.device}')
# #         print(f'ops_out : {ops_out}')
#         output = torch.einsum('dsb,ndbij->nsij', self.coeffs.double(), ops_out)  # N x S x m x m

#         # bias
# #         print(f'diag_bias shape : {self.diag_bias.shape}')
# #         print(f'eye shape : {torch.eye(torch.tensor(inputs.shape[3], dtype=torch.int32, device=self.device), device=self.device).shape}')
# #         mat_diag_bias = torch.mul(torch.unsqueeze(torch.unsqueeze(torch.eye(torch.tensor(inputs.shape[3], dtype=torch.int32, device=self.device), device=self.device), 0), 0), self.diag_bias)
#         mat_diag_bias = self.diag_bias.expand(-1,-1,inputs.shape[3],inputs.shape[3])
#         mat_diag_bias = torch.mul(mat_diag_bias, torch.eye(inputs.shape[3], device=self.device))
#         output = output + self.all_bias + mat_diag_bias
# #         print(f'mat_diag_bias shape : {mat_diag_bias.shape}')

#         return output

class equi_multi_local(torch.nn.Module):
    """equivariant nn layer."""

    def __init__(self, input_depth, output_depth, device, subgroups, repsin, repsout, hops):
        super(equi_multi_local, self).__init__()
        self.repsin = repsin
        self.repsout = repsout
        self.hops = hops
        self.device = device
        self.basis_dimension_22 = 15
        self.basis_dimension_21 = 5
        self.basis_dimension_12 = 5
        self.basis_dimension_20 = 2
        self.basis_dimension_02 = 2
        self.basis_dimension_11 = 2
        self.basis_dimension_10 = 1
        self.basis_dimension_01 = 1
        self.basis_dimension_00 = 1
       
        if 2 in repsin and 2 in repsout:
            coeffs_values_22 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_22), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
            self.coeffs22 = {}
            for sub in subgroups:
                self.coeffs22[f'{sub}'] = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.diag_bias22 = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
            self.all_bias22 = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
       
        if 2 in repsin and 1 in repsout:
            coeffs_values_21 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_21), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
            self.coeffs21 = {}
            for sub in subgroups:
                self.coeffs21[f'{sub}'] = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.bias21 = torch.nn.Parameter(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
           
        if 1 in repsin and 2 in repsout:
            coeffs_values_12 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_12), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
            self.coeffs12 = {}
            for sub in subgroups:
                self.coeffs12[f'{sub}'] = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.bias12 = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
           
        if 2 in repsin and 0 in repsout:
            coeffs_values_20 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_20), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
            self.coeffs20 = {}
            for sub in subgroups:
                self.coeffs20[f'{sub}'] = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.bias20 = torch.nn.Parameter(torch.zeros((1, output_depth), dtype=torch.float32), requires_grad=True)
           
        if 0 in repsin and 2 in repsout:
            coeffs_values_02 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_02), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
            self.coeffs02 = {}
            for sub in subgroups:
                self.coeffs02[f'{sub}'] = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.bias02 = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
           
        if 1 in repsin and 1 in repsout:
            coeffs_values_11 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_11), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
            self.coeffs11 = {}
            for sub in subgroups:
                self.coeffs11[f'{sub}'] = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.bias11 = torch.nn.Parameter(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
           
        if 1 in repsin and 0 in repsout:
            coeffs_values_10 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_10), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
            self.coeffs10 = {}
            for sub in subgroups:
                self.coeffs10[f'{sub}'] = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.bias10 = torch.nn.Parameter(torch.zeros((1, output_depth), dtype=torch.float32), requires_grad=True)
           
        if 0 in repsin and 1 in repsout:
            coeffs_values_01 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_01), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
            self.coeffs01 = {}
            for sub in subgroups:
                self.coeffs01[f'{sub}'] = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.bias01 = torch.nn.Parameter(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
           
        if 0 in repsin and 0 in repsout:
            coeffs_values_00 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_00), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
            self.coeffs00 = {}
            for sub in subgroups:
                self.coeffs00[f'{sub}'] = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.bias00 = torch.nn.Parameter(torch.zeros((1, output_depth), dtype=torch.float32), requires_grad=True)
       

    def ops_2_to_2(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
        diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
        sum_diag_part = torch.sum(diag_part, dim=2, keepdim=True)  # N x D x 1
        sum_of_rows = torch.sum(inputs, dim=3)  # N x D x m
        sum_of_cols = torch.sum(inputs, dim=2)  # N x D x m
        sum_all = torch.sum(sum_of_rows, dim=2)  # N x D

        # op1 - (1234) - extract diag
        op1 = torch.diag_embed(diag_part)  # N x D x m x m

        # op2 - (1234) + (12)(34) - place sum of diag on diag
        op2 = torch.diag_embed(sum_diag_part.repeat(1, 1, dim))  # N x D x m x m

        # op3 - (1234) + (123)(4) - place sum of row i on diag ii
        op3 = torch.diag_embed(sum_of_rows)  # N x D x m x m

        # op4 - (1234) + (124)(3) - place sum of col i on diag ii
        op4 = torch.diag_embed(sum_of_cols)  # N x D x m x m

        # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
        op5 = torch.diag_embed(torch.unsqueeze(sum_all, dim=2).repeat(1, 1, dim))  # N x D x m x m

        # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
        op6 = torch.unsqueeze(sum_of_cols, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

        # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
        op7 = torch.unsqueeze(sum_of_rows, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

        # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
        op8 = torch.unsqueeze(sum_of_cols, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

        # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
        op9 = torch.unsqueeze(sum_of_rows, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

        # op10 - (1234) + (14)(23) - identity
        op10 = inputs  # N x D x m x m

        # op11 - (1234) + (13)(24) - transpose
        op11 = inputs.permute(0, 1, 3, 2)  # N x D x m x m

        # op12 - (1234) + (234)(1) - place ii element in row i
        op12 = torch.unsqueeze(diag_part, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

        # op13 - (1234) + (134)(2) - place ii element in col i
        op13 = torch.unsqueeze(diag_part, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

        # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
        op14 = torch.unsqueeze(sum_diag_part, dim=3).repeat(1, 1, dim, dim)   # N x D x m x m

        # op15 - sum of all ops - place sum of all entries in all entries
        op15 = torch.unsqueeze(torch.unsqueeze(sum_all, dim=2), dim=3).repeat(1, 1, dim, dim)  # N x D x m x m

        if normalization is not None:
            float_dim = dim.type(torch.float32)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)
                op3 = torch.div(op3, float_dim)
                op4 = torch.div(op4, float_dim)
                op5 = torch.div(op5, float_dim**2)
                op6 = torch.div(op6, float_dim)
                op7 = torch.div(op7, float_dim)
                op8 = torch.div(op8, float_dim)
                op9 = torch.div(op9, float_dim)
                op14 = torch.div(op14, float_dim)
                op15 = torch.div(op15, float_dim**2)

        return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]
   
    def ops_2_to_1(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
        diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
        sum_diag_part = torch.sum(diag_part, dim=2, keepdim=True)  # N x D x 1
        sum_of_rows = torch.sum(inputs, dim=3)  # N x D x m
        sum_of_cols = torch.sum(inputs, dim=2)  # N x D x m
        sum_all = torch.sum(sum_of_rows, dim=2)  # N x D

        # op1 - (123) - extract diag
        op1 = diag_part  # N x D x m

        # op2 - (123) + (12)(3) - tile sum of diag part
        op2 = sum_diag_part.repeat(1, 1, dim)  # N x D x m

        # op3 - (123) + (13)(2) - place sum of row i in element i
        op3 = sum_of_rows  # N x D x m

        # op4 - (123) + (23)(1) - place sum of col i in element i
        op4 = sum_of_cols  # N x D x m

        # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
        op5 = torch.unsqueeze(sum_all, dim=2).repeat(1, 1, dim)  # N x D x m


        if normalization is not None:
            float_dim = dim.type(torch.float32)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)
                op3 = torch.div(op3, float_dim)
                op4 = torch.div(op4, float_dim)
                op5 = torch.div(op5, float_dim ** 2)

        return [op1, op2, op3, op4, op5]

   
    def ops_2_to_0(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
        diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
        sum_diag_part = torch.sum(diag_part, dim=2)  # N x D
        sum_of_rows = torch.sum(inputs, dim=3)  # N x D x m
        sum_all = torch.sum(sum_of_rows, dim=2)  # N x D

        # op1 -
        op1 = sum_diag_part  # N x D

        # op2 -
        op2 = sum_all  # N x D


        if normalization is not None:
            float_dim = dim.type(torch.float32)
            if normalization is 'inf':
                op1 = torch.div(op1, float_dim)
                op2 = torch.div(op2, float_dim ** 2)

        return [op1, op2]
   
    def ops_0_to_2(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D
       
        # op1 - (123) - place on diag
        op1 = torch.unsqueeze(inputs, dim=2).repeat(1, 1, dim)  # N x D x m
        op1 = torch.diag_embed(op1)  # N x D x m x m
#         op1 = F.pad(op1, (0,dim-inputs.shape[-1],0,dim-inputs.shape[-1]))

        # op2 - (123) + (12)(3) - tile in all entries
        op2 = torch.unsqueeze(torch.unsqueeze(inputs, dim=2), dim=3).repeat(1, 1, dim, dim)  # N x D x m x m

        if normalization is not None:
            float_dim = dim.type(torch.float32)
            if normalization is 'inf':
#                 op1 = torch.div(op1, float_dim)
                op2 = torch.div(op2, float_dim ** 2)

        return [op1, op2]


    def ops_1_to_2(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m
        sum_all = torch.sum(inputs, dim=2, keepdims=True)  # N x D x 1

        # op1 - (123) - place on diag
        op1 = torch.diag_embed(inputs)  # N x D x m x m

        # op2 - (123) + (12)(3) - tile sum on diag
        op2 = torch.diag_embed(sum_all.repeat(1, 1, dim))  # N x D x m x m

        # op3 - (123) + (13)(2) - tile element i in row i
        op3 = torch.unsqueeze(inputs, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

        # op4 - (123) + (23)(1) - tile element i in col i
        op4 = torch.unsqueeze(inputs, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

        # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
        op5 = torch.unsqueeze(sum_all, dim=3).repeat(1, 1, dim, dim)  # N x D x m x m

        if normalization is not None:
            float_dim = dim.type(torch.float32)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)
                op5 = torch.div(op5, float_dim)

        return [op1, op2, op3, op4, op5]


    def ops_1_to_1(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m
        sum_all = torch.sum(inputs, dim=2, keepdims=True)  # N x D x 1

        # op1 - (12) - identity
        op1 = inputs  # N x D x m

        # op2 - (1)(2) - tile sum of all
        op2 = sum_all.repeat(1, 1, dim)  # N x D x m

        if normalization is not None:
            float_dim = dim.type(torch.float32)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)

        return [op1, op2]
   
    def ops_1_to_0(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m
        sum_all = torch.sum(inputs, dim=2)  # N x D

        # op1 - (12) - identity
        op1 = sum_all  # N x D

        if normalization is not None:
            float_dim = dim.type(torch.float32)
            if normalization is 'inf':
                op1 = torch.div(op1, float_dim)

        return [op1]
   
    def ops_0_to_1(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D

        # op1
        op1 = torch.unsqueeze(inputs, dim=2).repeat(1, 1, dim)  # N x D x m
#         op1 = torch.diag_embed(inputs)  # N x D x m
#         op1 = F.pad(op1, (0,dim-inputs.shape[-1]))

#         if normalization is not None:
#             float_dim = dim.type(torch.float32)
#             if normalization is 'inf':
#                 op1 = torch.div(op1, float_dim)

        return [op1]
   
    def get_mask(self, A, i, rep, hops=1):
        indices = (A==1).nonzero(as_tuple=False)
#         print(f'indices : {indices.shape}')
        if hops==1:
            indices = indices[indices[:,1]==i]
        else:
            indices1hop = indices[indices[:,1]==i]
#             print(f'indices1hop : {indices1hop.shape}')
            indices2hop = []
#             print(f'ind shape : {indices1hop[0].shape}')
            for ind in indices1hop:
                inds = indices[(indices[:,0]==ind[0]) & (indices[:,1]==ind[2])]
                for val in inds:
                    indices2hop.append(val)
            if hops==2:
                indices = torch.stack(indices2hop, dim=0)
            else:
                indices2hop = torch.stack(indices2hop, dim=0)
               

        mask = torch.ones(A.shape, device=A.device)
        for index in indices:
            mask[index[0],index[1],index[2]] = 0
            mask[index[0],index[2],index[2]] = 0

        mask = torch.min(mask, torch.transpose(mask,-2,-1))
        mask = mask.bool()
        mask = torch.unsqueeze(mask, dim=1)
       
        if rep==1:
            mask = mask[:,:,i,:]
#         print(f'rep : {rep} - mask shape : {mask.shape}')
       
        degree = torch.unique(indices[:,0], return_counts=True)[1]
#         print(f'degree : {degree}')
       
        return mask, degree
       

    def forward(self, inputs, A, normalization='inf'):
       
        m = torch.tensor(inputs['2'].shape[3], dtype=torch.int32, device=A.device)  # extract dimension

        for i in range(inputs['2'].shape[-1]):
            if 2 in self.repsin and 2 in self.repsout:
                inputs_temp = torch.clone(inputs['2'])  # N x S x m x m
                if self.hops == -1:
                    degree = [-1 for _ in range(inputs_temp.shape[0])]
                else:
                    mask, degree = self.get_mask(A, i, rep=2, hops=self.hops)
                    inputs_temp.masked_fill_(mask, 0)
                ops_out = self.ops_2_to_2(inputs=inputs_temp, dim=m, normalization=normalization)
                ops_out = torch.stack(ops_out, dim=2)
                coeffs = [self.coeffs22[f'{deg}'] for deg in degree]
                coeffs = torch.stack(coeffs, dim=0)
                print(f'ops_out device : {ops_out.device}')
                print(f'coeffs device : {coeffs.device}')
                if i==0:
                    output22 = torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
                else:
                    output22 += torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
                   
               
            if 2 in self.repsin and 1 in self.repsout:
                inputs_temp = torch.clone(inputs['2'])  # N x S x m x m
                if self.hops == -1:
                    degree = [-1 for _ in range(inputs_temp.shape[0])]
                else:
                    mask, degree = self.get_mask(A, i, rep=2, hops=self.hops)
                    inputs_temp.masked_fill_(mask, 0)
                ops_out = self.ops_2_to_1(inputs=inputs_temp, dim=m, normalization=normalization)
                ops_out = torch.stack(ops_out, dim=2)
                coeffs = [self.coeffs21[f'{deg}'] for deg in degree]
                coeffs = torch.stack(coeffs, dim=0)
                if i==0:
                    output21 = torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
                else:
                    output21 += torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
                   
               
            if 2 in self.repsin and 0 in self.repsout:
                inputs_temp = torch.clone(inputs['2'])  # N x S x m x m
                if self.hops == -1:
                    degree = [-1 for _ in range(inputs_temp.shape[0])]
                else:
                    mask, degree = self.get_mask(A, i, rep=2, hops=self.hops)
                    inputs_temp.masked_fill_(mask, 0)
                ops_out = self.ops_2_to_0(inputs=inputs_temp, dim=m, normalization=normalization)
                ops_out = torch.stack(ops_out, dim=2)
                coeffs = [self.coeffs20[f'{deg}'] for deg in degree]
                coeffs = torch.stack(coeffs, dim=0)
                if i==0:
                    output20 = torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
                else:
                    output20 += torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
                   
               
            if 1 in self.repsin and 2 in self.repsout:
                inputs_temp = torch.clone(inputs['1'])  # N x S x m
                if self.hops == -1:
                    degree = [-1 for _ in range(inputs_temp.shape[0])]
                else:
                    mask, degree = self.get_mask(A, i, rep=1, hops=self.hops)
                    inputs_temp.masked_fill_(mask, 0)
                ops_out = self.ops_1_to_2(inputs=inputs_temp, dim=m, normalization=normalization)
                ops_out = torch.stack(ops_out, dim=2)
                coeffs = [self.coeffs12[f'{deg}'] for deg in degree]
                coeffs = torch.stack(coeffs, dim=0)
                if i==0:
                    output12 = torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
                else:
                    output12 += torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
                   

            if 1 in self.repsin and 1 in self.repsout:
                inputs_temp = torch.clone(inputs['1'])  # N x S x m
                if self.hops == -1:
                    degree = [-1 for _ in range(inputs_temp.shape[0])]
                else:
                    mask, degree = self.get_mask(A, i, rep=1, hops=self.hops)
                    inputs_temp.masked_fill_(mask, 0)
                ops_out = self.ops_1_to_1(inputs=inputs_temp, dim=m, normalization=normalization)
                ops_out = torch.stack(ops_out, dim=2)
                coeffs = [self.coeffs11[f'{deg}'] for deg in degree]
                coeffs = torch.stack(coeffs, dim=0)
                if i==0:
                    output11 = torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
                else:
                    output11 += torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
                   
                   
            if 1 in self.repsin and 0 in self.repsout:
                inputs_temp = torch.clone(inputs['1'])  # N x S x m
                if self.hops == -1:
                    degree = [-1 for _ in range(inputs_temp.shape[0])]
                else:
                    mask, degree = self.get_mask(A, i, rep=1, hops=self.hops)
                    inputs_temp.masked_fill_(mask, 0)
                ops_out = self.ops_1_to_0(inputs=inputs_temp, dim=m, normalization=normalization)
                ops_out = torch.stack(ops_out, dim=2)
                coeffs = [self.coeffs10[f'{deg}'] for deg in degree]
                coeffs = torch.stack(coeffs, dim=0)
                if i==0:
                    output10 = torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
                else:
                    output10 += torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
                   
       
            if 0 in self.repsin and 2 in self.repsout:
                inputs_temp = torch.clone(inputs['0'])  # N x S
                if self.hops == -1:
                    degree = [-1 for _ in range(inputs_temp.shape[0])]
                else:
                    mask, degree = self.get_mask(A, i, rep=0, hops=self.hops)
#                     inputs_temp.masked_fill_(mask, 0)
                ops_out = self.ops_0_to_2(inputs=inputs_temp, dim=m, normalization=normalization)
                ops_out = torch.stack(ops_out, dim=2)
                coeffs = [self.coeffs02[f'{deg}'] for deg in degree]
                coeffs = torch.stack(coeffs, dim=0)
                if i==0:
                    output02 = torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
                else:
                    output02 += torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
                   

            if 0 in self.repsin and 1 in self.repsout:
                inputs_temp = torch.clone(inputs['0'])  # N x S
                if self.hops == -1:
                    degree = [-1 for _ in range(inputs_temp.shape[0])]
                else:
                    mask, degree = self.get_mask(A, i, rep=0, hops=self.hops)
                ops_out = self.ops_0_to_1(inputs=inputs_temp, dim=m, normalization=normalization)
                ops_out = torch.stack(ops_out, dim=2)
                coeffs = [self.coeffs01[f'{deg}'] for deg in degree]
                coeffs = torch.stack(coeffs, dim=0)
                if i==0:
                    output01 = torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
                else:
                    output01 += torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
                   
                   
            if 0 in self.repsin and 0 in self.repsout:
                inputs_temp = torch.clone(inputs['0'])  # N x S
                if self.hops == -1:
                    degree = [-1 for _ in range(inputs_temp.shape[0])]
                else:
                    mask, degree = self.get_mask(A, i, rep=0, hops=self.hops)
                ops_out = torch.unsqueeze(inputs_temp, dim=2)
                coeffs = [self.coeffs00[f'{deg}'] for deg in degree]
                coeffs = torch.stack(coeffs, dim=0)
                if i==0:
                    output00 = torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
                else:
                    output00 += torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
                   
               
        if 2 in self.repsin and 2 in self.repsout:
            mat_diag_bias22 = self.diag_bias22.expand(-1,-1,inputs['2'].shape[3],inputs['2'].shape[3])
            mat_diag_bias22 = torch.mul(mat_diag_bias22, torch.eye(inputs['2'].shape[3], dtype=torch.float32, device=self.device))
            output22 = output22 + self.all_bias22 + mat_diag_bias22
        if 2 in self.repsin and 1 in self.repsout:
            output21 = output21 + self.bias21
        if 2 in self.repsin and 0 in self.repsout:
            output20 = output20 + self.bias20
        if 1 in self.repsin and 2 in self.repsout:
            output12 = output12 + self.bias12
        if 1 in self.repsin and 1 in self.repsout:
            output11 = output11 + self.bias11
        if 1 in self.repsin and 0 in self.repsout:
            output10 = output10 + self.bias10
        if 0 in self.repsin and 2 in self.repsout:
            output02 = output02 + self.bias02
        if 0 in self.repsin and 1 in self.repsout:
            output01 = output01 + self.bias01
        if 0 in self.repsin and 0 in self.repsout:
            output00 = output00 + self.bias00
       
        if 2 in self.repsout:
            if 2 in self.repsin and 1 in self.repsin and 0 in self.repsin:
                output2 = torch.cat((output22,output12,output02), dim=1)
            elif 2 in self.repsin and 1 in self.repsin:
                output2 = torch.cat((output22,output12), dim=1)
            elif 2 in self.repsin and 0 in self.repsin:
                output2 = torch.cat((output22,output02), dim=1)
            elif 1 in self.repsin and 0 in self.repsin:
                output2 = torch.cat((output12,output02), dim=1)
            elif 2 in self.repsin:
                output2 = output22
            elif 1 in self.repsin:
                output2 = output12
            elif 0 in self.repsin:
                output2 = output02
        if 1 in self.repsout:
            if 2 in self.repsin and 1 in self.repsin and 0 in self.repsin:
                output1 = torch.cat((output21,output11,output01), dim=1)
            elif 2 in self.repsin and 1 in self.repsin:
                output1 = torch.cat((output21,output11), dim=1)
            elif 2 in self.repsin and 0 in self.repsin:
                output1 = torch.cat((output21,output01), dim=1)
            elif 1 in self.repsin and 0 in self.repsin:
                output1 = torch.cat((output11,output01), dim=1)
            elif 2 in self.repsin:
                output1 = output21
            elif 1 in self.repsin:
                output1 = output11
            elif 0 in self.repsin:
                output1 = output01
        if 0 in self.repsout:
            if 2 in self.repsin and 1 in self.repsin and 0 in self.repsin:
                output0 = torch.cat((output20,output10,output00), dim=1)
            elif 2 in self.repsin and 1 in self.repsin:
                output0 = torch.cat((output20,output10), dim=1)
            elif 2 in self.repsin and 0 in self.repsin:
                output0 = torch.cat((output20,output00), dim=1)
            elif 1 in self.repsin and 0 in self.repsin:
                output0 = torch.cat((output10,output00), dim=1)
            elif 2 in self.repsin:
                output0 = output20
            elif 1 in self.repsin:
                output0 = output10
            elif 0 in self.repsin:
                output0 = output00
               
        if 2 in self.repsout and 1 in self.repsout and 0 in self.repsout:
            return {'2':output2, '1':output1, '0':output0}
        elif 2 in self.repsout and 1 in self.repsout:
            return {'2':output2, '1':output1}
        elif 2 in self.repsout and 0 in self.repsout:
            return {'2':output2, '0':output0}
        elif 1 in self.repsout and 0 in self.repsout:
            return {'1':output1, '0':output0}
        elif 2 in self.repsout:
            return {'2':output2}
        elif 1 in self.repsout:
            return {'1':output1}
        elif 0 in self.repsout:
            return {'0':output0}
        else:
            pass
        
        
class equi_multi_local_fixed(torch.nn.Module):
    """equivariant nn layer."""

    def __init__(self, input_depth, output_depth, device, subgroups, repsin, repsout, hops):
        super(equi_multi_local_fixed, self).__init__()
        self.repsin = repsin
        self.repsout = repsout
        self.hops = hops
        self.device = device
        self.basis_dimension_22 = 15
        self.basis_dimension_21 = 5
        self.basis_dimension_12 = 5
        self.basis_dimension_20 = 2
        self.basis_dimension_02 = 2
        self.basis_dimension_11 = 2
        self.basis_dimension_10 = 1
        self.basis_dimension_01 = 1
        self.basis_dimension_00 = 1
       
        if 2 in repsin and 2 in repsout:
            coeffs_values_22 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_22), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs22 = torch.nn.ParameterDict()
#             for sub in subgroups:  ## This isn't working on multiple gpus due to a pytorch bug, have to hard code subgroups for now https://github.com/pytorch/pytorch/issues/36035
#                 self.coeffs22[f'{sub}'] = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_1 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_0 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_2 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_3 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_4 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_5 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.diag_bias22 = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
            self.all_bias22 = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
       
        if 2 in repsin and 1 in repsout:
            coeffs_values_21 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_21), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs21 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs21[f'{sub}'] = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_1 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_0 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_2 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_3 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_4 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_5 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.bias21 = torch.nn.Parameter(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
           
        if 1 in repsin and 2 in repsout:
            coeffs_values_12 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_12), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs12 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs12[f'{sub}'] = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_1 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_0 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_2 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_3 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_4 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_5 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.bias12 = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
           
        if 2 in repsin and 0 in repsout:
            coeffs_values_20 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_20), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs20 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs20[f'{sub}'] = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_1 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_0 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_2 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_3 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_4 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_5 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.bias20 = torch.nn.Parameter(torch.zeros((1, output_depth), dtype=torch.float32), requires_grad=True)
           
        if 0 in repsin and 2 in repsout:
            coeffs_values_02 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_02), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs02 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs02[f'{sub}'] = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_1 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_0 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_2 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_3 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_4 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_5 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.bias02 = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
           
        if 1 in repsin and 1 in repsout:
            coeffs_values_11 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_11), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs11 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs11[f'{sub}'] = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_1 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_0 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_2 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_3 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_4 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_5 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.bias11 = torch.nn.Parameter(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
           
        if 1 in repsin and 0 in repsout:
            coeffs_values_10 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_10), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs10 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs10[f'{sub}'] = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_1 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_0 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_2 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_3 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_4 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_5 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.bias10 = torch.nn.Parameter(torch.zeros((1, output_depth), dtype=torch.float32), requires_grad=True)
           
        if 0 in repsin and 1 in repsout:
            coeffs_values_01 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_01), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs01 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs01[f'{sub}'] = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_1 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_0 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_2 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_3 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_4 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_5 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.bias01 = torch.nn.Parameter(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
           
        if 0 in repsin and 0 in repsout:
            coeffs_values_00 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_00), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs00 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs00[f'{sub}'] = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_1 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_0 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_2 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_3 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_4 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_5 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.bias00 = torch.nn.Parameter(torch.zeros((1, output_depth), dtype=torch.float32), requires_grad=True)
               

    def ops_2_to_2(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D x m x m
        diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
        sum_diag_part = torch.sum(diag_part, dim=3, keepdim=True)  # N x D x 1
        sum_of_rows = torch.sum(inputs, dim=4)  # N x D x m
        sum_of_cols = torch.sum(inputs, dim=3)  # N x D x m
        sum_all = torch.sum(sum_of_rows, dim=3)  # N x D

        # op1 - (1234) - extract diag
        op1 = torch.diag_embed(diag_part)  # N x D x m x m
#         print(f'inputs shape : {inputs.shape}')
#         print(f'diag_part shape : {diag_part.shape}')
#         print(f'dim : {dim}')
#         print(f'op1 max : {torch.max(op1)}')

        # op2 - (1234) + (12)(34) - place sum of diag on diag
        op2 = torch.diag_embed(sum_diag_part.repeat(1, 1, 1, dim)) - op1
#         print(f'op2 max : {torch.max(op2)}')

        # op3 - (1234) + (123)(4) - place sum of row i on diag ii
        op3 = torch.diag_embed(sum_of_rows) - op1
#         print(f'op3 max : {torch.max(op3)}')

        # op4 - (1234) + (124)(3) - place sum of col i on diag ii
        op4 = torch.diag_embed(sum_of_cols) - op1
#         print(f'op4 max : {torch.max(op4)}')
       
        # op10 - (1234) + (14)(23) - identity
        op10 = inputs  - op1
#         print(f'op10max : {torch.max(op10)}')

        # op11 - (1234) + (13)(24) - transpose
        op11 = inputs.permute(0, 1, 2, 4, 3) - op1
#         print(f'op11 max : {torch.max(op11)}')

        # op12 - (1234) + (234)(1) - place ii element in row i
        op12 = torch.unsqueeze(diag_part, dim=4).repeat(1, 1, 1, 1, dim) - op1
#         print(f'op12 max : {torch.max(op12)}')

        # op13 - (1234) + (134)(2) - place ii element in col i
        op13 = torch.unsqueeze(diag_part, dim=3).repeat(1, 1, 1, dim, 1) - op1
#         print(f'op13 max : {torch.max(op13)}')

        # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
        op5 = torch.diag_embed(torch.unsqueeze(sum_all, dim=3).repeat(1, 1, 1, dim)) - op1 - op4 - op3 - op2
#         print(f'op5 max : {torch.max(op5)}')

        # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
        op6 = torch.unsqueeze(sum_of_cols, dim=4).repeat(1, 1, 1, 1, dim) - op1 - op10 - op11 - op4
#         print(f'op6 max : {torch.max(op6)}')

        # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
        op7 = torch.unsqueeze(sum_of_rows, dim=4).repeat(1, 1, 1, 1, dim) - op1 - op10 - op12 - op3
#         print(f'op7 max : {torch.max(op7)}')

        # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
        op8 = torch.unsqueeze(sum_of_cols, dim=3).repeat(1, 1, 1, dim, 1) - op1 - op13 - op10 - op4
#         print(f'op8 max : {torch.max(op8)}')

        # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
        op9 = torch.unsqueeze(sum_of_rows, dim=3).repeat(1, 1, 1, dim, 1) - op1 - op11 - op13 - op3
#         print(f'op9 max : {torch.max(op9)}')

        # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
        op14 = torch.unsqueeze(sum_diag_part, dim=4).repeat(1, 1, 1, dim, dim) - op1 - op12 - op13 - op2
#         print(f'op14 max : {torch.max(op14)}')

        # op15 - sum of all ops - place sum of all entries in all entries
        op15 = torch.unsqueeze(torch.unsqueeze(sum_all, dim=3), dim=4).repeat(1, 1, 1, dim, dim) - op1 - op2 - op3 - op4 - op5 - op6 - op7 - op8 - op9 - op10 - op11 - op12 - op13 - op14
#         print(f'op15 max : {torch.max(op15)}')

        if dim > 2:
            if normalization is not None:
                float_dim = degrees.type(torch.FloatTensor)
#                 print(float_dim)
                extra_norm = torch.max(sum_all)
                if normalization is 'inf':
                    op2 = torch.div(op2, float_dim-1)
                    op3 = torch.div(op3, float_dim-1)
                    op4 = torch.div(op4, float_dim-1)
                    op5 = torch.div(op5, (float_dim-1)*(float_dim-2))
                    op6 = torch.div(op6, float_dim-2)
                    op7 = torch.div(op7, float_dim-2)
                    op8 = torch.div(op8, float_dim-2)
                    op9 = torch.div(op9, float_dim-2)
    #                 op10 = torch.div(op10, float_dim-2)
    #                 op11 = torch.div(op11, float_dim)
    #                 op12 = torch.div(op12, float_dim)
    #                 op13 = torch.div(op13, float_dim)
                    op14 = torch.div(op14, float_dim-2)
                    op15 = torch.div(op15, (float_dim-2)*(float_dim-3))
               
        return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]


#     def ops_2_to_2(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x k x D x m x m
#         diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x k x D x m
#         sum_diag_part = torch.sum(diag_part, dim=3, keepdim=True)  # N x k x D x 1
#         sum_of_rows = torch.sum(inputs, dim=4)  # N x k x D x m
#         sum_of_cols = torch.sum(inputs, dim=3)  # N x k x D x m
#         sum_all = torch.sum(sum_of_rows, dim=3)  # N x k x D

#         # op1 - (1234) - extract diag
#         op1 = torch.diag_embed(diag_part)  # N x k x D x m x m
# #         print(f'op1 max : {torch.max(op1)}')

#         # op2 - (1234) + (12)(34) - place sum of diag on diag
#         op2 = torch.diag_embed(sum_diag_part.repeat(1, 1, 1, dim))  # N x k x D x m x m
# #         print(f'op2 max : {torch.max(op2)}')

#         # op3 - (1234) + (123)(4) - place sum of row i on diag ii
#         op3 = torch.diag_embed(sum_of_rows)  # N x k x D x m x m
# #         print(f'op3 max : {torch.max(op3)}')

#         # op4 - (1234) + (124)(3) - place sum of col i on diag ii
#         op4 = torch.diag_embed(sum_of_cols)  # N x k x D x m x m
# #         print(f'op4 max : {torch.max(op4)}')

#         # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
#         op5 = torch.diag_embed(torch.unsqueeze(sum_all, dim=3).repeat(1, 1, 1, dim))  # N x k x D x m x m
# #         print(f'op5 max : {torch.max(op5)}')

#         # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
#         op6 = torch.unsqueeze(sum_of_cols, dim=4).repeat(1, 1, 1, 1, dim)  # N x k x D x m x m
# #         print(f'op6 max : {torch.max(op6)}')

#         # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
#         op7 = torch.unsqueeze(sum_of_rows, dim=4).repeat(1, 1, 1, 1, dim)  # N x k x D x m x m
# #         print(f'op7 max : {torch.max(op7)}')

#         # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
#         op8 = torch.unsqueeze(sum_of_cols, dim=3).repeat(1, 1, 1, dim, 1)  # N x k x D x m x m
# #         print(f'op8 max : {torch.max(op8)}')

#         # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
#         op9 = torch.unsqueeze(sum_of_rows, dim=3).repeat(1, 1, 1, dim, 1)  # N x k x D x m x m
# #         print(f'op9 max : {torch.max(op9)}')

#         # op10 - (1234) + (14)(23) - identity
#         op10 = inputs  # N x k x D x m x m
# #         print(f'op10 max : {torch.max(op10)}')

#         # op11 - (1234) + (13)(24) - transpose
#         op11 = inputs.permute(0, 1, 2, 4, 3)  # N x k x D x m x m
# #         print(f'op11 max : {torch.max(op11)}')

#         # op12 - (1234) + (234)(1) - place ii element in row i
#         op12 = torch.unsqueeze(diag_part, dim=4).repeat(1, 1, 1, 1, dim)  # N x k x D x m x m
# #         print(f'op12 max : {torch.max(op12)}')

#         # op13 - (1234) + (134)(2) - place ii element in col i
#         op13 = torch.unsqueeze(diag_part, dim=3).repeat(1, 1, 1, dim, 1)  # N x k x D x m x m
# #         print(f'op13 max : {torch.max(op13)}')

#         # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
#         op14 = torch.unsqueeze(sum_diag_part, dim=4).repeat(1, 1, 1, dim, dim)   # N x k x D x m x m
# #         print(f'op14 max : {torch.max(op14)}')

#         # op15 - sum of all ops - place sum of all entries in all entries
#         op15 = torch.unsqueeze(torch.unsqueeze(sum_all, dim=3), dim=4).repeat(1, 1, 1, dim, dim)  # N x k x D x m x m
# #         print(f'op15 max : {torch.max(op15)}')

#         if normalization is not None:
#             float_dim = degrees.type(torch.float32)
#             if normalization is 'inf':
#                 op2 = torch.div(op2, float_dim)
#                 op3 = torch.div(op3, float_dim)
#                 op4 = torch.div(op4, float_dim)
#                 op5 = torch.div(op5, float_dim**2)
#                 op6 = torch.div(op6, float_dim)
#                 op7 = torch.div(op7, float_dim)
#                 op8 = torch.div(op8, float_dim)
#                 op9 = torch.div(op9, float_dim)
#                 op14 = torch.div(op14, float_dim)
#                 op15 = torch.div(op15, float_dim**2)

#         return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]

#     def ops_2_to_1(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
#         diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
#         sum_diag_part = torch.sum(diag_part, dim=2, keepdim=True)  # N x D x 1
#         sum_of_rows = torch.sum(inputs, dim=3)  # N x D x m
#         sum_of_cols = torch.sum(inputs, dim=2)  # N x D x m
#         sum_all = torch.sum(sum_of_rows, dim=2)  # N x D

#         # op1 - (123) - extract diag
#         op1 = diag_part  # N x D x m

#         # op2 - (123) + (12)(3) - tile sum of diag part
#         op2 = sum_diag_part.repeat(1, 1, dim)  # N x D x m

#         # op3 - (123) + (13)(2) - place sum of row i in element i
#         op3 = sum_of_rows  # N x D x m

#         # op4 - (123) + (23)(1) - place sum of col i in element i
#         op4 = sum_of_cols  # N x D x m

#         # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
#         op5 = torch.unsqueeze(sum_all, dim=2).repeat(1, 1, dim)  # N x D x m


#         if normalization is not None:
#             float_dim = dim.type(torch.float32)
#             if normalization is 'inf':
#                 op2 = torch.div(op2, float_dim)
#                 op3 = torch.div(op3, float_dim)
#                 op4 = torch.div(op4, float_dim)
#                 op5 = torch.div(op5, float_dim ** 2)

#         return [op1, op2, op3, op4, op5]

   
    def ops_2_to_1(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D x m x m
        diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
        sum_diag_part = torch.sum(diag_part, dim=3, keepdim=True)  # N x D x 1
        sum_of_rows = torch.sum(inputs, dim=4)  # N x D x m
        sum_of_cols = torch.sum(inputs, dim=3)  # N x D x m
        sum_all = torch.sum(sum_of_rows, dim=3)  # N x D

        # op1 - (123) - extract diag
        op1 = diag_part  # N x D x m

        # op2 - (123) + (12)(3) - tile sum of diag part
        op2 = sum_diag_part.repeat(1, 1, 1, dim)  # N x D x m

        # op3 - (123) + (13)(2) - place sum of row i in element i
        op3 = sum_of_rows  # N x D x m

        # op4 - (123) + (23)(1) - place sum of col i in element i
        op4 = sum_of_cols  # N x D x m

        # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
        op5 = torch.unsqueeze(sum_all, dim=3).repeat(1, 1, 1, dim)  # N x D x m


        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)
                op3 = torch.div(op3, float_dim)
                op4 = torch.div(op4, float_dim)
                op5 = torch.div(op5, float_dim ** 2)

        return [op1, op2, op3, op4, op5]


#     def ops_2_to_0(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
#         diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
#         sum_diag_part = torch.sum(diag_part, dim=2)  # N x D
#         sum_of_rows = torch.sum(inputs, dim=3)  # N x D x m
#         sum_all = torch.sum(sum_of_rows, dim=2)  # N x D

#         # op1 -
#         op1 = sum_diag_part  # N x D

#         # op2 -
#         op2 = sum_all  # N x D


#         if normalization is not None:
#             float_dim = dim.type(torch.float32)
#             if normalization is 'inf':
#                 op1 = torch.div(op1, float_dim)
#                 op2 = torch.div(op2, float_dim ** 2)

#         return [op1, op2]
   
    def ops_2_to_0(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D x m x m
        diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
        sum_diag_part = torch.sum(diag_part, dim=3)  # N x D
        sum_of_rows = torch.sum(inputs, dim=4)  # N x D x m
        sum_all = torch.sum(sum_of_rows, dim=3)  # N x D

        # op1 -
        op1 = sum_diag_part  # N x D

        # op2 -
        op2 = sum_all  # N x D


        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
                op1 = torch.div(op1, float_dim)
                op2 = torch.div(op2, float_dim ** 2)

        return [op1, op2]

#     def ops_0_to_2(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D
       
#         # op1 - (123) - place on diag
#         op1 = torch.unsqueeze(inputs, dim=2).repeat(1, 1, dim)  # N x D x m
#         op1 = torch.diag_embed(op1)  # N x D x m x m
# #         op1 = F.pad(op1, (0,dim-inputs.shape[-1],0,dim-inputs.shape[-1]))

#         # op2 - (123) + (12)(3) - tile in all entries
#         op2 = torch.unsqueeze(torch.unsqueeze(inputs, dim=2), dim=3).repeat(1, 1, dim, dim)  # N x D x m x m

#         if normalization is not None:
#             float_dim = dim.type(torch.float32)
#             if normalization is 'inf':
# #                 op1 = torch.div(op1, float_dim)
#                 op2 = torch.div(op2, float_dim ** 2)

#         return [op1, op2]


    def ops_0_to_2(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D
       
        # op1 - (123) - place on diag
        op1 = torch.unsqueeze(inputs, dim=3).repeat(1, 1, 1, dim)  # N x D x m
        op1 = torch.diag_embed(op1)  # N x D x m x m
#         op1 = F.pad(op1, (0,dim-inputs.shape[-1],0,dim-inputs.shape[-1]))

        # op2 - (123) + (12)(3) - tile in all entries
        op2 = torch.unsqueeze(torch.unsqueeze(inputs, dim=3), dim=4).repeat(1, 1, 1, dim, dim)  # N x D x m x m

        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
#                 op1 = torch.div(op1, float_dim)
                op2 = torch.div(op2, float_dim ** 2)

        return [op1, op2]


#     def ops_1_to_2(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m
#         sum_all = torch.sum(inputs, dim=2, keepdims=True)  # N x D x 1

#         # op1 - (123) - place on diag
#         op1 = torch.diag_embed(inputs)  # N x D x m x m

#         # op2 - (123) + (12)(3) - tile sum on diag
#         op2 = torch.diag_embed(sum_all.repeat(1, 1, dim))  # N x D x m x m

#         # op3 - (123) + (13)(2) - tile element i in row i
#         op3 = torch.unsqueeze(inputs, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

#         # op4 - (123) + (23)(1) - tile element i in col i
#         op4 = torch.unsqueeze(inputs, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

#         # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
#         op5 = torch.unsqueeze(sum_all, dim=3).repeat(1, 1, dim, dim)  # N x D x m x m

#         if normalization is not None:
#             float_dim = dim.type(torch.float32)
#             if normalization is 'inf':
#                 op2 = torch.div(op2, float_dim)
#                 op5 = torch.div(op5, float_dim)

#         return [op1, op2, op3, op4, op5]


    def ops_1_to_2(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D x m
        sum_all = torch.sum(inputs, dim=3, keepdims=True)  # N x D x 1

        # op1 - (123) - place on diag
        op1 = torch.diag_embed(inputs)  # N x D x m x m

        # op2 - (123) + (12)(3) - tile sum on diag
        op2 = torch.diag_embed(sum_all.repeat(1, 1, 1, dim))  # N x D x m x m

        # op3 - (123) + (13)(2) - tile element i in row i
        op3 = torch.unsqueeze(inputs, dim=3).repeat(1, 1, 1, dim, 1)  # N x D x m x m

        # op4 - (123) + (23)(1) - tile element i in col i
        op4 = torch.unsqueeze(inputs, dim=4).repeat(1, 1, 1, 1, dim)  # N x D x m x m

        # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
        op5 = torch.unsqueeze(sum_all, dim=4).repeat(1, 1, 1, dim, dim)  # N x D x m x m

        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)
                op5 = torch.div(op5, float_dim)

        return [op1, op2, op3, op4, op5]


#     def ops_1_to_1(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m
#         sum_all = torch.sum(inputs, dim=2, keepdims=True)  # N x D x 1

#         # op1 - (12) - identity
#         op1 = inputs  # N x D x m

#         # op2 - (1)(2) - tile sum of all
#         op2 = sum_all.repeat(1, 1, dim)  # N x D x m

#         if normalization is not None:
#             float_dim = dim.type(torch.float32)
#             if normalization is 'inf':
#                 op2 = torch.div(op2, float_dim)

#         return [op1, op2]
   
    def ops_1_to_1(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D x m
        sum_all = torch.sum(inputs, dim=3, keepdims=True)  # N x D x 1

        # op1 - (12) - identity
        op1 = inputs  # N x D x m

        # op2 - (1)(2) - tile sum of all
        op2 = sum_all.repeat(1, 1, 1, dim)  # N x D x m

        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)

        return [op1, op2]

#     def ops_1_to_0(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m
#         sum_all = torch.sum(inputs, dim=2)  # N x D

#         # op1 - (12) - identity
#         op1 = sum_all  # N x D

#         if normalization is not None:
#             float_dim = dim.type(torch.float32)
#             if normalization is 'inf':
#                 op1 = torch.div(op1, float_dim)

#         return [op1]
   
    def ops_1_to_0(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D x m
        sum_all = torch.sum(inputs, dim=3)  # N x D

        # op1 - (12) - identity
        op1 = sum_all  # N x D

        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
                op1 = torch.div(op1, float_dim)

        return [op1]

#     def ops_0_to_1(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D

#         # op1
#         op1 = torch.unsqueeze(inputs, dim=2).repeat(1, 1, dim)  # N x D x m
# #         op1 = torch.diag_embed(inputs)  # N x D x m
# #         op1 = F.pad(op1, (0,dim-inputs.shape[-1]))

# #         if normalization is not None:
# #             float_dim = dim.type(torch.float32)
# #             if normalization is 'inf':
# #                 op1 = torch.div(op1, float_dim)

#         return [op1]

    def ops_0_to_1(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D

        # op1
        op1 = torch.unsqueeze(inputs, dim=3).repeat(1, 1, 1, dim)  # N x D x m
#         op1 = torch.diag_embed(inputs)  # N x D x m
#         op1 = F.pad(op1, (0,dim-inputs.shape[-1]))

#         if normalization is not None:
#             float_dim = degrees.type(torch.float32)
#             if normalization is 'inf':
#                 op1 = torch.div(op1, float_dim)

        return [op1]
   
    def get_mask(self, A, i, hops=1):
        indices = (A==1).nonzero(as_tuple=False)
#         print(f'indices : {indices.shape}')
        if hops==1:
            indices = indices[indices[:,1]==i]
        else:
            indices1hop = indices[indices[:,1]==i]
#             print(f'indices1hop : {indices1hop.shape}')
            indices2hop = []
#             print(f'ind shape : {indices1hop[0].shape}')
            for ind in indices1hop:
                inds = indices[(indices[:,0]==ind[0]) & (indices[:,1]==ind[2])]
                for val in inds:
                    indices2hop.append(val)
            if hops==2:
                indices = torch.stack(indices2hop, dim=0)
            else:
                indices2hop = torch.stack(indices2hop, dim=0)
               
        mask = torch.ones(A.shape, device=A.device)
#         print(indices)
        batches = torch.unique(indices[:,0], return_counts=True)[0]
#         print(batches)
        for batch in batches:
            indices_i = indices[indices[:,0]==batch]
#             print(indices_i)
            for i in indices_i[:,2]:
                for j in indices_i[:,2]:
#                     print(f'batch : {batch}, index1 : {i}, index : {j}')
                    mask[batch,i,j] = 0
#         for index in indices:
#             mask[index[0],index[1],index[2]] = 0
#             mask[index[0],index[2],index[2]] = 0

        mask = torch.min(mask, torch.transpose(mask,-2,-1))
        mask = mask.bool()
        mask = torch.unsqueeze(mask, dim=1)
       
        mask1 = mask[:,:,i,:]
#         print(f'rep : {rep} - mask shape : {mask.shape}')
       
        degree = torch.unique(indices[:,0], return_counts=True)[1]
#         print(f'degree : {degree}')
#         print(f'mask device : {mask.device}')
#         print(f'degree device : {degree.device}')
       
        return mask, mask1, degree
   
    def get_coeffs22(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs22_1)
            elif deg == 2:
                coeffs.append(self.coeffs22_2)
            elif deg == 3:
                coeffs.append(self.coeffs22_3)
            elif deg == 4:
                coeffs.append(self.coeffs22_4)
            elif deg == 5:
                coeffs.append(self.coeffs22_5)
            else:
                coeffs.append(self.coeffs22_0)
        return coeffs
           
    def get_coeffs21(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs21_1)
            elif deg == 2:
                coeffs.append(self.coeffs21_2)
            elif deg == 3:
                coeffs.append(self.coeffs21_3)
            elif deg == 4:
                coeffs.append(self.coeffs21_4)
            elif deg == 5:
                coeffs.append(self.coeffs21_5)
            else:
                coeffs.append(self.coeffs21_0)
        return coeffs
   
    def get_coeffs12(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs12_1)
            elif deg == 2:
                coeffs.append(self.coeffs12_2)
            elif deg == 3:
                coeffs.append(self.coeffs12_3)
            elif deg == 4:
                coeffs.append(self.coeffs12_4)
            elif deg == 5:
                coeffs.append(self.coeffs12_5)
            else:
                coeffs.append(self.coeffs12_0)
        return coeffs
   
    def get_coeffs11(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs11_1)
            elif deg == 2:
                coeffs.append(self.coeffs11_2)
            elif deg == 3:
                coeffs.append(self.coeffs11_3)
            elif deg == 4:
                coeffs.append(self.coeffs11_4)
            elif deg == 5:
                coeffs.append(self.coeffs11_5)
            else:
                coeffs.append(self.coeffs11_0)
        return coeffs
   
    def get_coeffs10(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs10_1)
            elif deg == 2:
                coeffs.append(self.coeffs10_2)
            elif deg == 3:
                coeffs.append(self.coeffs10_3)
            elif deg == 4:
                coeffs.append(self.coeffs10_4)
            elif deg == 5:
                coeffs.append(self.coeffs10_5)
            else:
                coeffs.append(self.coeffs10_0)
        return coeffs
   
    def get_coeffs01(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs01_1)
            elif deg == 2:
                coeffs.append(self.coeffs01_2)
            elif deg == 3:
                coeffs.append(self.coeffs01_3)
            elif deg == 4:
                coeffs.append(self.coeffs01_4)
            elif deg == 5:
                coeffs.append(self.coeffs01_5)
            else:
                coeffs.append(self.coeffs01_0)
        return coeffs
   
    def get_coeffs20(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs20_1)
            elif deg == 2:
                coeffs.append(self.coeffs20_2)
            elif deg == 3:
                coeffs.append(self.coeffs20_3)
            elif deg == 4:
                coeffs.append(self.coeffs20_4)
            elif deg == 5:
                coeffs.append(self.coeffs20_5)
            else:
                coeffs.append(self.coeffs20_0)
        return coeffs
   
    def get_coeffs02(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs02_1)
            elif deg == 2:
                coeffs.append(self.coeffs02_2)
            elif deg == 3:
                coeffs.append(self.coeffs02_3)
            elif deg == 4:
                coeffs.append(self.coeffs02_4)
            elif deg == 5:
                coeffs.append(self.coeffs02_5)
            else:
                coeffs.append(self.coeffs02_0)
        return coeffs
   
    def get_coeffs00(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs00_1)
            elif deg == 2:
                coeffs.append(self.coeffs00_2)
            elif deg == 3:
                coeffs.append(self.coeffs00_3)
            elif deg == 4:
                coeffs.append(self.coeffs00_4)
            elif deg == 5:
                coeffs.append(self.coeffs00_5)
            else:
                coeffs.append(self.coeffs00_0)
        return coeffs
   
   

    def forward(self, inputs, A, normalization='inf'):
#         print(f'A device : {A.device}')
#         print(f'coeffs22 : {self.coeffs22}')
#         print(self.coeffs22)

        m = torch.tensor(inputs['2'].shape[3], dtype=torch.int32, device=A.device)  # extract dimension
#         print(f'm device : {m.device}')

        if 2 in self.repsin and 2 in self.repsout:
            inputs_temp = inputs['2']  # N x S x m x m
            coeffs = []
            masks = []
            degrees = []
            for i in range(inputs_temp.shape[-1]):
                if self.hops == -1:
                    degree = [m for _ in range(inputs_temp.shape[0])]
                else:
                    mask, _, degree = self.get_mask(A, i, hops=self.hops)
#                     print(i)
#                     print(mask)
                    masks.append(mask)
                    degrees.append(degree)
                coeffs_temp = self.get_coeffs22(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            degrees = torch.stack(degrees, dim=1)
            degrees = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(degrees,dim=-1),dim=-1),dim=-1)
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, inputs_temp.shape[-1], 1, 1, 1)
            if self.hops != -1:
                masks = torch.stack(masks, dim=1)
                inputs_temp.masked_fill_(masks, 0)
            ops_out = self.ops_2_to_2(inputs=inputs_temp, dim=m, degrees=m, normalization=normalization)
            ops_out = torch.stack(ops_out, dim=3)
            if self.hops != -1:
                masks = torch.unsqueeze(masks, dim=3)
                ops_out.masked_fill_(masks, 0)

            output22 = torch.einsum('nkdsb,nkdbij->nsij', coeffs, ops_out)  # N x S x m x m
#             print(output22)
            degrees = torch.squeeze(torch.squeeze(torch.squeeze(degrees, dim=-1), dim=-1), dim=-1)
            degrees = torch.diag_embed(degrees)
            degrees = torch.unsqueeze(degrees, dim=1)
            degrees[degrees==0] = 2
#             print(degrees)
#             degrees = torch.unsqueeze(degrees, dim=-1).repeat(1, 1, 1, degrees.shape[-1])
#             print(degrees.shape)
            output22 = torch.div(output22,degrees.float())
#             print(output22)

        if 2 in self.repsin and 1 in self.repsout:
            inputs_temp = inputs['2']  # N x S x m x m
            coeffs = []
            masksin = []
            masksout = []
            degrees = []
            for i in range(inputs_temp.shape[-1]):
                if self.hops == -1:
                    degree = [m for _ in range(inputs_temp.shape[0])]
                else:
                    mask2, mask1, degree = self.get_mask(A, i, hops=self.hops)
                    masksin.append(mask2)
                    masksout.append(mask1)
                    degrees.append(degree)
                coeffs_temp = self.get_coeffs21(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            degrees = torch.stack(degrees, dim=1)
            degrees = torch.unsqueeze(torch.unsqueeze(degrees,dim=-1),dim=-1)
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, inputs_temp.shape[-1], 1, 1, 1)
            if self.hops != -1:
                masksin = torch.stack(masksin, dim=1)
                inputs_temp.masked_fill_(masksin, 0)
            ops_out = self.ops_2_to_1(inputs=inputs_temp, dim=m, degrees=m, normalization=normalization)
            ops_out = torch.stack(ops_out, dim=3)
            if self.hops != -1:
                masksout = torch.stack(masksout, dim=1)
                masksout = torch.unsqueeze(masksout, dim=3)
                ops_out.masked_fill_(masksout, 0)
                
            output21 = torch.einsum('nkdsb,nkdbi->nsi', coeffs, ops_out)  # N x S x m 
            degrees = torch.squeeze(torch.squeeze(degrees, dim=-1), dim=-1)
            degrees = torch.unsqueeze(degrees, dim=1)
            degrees[degrees==0] = 2
#             print(output21.shape)
#             print(degrees.shape)
            output21 = torch.div(output21,degrees.float())
            
        if 2 in self.repsin and 0 in self.repsout:
            inputs_temp = inputs['2']  # N x S x m x m
            coeffs = []
            masks = []
            degrees = []
            for i in range(inputs_temp.shape[-1]):
                if self.hops == -1:
                    degree = [m for _ in range(inputs_temp.shape[0])]
                else:
                    mask, _, degree = self.get_mask(A, i, hops=self.hops)
                    masks.append(mask)
                    degrees.append(degree)
                coeffs_temp = self.get_coeffs20(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            degrees = torch.stack(degrees, dim=1)
            degrees = torch.unsqueeze(degrees,dim=-1)
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, inputs_temp.shape[-1], 1, 1, 1)
            if self.hops != -1:
                masks = torch.stack(masks, dim=1)
                inputs_temp.masked_fill_(masks, 0)
            ops_out = self.ops_2_to_0(inputs=inputs_temp, dim=m, degrees=m, normalization=normalization)
            ops_out = torch.stack(ops_out, dim=3)

            output20 = torch.einsum('nkdsb,nkdb->ns', coeffs, ops_out)  # N x S
            
        if 1 in self.repsin and 2 in self.repsout:
            inputs_temp = inputs['1']  # N x S x m
            coeffs = []
            masksin = []
            masksout = []
            degrees = []
            for i in range(inputs_temp.shape[-1]):
                if self.hops == -1:
                    degree = [m for _ in range(inputs_temp.shape[0])]
                else:
                    mask2, mask1, degree = self.get_mask(A, i, hops=self.hops)
                    masksin.append(mask1)
                    masksout.append(mask2)
                    degrees.append(degree)
                coeffs_temp = self.get_coeffs12(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            degrees = torch.stack(degrees, dim=1)
            degrees = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(degrees,dim=-1),dim=-1),dim=-1)
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, inputs_temp.shape[-1], 1, 1)
            if self.hops != -1:
                masksin = torch.stack(masksin, dim=1)
                inputs_temp.masked_fill_(masksin, 0)
            ops_out = self.ops_1_to_2(inputs=inputs_temp, dim=m, degrees=m, normalization=normalization)
            ops_out = torch.stack(ops_out, dim=3)
            if self.hops != -1:
                masksout = torch.stack(masksout, dim=1)
                masksout = torch.unsqueeze(masksout, dim=3)
                ops_out.masked_fill_(masksout, 0)

            output12 = torch.einsum('nkdsb,nkdbij->nsij', coeffs, ops_out)  # N x S x m x m
#             print(output12.shape)
            degrees = torch.squeeze(torch.squeeze(torch.squeeze(degrees, dim=-1), dim=-1), dim=-1)
            degrees = torch.diag_embed(degrees)
            degrees = torch.unsqueeze(degrees, dim=1)
            degrees[degrees==0] = 2
#             print(degrees.shape)
            output12 = torch.div(output12,degrees.float())

        if 1 in self.repsin and 1 in self.repsout:
            inputs_temp = inputs['1']  # N x S x m
            coeffs = []
            masks = []
            degrees = []
            for i in range(inputs_temp.shape[-1]):
                if self.hops == -1:
                    degree = [m for _ in range(inputs_temp.shape[0])]
                else:
                    _, mask, degree = self.get_mask(A, i, hops=self.hops)
                    masks.append(mask)
                    degrees.append(degree)
                coeffs_temp = self.get_coeffs11(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            degrees = torch.stack(degrees, dim=1)
            degrees = torch.unsqueeze(torch.unsqueeze(degrees,dim=-1),dim=-1)
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, inputs_temp.shape[-1], 1, 1)
            if self.hops != -1:
                masks = torch.stack(masks, dim=1)
                inputs_temp.masked_fill_(masks, 0)
            ops_out = self.ops_1_to_1(inputs=inputs_temp, dim=m, degrees=m, normalization=normalization)
            ops_out = torch.stack(ops_out, dim=3)
            if self.hops != -1:
                masks = torch.unsqueeze(masks, dim=3)
                ops_out.masked_fill_(masks, 0)

            output11 = torch.einsum('nkdsb,nkdbi->nsi', coeffs, ops_out)  # N x S x m
#             print(output11.shape)
            degrees = torch.squeeze(torch.squeeze(degrees, dim=-1), dim=-1)
            degrees = torch.unsqueeze(degrees, dim=1)
            degrees[degrees==0] = 2
#             print(degrees.shape)
            output11 = torch.div(output11,degrees.float())
            
        if 1 in self.repsin and 0 in self.repsout:
            inputs_temp = inputs['1']  # N x S x m
            coeffs = []
            masks = []
            degrees = []
            for i in range(inputs_temp.shape[-1]):
                if self.hops == -1:
                    degree = [m for _ in range(inputs_temp.shape[0])]
                else:
                    _, mask, degree = self.get_mask(A, i, hops=self.hops)
                    masks.append(mask)
                    degrees.append(degree)
                coeffs_temp = self.get_coeffs10(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            degrees = torch.stack(degrees, dim=1)
            degrees = torch.unsqueeze(degrees,dim=-1)
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, inputs_temp.shape[-1], 1, 1)
            if self.hops != -1:
                masks = torch.stack(masks, dim=1)
                inputs_temp.masked_fill_(masks, 0)
            ops_out = self.ops_1_to_0(inputs=inputs_temp, dim=m, degrees=m, normalization=normalization)
            ops_out = torch.stack(ops_out, dim=3)

            output10 = torch.einsum('nkdsb,nkdb->ns', coeffs, ops_out)  # N x S
            
        if 0 in self.repsin and 2 in self.repsout:
            inputs_temp = inputs['0']  # N x S
            coeffs = []
            masks = []
            degrees = []
            for i in range(A.shape[-1]):
                if self.hops == -1:
                    degree = [m for _ in range(inputs_temp.shape[0])]
                else:
                    mask, _, degree = self.get_mask(A, i, hops=self.hops)
                    masks.append(mask)
                    degrees.append(degree)
                coeffs_temp = self.get_coeffs02(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            degrees = torch.stack(degrees, dim=1)
            degrees = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(degrees,dim=-1),dim=-1),dim=-1)
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, A.shape[-1], 1)
            if self.hops != -1:
                masks = torch.stack(masks, dim=1)
#             inputs_temp.masked_fill_(masks, 0)
            ops_out = self.ops_0_to_2(inputs=inputs_temp, dim=m, degrees=m, normalization=normalization)
            ops_out = torch.stack(ops_out, dim=3)
            if self.hops != -1:
                masks = torch.unsqueeze(masks, dim=3)
                ops_out.masked_fill_(masks, 0)

#             print(coeffs.shape)
#             print(ops_out.shape)
            output02 = torch.einsum('nkdsb,nkdbij->nsij', coeffs, ops_out)  # N x S x m x m
#             print(output02.shape)
            degrees = torch.squeeze(torch.squeeze(torch.squeeze(degrees, dim=-1), dim=-1), dim=-1)
            degrees = torch.diag_embed(degrees)
            degrees = torch.unsqueeze(degrees, dim=1)
            degrees[degrees==0] = 2
#             print(degrees.shape)
            output02 = torch.div(output02,degrees.float())

        if 0 in self.repsin and 1 in self.repsout:
            inputs_temp = inputs['0']  # N x S
            coeffs = []
            masks = []
            degrees = []
            for i in range(A.shape[-1]):
                if self.hops == -1:
                    degree = [-1 for _ in range(inputs_temp.shape[0])]
                else:
                    _, mask, degree = self.get_mask(A, i, hops=self.hops)
                    masks.append(mask)
                    degrees.append(degree)
                coeffs_temp = self.get_coeffs01(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            degrees = torch.stack(degrees, dim=1)
            degrees = torch.unsqueeze(torch.unsqueeze(degrees,dim=-1),dim=-1)
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, A.shape[-1], 1)
            if self.hops != -1:
                masks = torch.stack(masks, dim=1)
#             inputs_temp.masked_fill_(masks, 0)
            ops_out = self.ops_0_to_1(inputs=inputs_temp, dim=m, degrees=m, normalization=normalization)
            ops_out = torch.stack(ops_out, dim=3)
            if self.hops != -1:
                masks = torch.unsqueeze(masks, dim=3)
                ops_out.masked_fill_(masks, 0)

            output01 = torch.einsum('nkdsb,nkdbi->nsi', coeffs, ops_out)  # N x S x m
#             print(output01.shape)
            degrees = torch.squeeze(torch.squeeze(degrees, dim=-1), dim=-1)
            degrees = torch.unsqueeze(degrees, dim=1)
            degrees[degrees==0] = 2
#             print(degrees.shape)
            output01 = torch.div(output01,degrees.float())
            
        if 0 in self.repsin and 0 in self.repsout:
            inputs_temp = inputs['0']  # N x S
            coeffs = []
            for i in range(A.shape[-1]):
                if self.hops == -1:
                    degree = [-1 for _ in range(inputs_temp.shape[0])]
                else:
                    _, _, degree = self.get_mask(A, i, hops=self.hops)
                coeffs_temp = self.get_coeffs00(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, A.shape[-1], 1)
#             masks = torch.stack(masks, dim=1)
#             inputs_temp.masked_fill_(masks, 0)
            ops_out = torch.unsqueeze(inputs_temp, dim=3)

            output00 = torch.einsum('nkdsb,nkdb->ns', coeffs, ops_out)  # N x S x m x m

#         for i in range(inputs['2'].shape[-1]):
#             if 2 in self.repsin and 2 in self.repsout:
#                 inputs_temp = inputs['2']  # N x S x m x m
# #                 print(f'inputs_temp device : {inputs_temp.device}')
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=2, hops=self.hops)
#                     inputs_temp.masked_fill_(mask, 0)
#                 ops_out = self.ops_2_to_2(inputs=inputs_temp, dim=m, normalization=normalization)
#                 ops_out = torch.stack(ops_out, dim=2)
# #                 print(f'coeffs22 : {self.coeffs22}')
# #                 print(f'coeffs22 keys : {self.coeffs22.keys()}')
# #                 coeffs = [self.coeffs22[f'{deg}'] for deg in degree]
#                 coeffs = self.get_coeffs22(degree)
#                 coeffs = torch.stack(coeffs, dim=0)
# #                 print(f'coeffs device : {coeffs.device}')
# #                 print(f'ops_out device : {ops_out.device}')
# #                 print(f'ops_out shape : {ops_out.shape}')
#                 if i==0:
#                     output22 = torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
#                 else:
#                     output22 += torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
                   
               
#             if 2 in self.repsin and 1 in self.repsout:
#                 inputs_temp = inputs['2']  # N x S x m x m
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=2, hops=self.hops)
#                     inputs_temp.masked_fill_(mask, 0)
#                 ops_out = self.ops_2_to_1(inputs=inputs_temp, dim=m, normalization=normalization)
#                 ops_out = torch.stack(ops_out, dim=2)
#                 coeffs = self.get_coeffs21(degree)
# #                 coeffs = [self.coeffs21[f'{deg}'] for deg in degree]
#                 coeffs = torch.stack(coeffs, dim=0)
#                 if i==0:
#                     output21 = torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
#                 else:
#                     output21 += torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
                   
               
#             if 2 in self.repsin and 0 in self.repsout:
#                 inputs_temp = inputs['2']  # N x S x m x m
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=2, hops=self.hops)
#                     inputs_temp.masked_fill_(mask, 0)
#                 ops_out = self.ops_2_to_0(inputs=inputs_temp, dim=m, normalization=normalization)
#                 ops_out = torch.stack(ops_out, dim=2)
#                 coeffs = self.get_coeffs20(degree)
# #                 coeffs = [self.coeffs20[f'{deg}'] for deg in degree]
#                 coeffs = torch.stack(coeffs, dim=0)
#                 if i==0:
#                     output20 = torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
#                 else:
#                     output20 += torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
                   
               
#             if 1 in self.repsin and 2 in self.repsout:
#                 inputs_temp = inputs['1']  # N x S x m
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=1, hops=self.hops)
#                     inputs_temp.masked_fill_(mask, 0)
#                 ops_out = self.ops_1_to_2(inputs=inputs_temp, dim=m, normalization=normalization)
#                 ops_out = torch.stack(ops_out, dim=2)
#                 coeffs = self.get_coeffs12(degree)
# #                 coeffs = [self.coeffs12[f'{deg}'] for deg in degree]
#                 coeffs = torch.stack(coeffs, dim=0)
#                 if i==0:
#                     output12 = torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
#                 else:
#                     output12 += torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
                   

#             if 1 in self.repsin and 1 in self.repsout:
#                 inputs_temp = inputs['1']  # N x S x m
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=1, hops=self.hops)
#                     inputs_temp.masked_fill_(mask, 0)
#                 ops_out = self.ops_1_to_1(inputs=inputs_temp, dim=m, normalization=normalization)
#                 ops_out = torch.stack(ops_out, dim=2)
#                 coeffs = self.get_coeffs11(degree)
# #                 coeffs = [self.coeffs11[f'{deg}'] for deg in degree]
#                 coeffs = torch.stack(coeffs, dim=0)
#                 if i==0:
#                     output11 = torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
#                 else:
#                     output11 += torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
                   
                   
#             if 1 in self.repsin and 0 in self.repsout:
#                 inputs_temp = inputs['1']  # N x S x m
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=1, hops=self.hops)
#                     inputs_temp.masked_fill_(mask, 0)
#                 ops_out = self.ops_1_to_0(inputs=inputs_temp, dim=m, normalization=normalization)
#                 ops_out = torch.stack(ops_out, dim=2)
#                 coeffs = self.get_coeffs10(degree)
# #                 coeffs = [self.coeffs10[f'{deg}'] for deg in degree]
#                 coeffs = torch.stack(coeffs, dim=0)
#                 if i==0:
#                     output10 = torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
#                 else:
#                     output10 += torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
                   
       
#             if 0 in self.repsin and 2 in self.repsout:
#                 inputs_temp = inputs['0']  # N x S
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=0, hops=self.hops)
# #                     inputs_temp.masked_fill_(mask, 0)
#                 ops_out = self.ops_0_to_2(inputs=inputs_temp, dim=m, normalization=normalization)
#                 ops_out = torch.stack(ops_out, dim=2)
#                 coeffs = self.get_coeffs02(degree)
# #                 coeffs = [self.coeffs02[f'{deg}'] for deg in degree]
#                 coeffs = torch.stack(coeffs, dim=0)
#                 if i==0:
#                     output02 = torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
#                 else:
#                     output02 += torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
                   

#             if 0 in self.repsin and 1 in self.repsout:
#                 inputs_temp = inputs['0']  # N x S
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=0, hops=self.hops)
#                 ops_out = self.ops_0_to_1(inputs=inputs_temp, dim=m, normalization=normalization)
#                 ops_out = torch.stack(ops_out, dim=2)
#                 coeffs = self.get_coeffs01(degree)
# #                 coeffs = [self.coeffs01[f'{deg}'] for deg in degree]
#                 coeffs = torch.stack(coeffs, dim=0)
#                 if i==0:
#                     output01 = torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
#                 else:
#                     output01 += torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
                   
                   
#             if 0 in self.repsin and 0 in self.repsout:
#                 inputs_temp = inputs['0']  # N x S
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=0, hops=self.hops)
#                 ops_out = torch.unsqueeze(inputs_temp, dim=2)
#                 coeffs = self.get_coeffs00(degree)
# #                 coeffs = [self.coeffs00[f'{deg}'] for deg in degree]
#                 coeffs = torch.stack(coeffs, dim=0)
#                 if i==0:
#                     output00 = torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
#                 else:
#                     output00 += torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
                   
               
        if 2 in self.repsin and 2 in self.repsout:
            mat_diag_bias22 = self.diag_bias22.expand(-1,-1,inputs['2'].shape[3],inputs['2'].shape[3])
            mat_diag_bias22 = torch.mul(mat_diag_bias22, torch.eye(inputs['2'].shape[3], dtype=torch.float32, device=mat_diag_bias22.device))
            output22 = output22 + self.all_bias22 + mat_diag_bias22
        if 2 in self.repsin and 1 in self.repsout:
            output21 = output21 + self.bias21
        if 2 in self.repsin and 0 in self.repsout:
            output20 = output20 + self.bias20
        if 1 in self.repsin and 2 in self.repsout:
            output12 = output12 + self.bias12
        if 1 in self.repsin and 1 in self.repsout:
            output11 = output11 + self.bias11
        if 1 in self.repsin and 0 in self.repsout:
            output10 = output10 + self.bias10
        if 0 in self.repsin and 2 in self.repsout:
            output02 = output02 + self.bias02
        if 0 in self.repsin and 1 in self.repsout:
            output01 = output01 + self.bias01
        if 0 in self.repsin and 0 in self.repsout:
            output00 = output00 + self.bias00
       
        if 2 in self.repsout:
            if 2 in self.repsin and 1 in self.repsin and 0 in self.repsin:
                output2 = torch.cat((output22,output12,output02), dim=1)
            elif 2 in self.repsin and 1 in self.repsin:
                output2 = torch.cat((output22,output12), dim=1)
            elif 2 in self.repsin and 0 in self.repsin:
                output2 = torch.cat((output22,output02), dim=1)
            elif 1 in self.repsin and 0 in self.repsin:
                output2 = torch.cat((output12,output02), dim=1)
            elif 2 in self.repsin:
                output2 = output22
            elif 1 in self.repsin:
                output2 = output12
            elif 0 in self.repsin:
                output2 = output02
        if 1 in self.repsout:
            if 2 in self.repsin and 1 in self.repsin and 0 in self.repsin:
                output1 = torch.cat((output21,output11,output01), dim=1)
            elif 2 in self.repsin and 1 in self.repsin:
                output1 = torch.cat((output21,output11), dim=1)
            elif 2 in self.repsin and 0 in self.repsin:
                output1 = torch.cat((output21,output01), dim=1)
            elif 1 in self.repsin and 0 in self.repsin:
                output1 = torch.cat((output11,output01), dim=1)
            elif 2 in self.repsin:
                output1 = output21
            elif 1 in self.repsin:
                output1 = output11
            elif 0 in self.repsin:
                output1 = output01
        if 0 in self.repsout:
            if 2 in self.repsin and 1 in self.repsin and 0 in self.repsin:
                output0 = torch.cat((output20,output10,output00), dim=1)
            elif 2 in self.repsin and 1 in self.repsin:
                output0 = torch.cat((output20,output10), dim=1)
            elif 2 in self.repsin and 0 in self.repsin:
                output0 = torch.cat((output20,output00), dim=1)
            elif 1 in self.repsin and 0 in self.repsin:
                output0 = torch.cat((output10,output00), dim=1)
            elif 2 in self.repsin:
                output0 = output20
            elif 1 in self.repsin:
                output0 = output10
            elif 0 in self.repsin:
                output0 = output00
               
        if 2 in self.repsout and 1 in self.repsout and 0 in self.repsout:
            return {'2':output2, '1':output1, '0':output0}
        elif 2 in self.repsout and 1 in self.repsout:
            return {'2':output2, '1':output1}
        elif 2 in self.repsout and 0 in self.repsout:
            return {'2':output2, '0':output0}
        elif 1 in self.repsout and 0 in self.repsout:
            return {'1':output1, '0':output0}
        elif 2 in self.repsout:
            return {'2':output2}
        elif 1 in self.repsout:
            return {'1':output1}
        elif 0 in self.repsout:
            return {'0':output0}
        else:
            pass
        
        
class equi_multi_local_fixed2(torch.nn.Module):
    """equivariant nn layer."""

    def __init__(self, input_depth, output_depth, device, subgroups, repsin, repsout, hops):
        super(equi_multi_local_fixed2, self).__init__()
        self.repsin = repsin
        self.repsout = repsout
        self.hops = hops
        self.device = device
        self.basis_dimension_22 = 15
        self.basis_dimension_21 = 5
        self.basis_dimension_12 = 5
        self.basis_dimension_20 = 2
        self.basis_dimension_02 = 2
        self.basis_dimension_11 = 2
        self.basis_dimension_10 = 1
        self.basis_dimension_01 = 1
        self.basis_dimension_00 = 1
       
        if 2 in repsin and 2 in repsout:
            coeffs_values_22 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_22), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs22 = torch.nn.ParameterDict()
#             for sub in subgroups:  ## This isn't working on multiple gpus due to a pytorch bug, have to hard code subgroups for now https://github.com/pytorch/pytorch/issues/36035
#                 self.coeffs22[f'{sub}'] = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_1 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_0 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_2 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_3 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_4 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_5 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.diag_bias22 = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
            self.all_bias22 = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
       
        if 2 in repsin and 1 in repsout:
            coeffs_values_21 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_21), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs21 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs21[f'{sub}'] = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_1 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_0 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_2 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_3 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_4 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_5 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.bias21 = torch.nn.Parameter(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
           
        if 1 in repsin and 2 in repsout:
            coeffs_values_12 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_12), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs12 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs12[f'{sub}'] = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_1 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_0 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_2 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_3 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_4 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_5 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.bias12 = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
           
        if 2 in repsin and 0 in repsout:
            coeffs_values_20 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_20), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs20 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs20[f'{sub}'] = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_1 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_0 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_2 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_3 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_4 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_5 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.bias20 = torch.nn.Parameter(torch.zeros((1, output_depth), dtype=torch.float32), requires_grad=True)
           
        if 0 in repsin and 2 in repsout:
            coeffs_values_02 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_02), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs02 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs02[f'{sub}'] = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_1 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_0 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_2 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_3 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_4 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_5 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.bias02 = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
           
        if 1 in repsin and 1 in repsout:
            coeffs_values_11 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_11), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs11 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs11[f'{sub}'] = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_1 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_0 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_2 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_3 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_4 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_5 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.bias11 = torch.nn.Parameter(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
           
        if 1 in repsin and 0 in repsout:
            coeffs_values_10 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_10), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs10 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs10[f'{sub}'] = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_1 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_0 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_2 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_3 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_4 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_5 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.bias10 = torch.nn.Parameter(torch.zeros((1, output_depth), dtype=torch.float32), requires_grad=True)
           
        if 0 in repsin and 1 in repsout:
            coeffs_values_01 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_01), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs01 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs01[f'{sub}'] = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_1 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_0 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_2 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_3 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_4 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_5 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.bias01 = torch.nn.Parameter(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
           
        if 0 in repsin and 0 in repsout:
            coeffs_values_00 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_00), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs00 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs00[f'{sub}'] = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_1 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_0 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_2 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_3 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_4 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_5 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.bias00 = torch.nn.Parameter(torch.zeros((1, output_depth), dtype=torch.float32), requires_grad=True)
               

    def ops_2_to_2(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D x m x m
        diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
        sum_diag_part = torch.sum(diag_part, dim=3, keepdim=True)  # N x D x 1
        sum_of_rows = torch.sum(inputs, dim=4)  # N x D x m
        sum_of_cols = torch.sum(inputs, dim=3)  # N x D x m
        sum_all = torch.sum(sum_of_rows, dim=3)  # N x D

        # op1 - (1234) - extract diag
        op1 = torch.diag_embed(diag_part)  # N x D x m x m
#         print(f'inputs shape : {inputs.shape}')
#         print(f'diag_part shape : {diag_part.shape}')
#         print(f'dim : {dim}')
#         print(f'op1 max : {torch.max(op1)}')

        # op2 - (1234) + (12)(34) - place sum of diag on diag
        op2 = torch.diag_embed(sum_diag_part.repeat(1, 1, 1, dim)) - op1
#         print(f'op2 max : {torch.max(op2)}')

        # op3 - (1234) + (123)(4) - place sum of row i on diag ii
        op3 = torch.diag_embed(sum_of_rows) - op1
#         print(f'op3 max : {torch.max(op3)}')

        # op4 - (1234) + (124)(3) - place sum of col i on diag ii
        op4 = torch.diag_embed(sum_of_cols) - op1
#         print(f'op4 max : {torch.max(op4)}')
       
        # op10 - (1234) + (14)(23) - identity
        op10 = inputs  - op1
#         print(f'op10max : {torch.max(op10)}')

        # op11 - (1234) + (13)(24) - transpose
        op11 = inputs.permute(0, 1, 2, 4, 3) - op1
#         print(f'op11 max : {torch.max(op11)}')

        # op12 - (1234) + (234)(1) - place ii element in row i
        op12 = torch.unsqueeze(diag_part, dim=4).repeat(1, 1, 1, 1, dim) - op1
#         print(f'op12 max : {torch.max(op12)}')

        # op13 - (1234) + (134)(2) - place ii element in col i
        op13 = torch.unsqueeze(diag_part, dim=3).repeat(1, 1, 1, dim, 1) - op1
#         print(f'op13 max : {torch.max(op13)}')

        # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
        op5 = torch.diag_embed(torch.unsqueeze(sum_all, dim=3).repeat(1, 1, 1, dim)) - op1 - op4 - op3 - op2
#         print(f'op5 max : {torch.max(op5)}')

        # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
        op6 = torch.unsqueeze(sum_of_cols, dim=4).repeat(1, 1, 1, 1, dim) - op1 - op10 - op11 - op4
#         print(f'op6 max : {torch.max(op6)}')

        # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
        op7 = torch.unsqueeze(sum_of_rows, dim=4).repeat(1, 1, 1, 1, dim) - op1 - op10 - op12 - op3
#         print(f'op7 max : {torch.max(op7)}')

        # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
        op8 = torch.unsqueeze(sum_of_cols, dim=3).repeat(1, 1, 1, dim, 1) - op1 - op13 - op10 - op4
#         print(f'op8 max : {torch.max(op8)}')

        # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
        op9 = torch.unsqueeze(sum_of_rows, dim=3).repeat(1, 1, 1, dim, 1) - op1 - op11 - op13 - op3
#         print(f'op9 max : {torch.max(op9)}')

        # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
        op14 = torch.unsqueeze(sum_diag_part, dim=4).repeat(1, 1, 1, dim, dim) - op1 - op12 - op13 - op2
#         print(f'op14 max : {torch.max(op14)}')

        # op15 - sum of all ops - place sum of all entries in all entries
        op15 = torch.unsqueeze(torch.unsqueeze(sum_all, dim=3), dim=4).repeat(1, 1, 1, dim, dim) - op1 - op2 - op3 - op4 - op5 - op6 - op7 - op8 - op9 - op10 - op11 - op12 - op13 - op14
#         print(f'op15 max : {torch.max(op15)}')

        if dim > 2:
            if normalization is not None:
                float_dim = degrees.type(torch.FloatTensor)
#                 print(float_dim)
                extra_norm = torch.max(sum_all)
                if normalization is 'inf':
                    op2 = torch.div(op2, float_dim-1)
                    op3 = torch.div(op3, float_dim-1)
                    op4 = torch.div(op4, float_dim-1)
                    op5 = torch.div(op5, (float_dim-1)*(float_dim-2))
                    op6 = torch.div(op6, float_dim-2)
                    op7 = torch.div(op7, float_dim-2)
                    op8 = torch.div(op8, float_dim-2)
                    op9 = torch.div(op9, float_dim-2)
    #                 op10 = torch.div(op10, float_dim-2)
    #                 op11 = torch.div(op11, float_dim)
    #                 op12 = torch.div(op12, float_dim)
    #                 op13 = torch.div(op13, float_dim)
                    op14 = torch.div(op14, float_dim-2)
                    op15 = torch.div(op15, (float_dim-2)*(float_dim-3))
               
        return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]

   
    def ops_2_to_1(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D x m x m
        diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
        sum_diag_part = torch.sum(diag_part, dim=3, keepdim=True)  # N x D x 1
        sum_of_rows = torch.sum(inputs, dim=4)  # N x D x m
        sum_of_cols = torch.sum(inputs, dim=3)  # N x D x m
        sum_all = torch.sum(sum_of_rows, dim=3)  # N x D

        # op1 - (123) - extract diag
        op1 = diag_part  # N x D x m

        # op2 - (123) + (12)(3) - tile sum of diag part
        op2 = sum_diag_part.repeat(1, 1, 1, dim)  # N x D x m

        # op3 - (123) + (13)(2) - place sum of row i in element i
        op3 = sum_of_rows  # N x D x m

        # op4 - (123) + (23)(1) - place sum of col i in element i
        op4 = sum_of_cols  # N x D x m

        # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
        op5 = torch.unsqueeze(sum_all, dim=3).repeat(1, 1, 1, dim)  # N x D x m


        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)
                op3 = torch.div(op3, float_dim)
                op4 = torch.div(op4, float_dim)
                op5 = torch.div(op5, float_dim ** 2)

        return [op1, op2, op3, op4, op5]

   
    def ops_2_to_0(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D x m x m
        diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
        sum_diag_part = torch.sum(diag_part, dim=3)  # N x D
        sum_of_rows = torch.sum(inputs, dim=4)  # N x D x m
        sum_all = torch.sum(sum_of_rows, dim=3)  # N x D

        # op1 -
        op1 = sum_diag_part  # N x D

        # op2 -
        op2 = sum_all  # N x D


        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
                op1 = torch.div(op1, float_dim)
                op2 = torch.div(op2, float_dim ** 2)

        return [op1, op2]


    def ops_0_to_2(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D
       
        # op1 - (123) - place on diag
        op1 = torch.unsqueeze(inputs, dim=3).repeat(1, 1, 1, dim)  # N x D x m
        op1 = torch.diag_embed(op1)  # N x D x m x m
#         op1 = F.pad(op1, (0,dim-inputs.shape[-1],0,dim-inputs.shape[-1]))

        # op2 - (123) + (12)(3) - tile in all entries
        op2 = torch.unsqueeze(torch.unsqueeze(inputs, dim=3), dim=4).repeat(1, 1, 1, dim, dim)  # N x D x m x m

        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
#                 op1 = torch.div(op1, float_dim)
                op2 = torch.div(op2, float_dim ** 2)

        return [op1, op2]


    def ops_1_to_2(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D x m
        sum_all = torch.sum(inputs, dim=3, keepdims=True)  # N x D x 1

        # op1 - (123) - place on diag
        op1 = torch.diag_embed(inputs)  # N x D x m x m

        # op2 - (123) + (12)(3) - tile sum on diag
        op2 = torch.diag_embed(sum_all.repeat(1, 1, 1, dim))  # N x D x m x m

        # op3 - (123) + (13)(2) - tile element i in row i
        op3 = torch.unsqueeze(inputs, dim=3).repeat(1, 1, 1, dim, 1)  # N x D x m x m

        # op4 - (123) + (23)(1) - tile element i in col i
        op4 = torch.unsqueeze(inputs, dim=4).repeat(1, 1, 1, 1, dim)  # N x D x m x m

        # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
        op5 = torch.unsqueeze(sum_all, dim=4).repeat(1, 1, 1, dim, dim)  # N x D x m x m

        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)
                op5 = torch.div(op5, float_dim)

        return [op1, op2, op3, op4, op5]

   
    def ops_1_to_1(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D x m
        sum_all = torch.sum(inputs, dim=3, keepdims=True)  # N x D x 1

        # op1 - (12) - identity
        op1 = inputs  # N x D x m

        # op2 - (1)(2) - tile sum of all
        op2 = sum_all.repeat(1, 1, 1, dim)  # N x D x m

        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)

        return [op1, op2]

   
    def ops_1_to_0(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D x m
        sum_all = torch.sum(inputs, dim=3)  # N x D

        # op1 - (12) - identity
        op1 = sum_all  # N x D

        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
                op1 = torch.div(op1, float_dim)

        return [op1]


    def ops_0_to_1(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D

        # op1
        op1 = torch.unsqueeze(inputs, dim=3).repeat(1, 1, 1, dim)  # N x D x m
#         op1 = torch.diag_embed(inputs)  # N x D x m
#         op1 = F.pad(op1, (0,dim-inputs.shape[-1]))

#         if normalization is not None:
#             float_dim = degrees.type(torch.float32)
#             if normalization is 'inf':
#                 op1 = torch.div(op1, float_dim)

        return [op1]
   
    def get_mask(self, A, i, hops=1):
        indices = (A==1).nonzero(as_tuple=False)
#         print(f'indices : {indices.shape}')
        if hops==1:
            indices = indices[indices[:,1]==i]
        else:
            indices1hop = indices[indices[:,1]==i]
#             print(f'indices1hop : {indices1hop.shape}')
            indices2hop = []
#             print(f'ind shape : {indices1hop[0].shape}')
            for ind in indices1hop:
                inds = indices[(indices[:,0]==ind[0]) & (indices[:,1]==ind[2])]
                for val in inds:
                    indices2hop.append(val)
            if hops==2:
                indices = torch.stack(indices2hop, dim=0)
            else:
                indices2hop = torch.stack(indices2hop, dim=0)
               
        mask = torch.ones(A.shape, device=A.device)
#         print(indices)
        batches = torch.unique(indices[:,0], return_counts=True)[0]
#         print(batches)
        for batch in batches:
            indices_i = indices[indices[:,0]==batch]
#             print(indices_i)
            for i in indices_i[:,2]:
                for j in indices_i[:,2]:
#                     print(f'batch : {batch}, index1 : {i}, index : {j}')
                    mask[batch,i,j] = 0
#         for index in indices:
#             mask[index[0],index[1],index[2]] = 0
#             mask[index[0],index[2],index[2]] = 0

        mask = torch.min(mask, torch.transpose(mask,-2,-1))
        mask = mask.bool()
        mask = torch.unsqueeze(mask, dim=1)
       
        mask1 = mask[:,:,i,:]
#         print(f'rep : {rep} - mask shape : {mask.shape}')
       
        degree = torch.unique(indices[:,0], return_counts=True)[1]
#         print(f'degree : {degree}')
#         print(f'mask device : {mask.device}')
#         print(f'degree device : {degree.device}')
       
        return mask, mask1, degree
   
    def get_coeffs22(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs22_1)
            elif deg == 2:
                coeffs.append(self.coeffs22_2)
            elif deg == 3:
                coeffs.append(self.coeffs22_3)
            elif deg == 4:
                coeffs.append(self.coeffs22_4)
            elif deg == 5:
                coeffs.append(self.coeffs22_5)
            else:
                coeffs.append(self.coeffs22_0)
        return coeffs
           
    def get_coeffs21(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs21_1)
            elif deg == 2:
                coeffs.append(self.coeffs21_2)
            elif deg == 3:
                coeffs.append(self.coeffs21_3)
            elif deg == 4:
                coeffs.append(self.coeffs21_4)
            elif deg == 5:
                coeffs.append(self.coeffs21_5)
            else:
                coeffs.append(self.coeffs21_0)
        return coeffs
   
    def get_coeffs12(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs12_1)
            elif deg == 2:
                coeffs.append(self.coeffs12_2)
            elif deg == 3:
                coeffs.append(self.coeffs12_3)
            elif deg == 4:
                coeffs.append(self.coeffs12_4)
            elif deg == 5:
                coeffs.append(self.coeffs12_5)
            else:
                coeffs.append(self.coeffs12_0)
        return coeffs
   
    def get_coeffs11(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs11_1)
            elif deg == 2:
                coeffs.append(self.coeffs11_2)
            elif deg == 3:
                coeffs.append(self.coeffs11_3)
            elif deg == 4:
                coeffs.append(self.coeffs11_4)
            elif deg == 5:
                coeffs.append(self.coeffs11_5)
            else:
                coeffs.append(self.coeffs11_0)
        return coeffs
   
    def get_coeffs10(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs10_1)
            elif deg == 2:
                coeffs.append(self.coeffs10_2)
            elif deg == 3:
                coeffs.append(self.coeffs10_3)
            elif deg == 4:
                coeffs.append(self.coeffs10_4)
            elif deg == 5:
                coeffs.append(self.coeffs10_5)
            else:
                coeffs.append(self.coeffs10_0)
        return coeffs
   
    def get_coeffs01(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs01_1)
            elif deg == 2:
                coeffs.append(self.coeffs01_2)
            elif deg == 3:
                coeffs.append(self.coeffs01_3)
            elif deg == 4:
                coeffs.append(self.coeffs01_4)
            elif deg == 5:
                coeffs.append(self.coeffs01_5)
            else:
                coeffs.append(self.coeffs01_0)
        return coeffs
   
    def get_coeffs20(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs20_1)
            elif deg == 2:
                coeffs.append(self.coeffs20_2)
            elif deg == 3:
                coeffs.append(self.coeffs20_3)
            elif deg == 4:
                coeffs.append(self.coeffs20_4)
            elif deg == 5:
                coeffs.append(self.coeffs20_5)
            else:
                coeffs.append(self.coeffs20_0)
        return coeffs
   
    def get_coeffs02(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs02_1)
            elif deg == 2:
                coeffs.append(self.coeffs02_2)
            elif deg == 3:
                coeffs.append(self.coeffs02_3)
            elif deg == 4:
                coeffs.append(self.coeffs02_4)
            elif deg == 5:
                coeffs.append(self.coeffs02_5)
            else:
                coeffs.append(self.coeffs02_0)
        return coeffs
   
    def get_coeffs00(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs00_1)
            elif deg == 2:
                coeffs.append(self.coeffs00_2)
            elif deg == 3:
                coeffs.append(self.coeffs00_3)
            elif deg == 4:
                coeffs.append(self.coeffs00_4)
            elif deg == 5:
                coeffs.append(self.coeffs00_5)
            else:
                coeffs.append(self.coeffs00_0)
        return coeffs
   
   

    def forward(self, inputs, A, normalization='inf'):
#         print(f'A device : {A.device}')
#         print(f'coeffs22 : {self.coeffs22}')
#         print(self.coeffs22)
        print(f'A shape : {A.shape}')
        A_sparse = dense_to_sparse(A)
        print(f'A_sparse shape : {A_sparse[0].shape}')
        print(f'A_sparse shape : {A_sparse[1].shape}')
        print(f'A : {A}')
        print(f'A_sparse : {A_sparse[0]}')
        print(f'A_sparse : {A_sparse[1]}')
        
        print(f'inputs : {inputs["2"].shape}')
        edge_index = A_sparse[0]
        
        print(f'edge_index 0 : {edge_index[:,edge_index[0,:]==0]}')
        

        m = torch.tensor(inputs['2'].shape[3], dtype=torch.int32, device=A.device)  # extract dimension
#         print(f'm device : {m.device}')

        if 2 in self.repsin and 2 in self.repsout:
            inputs_temp = inputs['2']  # N x S x m x m
            coeffs = []
            masks = []
            degrees = []
            for i in range(inputs_temp.shape[-1]):
                if self.hops == -1:
                    degree = [m for _ in range(inputs_temp.shape[0])]
                else:
                    mask, _, degree = self.get_mask(A, i, hops=self.hops)
#                     print(i)
#                     print(mask)
                    masks.append(mask)
                    degrees.append(degree)
                coeffs_temp = self.get_coeffs22(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            degrees = torch.stack(degrees, dim=1)
            print(f'degrees shape : {degrees.shape}')
            degrees = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(degrees,dim=-1),dim=-1),dim=-1)
            print(f'degrees shape : {degrees.shape}')
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, inputs_temp.shape[-1], 1, 1, 1)
            if self.hops != -1:
                masks = torch.stack(masks, dim=1)
                print(f'masks shape : {masks.shape}')
                print(f'inputs_temp shape : {inputs_temp.shape}')
                inputs_temp.masked_fill_(masks, 0)
            ops_out = self.ops_2_to_2(inputs=inputs_temp, dim=m, degrees=m, normalization=normalization)
            ops_out = torch.stack(ops_out, dim=3)
            if self.hops != -1:
                masks = torch.unsqueeze(masks, dim=3)
                ops_out.masked_fill_(masks, 0)

            output22 = torch.einsum('nkdsb,nkdbij->nsij', coeffs, ops_out)  # N x S x m x m
#             print(output22)
            degrees = torch.squeeze(torch.squeeze(torch.squeeze(degrees, dim=-1), dim=-1), dim=-1)
            degrees = torch.diag_embed(degrees)
            degrees = torch.unsqueeze(degrees, dim=1)
            degrees[degrees==0] = 2
#             print(degrees)
#             degrees = torch.unsqueeze(degrees, dim=-1).repeat(1, 1, 1, degrees.shape[-1])
#             print(degrees.shape)
            output22 = torch.div(output22,degrees.float())
#             print(output22)

        if 2 in self.repsin and 1 in self.repsout:
            inputs_temp = inputs['2']  # N x S x m x m
            coeffs = []
            masksin = []
            masksout = []
            degrees = []
            for i in range(inputs_temp.shape[-1]):
                if self.hops == -1:
                    degree = [m for _ in range(inputs_temp.shape[0])]
                else:
                    mask2, mask1, degree = self.get_mask(A, i, hops=self.hops)
                    masksin.append(mask2)
                    masksout.append(mask1)
                    degrees.append(degree)
                coeffs_temp = self.get_coeffs21(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            degrees = torch.stack(degrees, dim=1)
            degrees = torch.unsqueeze(torch.unsqueeze(degrees,dim=-1),dim=-1)
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, inputs_temp.shape[-1], 1, 1, 1)
            if self.hops != -1:
                masksin = torch.stack(masksin, dim=1)
                inputs_temp.masked_fill_(masksin, 0)
            ops_out = self.ops_2_to_1(inputs=inputs_temp, dim=m, degrees=m, normalization=normalization)
            ops_out = torch.stack(ops_out, dim=3)
            if self.hops != -1:
                masksout = torch.stack(masksout, dim=1)
                masksout = torch.unsqueeze(masksout, dim=3)
                ops_out.masked_fill_(masksout, 0)
                
            output21 = torch.einsum('nkdsb,nkdbi->nsi', coeffs, ops_out)  # N x S x m 
            degrees = torch.squeeze(torch.squeeze(degrees, dim=-1), dim=-1)
            degrees = torch.unsqueeze(degrees, dim=1)
            degrees[degrees==0] = 2
#             print(output21.shape)
#             print(degrees.shape)
            output21 = torch.div(output21,degrees.float())
            
        if 2 in self.repsin and 0 in self.repsout:
            inputs_temp = inputs['2']  # N x S x m x m
            coeffs = []
            masks = []
            degrees = []
            for i in range(inputs_temp.shape[-1]):
                if self.hops == -1:
                    degree = [m for _ in range(inputs_temp.shape[0])]
                else:
                    mask, _, degree = self.get_mask(A, i, hops=self.hops)
                    masks.append(mask)
                    degrees.append(degree)
                coeffs_temp = self.get_coeffs20(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            degrees = torch.stack(degrees, dim=1)
            degrees = torch.unsqueeze(degrees,dim=-1)
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, inputs_temp.shape[-1], 1, 1, 1)
            if self.hops != -1:
                masks = torch.stack(masks, dim=1)
                inputs_temp.masked_fill_(masks, 0)
            ops_out = self.ops_2_to_0(inputs=inputs_temp, dim=m, degrees=m, normalization=normalization)
            ops_out = torch.stack(ops_out, dim=3)

            output20 = torch.einsum('nkdsb,nkdb->ns', coeffs, ops_out)  # N x S
            
        if 1 in self.repsin and 2 in self.repsout:
            inputs_temp = inputs['1']  # N x S x m
            coeffs = []
            masksin = []
            masksout = []
            degrees = []
            for i in range(inputs_temp.shape[-1]):
                if self.hops == -1:
                    degree = [m for _ in range(inputs_temp.shape[0])]
                else:
                    mask2, mask1, degree = self.get_mask(A, i, hops=self.hops)
                    masksin.append(mask1)
                    masksout.append(mask2)
                    degrees.append(degree)
                coeffs_temp = self.get_coeffs12(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            degrees = torch.stack(degrees, dim=1)
            degrees = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(degrees,dim=-1),dim=-1),dim=-1)
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, inputs_temp.shape[-1], 1, 1)
            if self.hops != -1:
                masksin = torch.stack(masksin, dim=1)
                inputs_temp.masked_fill_(masksin, 0)
            ops_out = self.ops_1_to_2(inputs=inputs_temp, dim=m, degrees=m, normalization=normalization)
            ops_out = torch.stack(ops_out, dim=3)
            if self.hops != -1:
                masksout = torch.stack(masksout, dim=1)
                masksout = torch.unsqueeze(masksout, dim=3)
                ops_out.masked_fill_(masksout, 0)

            output12 = torch.einsum('nkdsb,nkdbij->nsij', coeffs, ops_out)  # N x S x m x m
#             print(output12.shape)
            degrees = torch.squeeze(torch.squeeze(torch.squeeze(degrees, dim=-1), dim=-1), dim=-1)
            degrees = torch.diag_embed(degrees)
            degrees = torch.unsqueeze(degrees, dim=1)
            degrees[degrees==0] = 2
#             print(degrees.shape)
            output12 = torch.div(output12,degrees.float())

        if 1 in self.repsin and 1 in self.repsout:
            inputs_temp = inputs['1']  # N x S x m
            coeffs = []
            masks = []
            degrees = []
            for i in range(inputs_temp.shape[-1]):
                if self.hops == -1:
                    degree = [m for _ in range(inputs_temp.shape[0])]
                else:
                    _, mask, degree = self.get_mask(A, i, hops=self.hops)
                    masks.append(mask)
                    degrees.append(degree)
                coeffs_temp = self.get_coeffs11(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            degrees = torch.stack(degrees, dim=1)
            degrees = torch.unsqueeze(torch.unsqueeze(degrees,dim=-1),dim=-1)
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, inputs_temp.shape[-1], 1, 1)
            if self.hops != -1:
                masks = torch.stack(masks, dim=1)
                inputs_temp.masked_fill_(masks, 0)
            ops_out = self.ops_1_to_1(inputs=inputs_temp, dim=m, degrees=m, normalization=normalization)
            ops_out = torch.stack(ops_out, dim=3)
            if self.hops != -1:
                masks = torch.unsqueeze(masks, dim=3)
                ops_out.masked_fill_(masks, 0)

            output11 = torch.einsum('nkdsb,nkdbi->nsi', coeffs, ops_out)  # N x S x m
#             print(output11.shape)
            degrees = torch.squeeze(torch.squeeze(degrees, dim=-1), dim=-1)
            degrees = torch.unsqueeze(degrees, dim=1)
            degrees[degrees==0] = 2
#             print(degrees.shape)
            output11 = torch.div(output11,degrees.float())
            
        if 1 in self.repsin and 0 in self.repsout:
            inputs_temp = inputs['1']  # N x S x m
            coeffs = []
            masks = []
            degrees = []
            for i in range(inputs_temp.shape[-1]):
                if self.hops == -1:
                    degree = [m for _ in range(inputs_temp.shape[0])]
                else:
                    _, mask, degree = self.get_mask(A, i, hops=self.hops)
                    masks.append(mask)
                    degrees.append(degree)
                coeffs_temp = self.get_coeffs10(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            degrees = torch.stack(degrees, dim=1)
            degrees = torch.unsqueeze(degrees,dim=-1)
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, inputs_temp.shape[-1], 1, 1)
            if self.hops != -1:
                masks = torch.stack(masks, dim=1)
                inputs_temp.masked_fill_(masks, 0)
            ops_out = self.ops_1_to_0(inputs=inputs_temp, dim=m, degrees=m, normalization=normalization)
            ops_out = torch.stack(ops_out, dim=3)

            output10 = torch.einsum('nkdsb,nkdb->ns', coeffs, ops_out)  # N x S
            
        if 0 in self.repsin and 2 in self.repsout:
            inputs_temp = inputs['0']  # N x S
            coeffs = []
            masks = []
            degrees = []
            for i in range(A.shape[-1]):
                if self.hops == -1:
                    degree = [m for _ in range(inputs_temp.shape[0])]
                else:
                    mask, _, degree = self.get_mask(A, i, hops=self.hops)
                    masks.append(mask)
                    degrees.append(degree)
                coeffs_temp = self.get_coeffs02(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            degrees = torch.stack(degrees, dim=1)
            degrees = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(degrees,dim=-1),dim=-1),dim=-1)
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, A.shape[-1], 1)
            if self.hops != -1:
                masks = torch.stack(masks, dim=1)
#             inputs_temp.masked_fill_(masks, 0)
            ops_out = self.ops_0_to_2(inputs=inputs_temp, dim=m, degrees=m, normalization=normalization)
            ops_out = torch.stack(ops_out, dim=3)
            if self.hops != -1:
                masks = torch.unsqueeze(masks, dim=3)
                ops_out.masked_fill_(masks, 0)

#             print(coeffs.shape)
#             print(ops_out.shape)
            output02 = torch.einsum('nkdsb,nkdbij->nsij', coeffs, ops_out)  # N x S x m x m
#             print(output02.shape)
            degrees = torch.squeeze(torch.squeeze(torch.squeeze(degrees, dim=-1), dim=-1), dim=-1)
            degrees = torch.diag_embed(degrees)
            degrees = torch.unsqueeze(degrees, dim=1)
            degrees[degrees==0] = 2
#             print(degrees.shape)
            output02 = torch.div(output02,degrees.float())

        if 0 in self.repsin and 1 in self.repsout:
            inputs_temp = inputs['0']  # N x S
            coeffs = []
            masks = []
            degrees = []
            for i in range(A.shape[-1]):
                if self.hops == -1:
                    degree = [-1 for _ in range(inputs_temp.shape[0])]
                else:
                    _, mask, degree = self.get_mask(A, i, hops=self.hops)
                    masks.append(mask)
                    degrees.append(degree)
                coeffs_temp = self.get_coeffs01(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            degrees = torch.stack(degrees, dim=1)
            degrees = torch.unsqueeze(torch.unsqueeze(degrees,dim=-1),dim=-1)
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, A.shape[-1], 1)
            if self.hops != -1:
                masks = torch.stack(masks, dim=1)
#             inputs_temp.masked_fill_(masks, 0)
            ops_out = self.ops_0_to_1(inputs=inputs_temp, dim=m, degrees=m, normalization=normalization)
            ops_out = torch.stack(ops_out, dim=3)
            if self.hops != -1:
                masks = torch.unsqueeze(masks, dim=3)
                ops_out.masked_fill_(masks, 0)

            output01 = torch.einsum('nkdsb,nkdbi->nsi', coeffs, ops_out)  # N x S x m
#             print(output01.shape)
            degrees = torch.squeeze(torch.squeeze(degrees, dim=-1), dim=-1)
            degrees = torch.unsqueeze(degrees, dim=1)
            degrees[degrees==0] = 2
#             print(degrees.shape)
            output01 = torch.div(output01,degrees.float())
            
        if 0 in self.repsin and 0 in self.repsout:
            inputs_temp = inputs['0']  # N x S
            coeffs = []
            for i in range(A.shape[-1]):
                if self.hops == -1:
                    degree = [-1 for _ in range(inputs_temp.shape[0])]
                else:
                    _, _, degree = self.get_mask(A, i, hops=self.hops)
                coeffs_temp = self.get_coeffs00(degree)
                coeffs_temp = torch.stack(coeffs_temp, dim=0)
                coeffs.append(coeffs_temp)
            coeffs = torch.stack(coeffs, dim=1)
            inputs_temp = torch.unsqueeze(inputs_temp, dim=1).repeat(1, A.shape[-1], 1)
#             masks = torch.stack(masks, dim=1)
#             inputs_temp.masked_fill_(masks, 0)
            ops_out = torch.unsqueeze(inputs_temp, dim=3)

            output00 = torch.einsum('nkdsb,nkdb->ns', coeffs, ops_out)  # N x S x m x m

#         for i in range(inputs['2'].shape[-1]):
#             if 2 in self.repsin and 2 in self.repsout:
#                 inputs_temp = inputs['2']  # N x S x m x m
# #                 print(f'inputs_temp device : {inputs_temp.device}')
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=2, hops=self.hops)
#                     inputs_temp.masked_fill_(mask, 0)
#                 ops_out = self.ops_2_to_2(inputs=inputs_temp, dim=m, normalization=normalization)
#                 ops_out = torch.stack(ops_out, dim=2)
# #                 print(f'coeffs22 : {self.coeffs22}')
# #                 print(f'coeffs22 keys : {self.coeffs22.keys()}')
# #                 coeffs = [self.coeffs22[f'{deg}'] for deg in degree]
#                 coeffs = self.get_coeffs22(degree)
#                 coeffs = torch.stack(coeffs, dim=0)
# #                 print(f'coeffs device : {coeffs.device}')
# #                 print(f'ops_out device : {ops_out.device}')
# #                 print(f'ops_out shape : {ops_out.shape}')
#                 if i==0:
#                     output22 = torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
#                 else:
#                     output22 += torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
                   
               
#             if 2 in self.repsin and 1 in self.repsout:
#                 inputs_temp = inputs['2']  # N x S x m x m
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=2, hops=self.hops)
#                     inputs_temp.masked_fill_(mask, 0)
#                 ops_out = self.ops_2_to_1(inputs=inputs_temp, dim=m, normalization=normalization)
#                 ops_out = torch.stack(ops_out, dim=2)
#                 coeffs = self.get_coeffs21(degree)
# #                 coeffs = [self.coeffs21[f'{deg}'] for deg in degree]
#                 coeffs = torch.stack(coeffs, dim=0)
#                 if i==0:
#                     output21 = torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
#                 else:
#                     output21 += torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
                   
               
#             if 2 in self.repsin and 0 in self.repsout:
#                 inputs_temp = inputs['2']  # N x S x m x m
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=2, hops=self.hops)
#                     inputs_temp.masked_fill_(mask, 0)
#                 ops_out = self.ops_2_to_0(inputs=inputs_temp, dim=m, normalization=normalization)
#                 ops_out = torch.stack(ops_out, dim=2)
#                 coeffs = self.get_coeffs20(degree)
# #                 coeffs = [self.coeffs20[f'{deg}'] for deg in degree]
#                 coeffs = torch.stack(coeffs, dim=0)
#                 if i==0:
#                     output20 = torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
#                 else:
#                     output20 += torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
                   
               
#             if 1 in self.repsin and 2 in self.repsout:
#                 inputs_temp = inputs['1']  # N x S x m
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=1, hops=self.hops)
#                     inputs_temp.masked_fill_(mask, 0)
#                 ops_out = self.ops_1_to_2(inputs=inputs_temp, dim=m, normalization=normalization)
#                 ops_out = torch.stack(ops_out, dim=2)
#                 coeffs = self.get_coeffs12(degree)
# #                 coeffs = [self.coeffs12[f'{deg}'] for deg in degree]
#                 coeffs = torch.stack(coeffs, dim=0)
#                 if i==0:
#                     output12 = torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
#                 else:
#                     output12 += torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
                   

#             if 1 in self.repsin and 1 in self.repsout:
#                 inputs_temp = inputs['1']  # N x S x m
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=1, hops=self.hops)
#                     inputs_temp.masked_fill_(mask, 0)
#                 ops_out = self.ops_1_to_1(inputs=inputs_temp, dim=m, normalization=normalization)
#                 ops_out = torch.stack(ops_out, dim=2)
#                 coeffs = self.get_coeffs11(degree)
# #                 coeffs = [self.coeffs11[f'{deg}'] for deg in degree]
#                 coeffs = torch.stack(coeffs, dim=0)
#                 if i==0:
#                     output11 = torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
#                 else:
#                     output11 += torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
                   
                   
#             if 1 in self.repsin and 0 in self.repsout:
#                 inputs_temp = inputs['1']  # N x S x m
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=1, hops=self.hops)
#                     inputs_temp.masked_fill_(mask, 0)
#                 ops_out = self.ops_1_to_0(inputs=inputs_temp, dim=m, normalization=normalization)
#                 ops_out = torch.stack(ops_out, dim=2)
#                 coeffs = self.get_coeffs10(degree)
# #                 coeffs = [self.coeffs10[f'{deg}'] for deg in degree]
#                 coeffs = torch.stack(coeffs, dim=0)
#                 if i==0:
#                     output10 = torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
#                 else:
#                     output10 += torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
                   
       
#             if 0 in self.repsin and 2 in self.repsout:
#                 inputs_temp = inputs['0']  # N x S
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=0, hops=self.hops)
# #                     inputs_temp.masked_fill_(mask, 0)
#                 ops_out = self.ops_0_to_2(inputs=inputs_temp, dim=m, normalization=normalization)
#                 ops_out = torch.stack(ops_out, dim=2)
#                 coeffs = self.get_coeffs02(degree)
# #                 coeffs = [self.coeffs02[f'{deg}'] for deg in degree]
#                 coeffs = torch.stack(coeffs, dim=0)
#                 if i==0:
#                     output02 = torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
#                 else:
#                     output02 += torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
                   

#             if 0 in self.repsin and 1 in self.repsout:
#                 inputs_temp = inputs['0']  # N x S
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=0, hops=self.hops)
#                 ops_out = self.ops_0_to_1(inputs=inputs_temp, dim=m, normalization=normalization)
#                 ops_out = torch.stack(ops_out, dim=2)
#                 coeffs = self.get_coeffs01(degree)
# #                 coeffs = [self.coeffs01[f'{deg}'] for deg in degree]
#                 coeffs = torch.stack(coeffs, dim=0)
#                 if i==0:
#                     output01 = torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
#                 else:
#                     output01 += torch.einsum('ndsb,ndbi->nsi', coeffs, ops_out)  # N x S x m
                   
                   
#             if 0 in self.repsin and 0 in self.repsout:
#                 inputs_temp = inputs['0']  # N x S
#                 if self.hops == -1:
#                     degree = [-1 for _ in range(inputs_temp.shape[0])]
#                 else:
#                     mask, degree = self.get_mask(A, i, rep=0, hops=self.hops)
#                 ops_out = torch.unsqueeze(inputs_temp, dim=2)
#                 coeffs = self.get_coeffs00(degree)
# #                 coeffs = [self.coeffs00[f'{deg}'] for deg in degree]
#                 coeffs = torch.stack(coeffs, dim=0)
#                 if i==0:
#                     output00 = torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
#                 else:
#                     output00 += torch.einsum('ndsb,ndb->ns', coeffs, ops_out)  # N x S
                   
               
        if 2 in self.repsin and 2 in self.repsout:
            mat_diag_bias22 = self.diag_bias22.expand(-1,-1,inputs['2'].shape[3],inputs['2'].shape[3])
            mat_diag_bias22 = torch.mul(mat_diag_bias22, torch.eye(inputs['2'].shape[3], dtype=torch.float32, device=mat_diag_bias22.device))
            output22 = output22 + self.all_bias22 + mat_diag_bias22
        if 2 in self.repsin and 1 in self.repsout:
            output21 = output21 + self.bias21
        if 2 in self.repsin and 0 in self.repsout:
            output20 = output20 + self.bias20
        if 1 in self.repsin and 2 in self.repsout:
            output12 = output12 + self.bias12
        if 1 in self.repsin and 1 in self.repsout:
            output11 = output11 + self.bias11
        if 1 in self.repsin and 0 in self.repsout:
            output10 = output10 + self.bias10
        if 0 in self.repsin and 2 in self.repsout:
            output02 = output02 + self.bias02
        if 0 in self.repsin and 1 in self.repsout:
            output01 = output01 + self.bias01
        if 0 in self.repsin and 0 in self.repsout:
            output00 = output00 + self.bias00
       
        if 2 in self.repsout:
            if 2 in self.repsin and 1 in self.repsin and 0 in self.repsin:
                output2 = torch.cat((output22,output12,output02), dim=1)
            elif 2 in self.repsin and 1 in self.repsin:
                output2 = torch.cat((output22,output12), dim=1)
            elif 2 in self.repsin and 0 in self.repsin:
                output2 = torch.cat((output22,output02), dim=1)
            elif 1 in self.repsin and 0 in self.repsin:
                output2 = torch.cat((output12,output02), dim=1)
            elif 2 in self.repsin:
                output2 = output22
            elif 1 in self.repsin:
                output2 = output12
            elif 0 in self.repsin:
                output2 = output02
        if 1 in self.repsout:
            if 2 in self.repsin and 1 in self.repsin and 0 in self.repsin:
                output1 = torch.cat((output21,output11,output01), dim=1)
            elif 2 in self.repsin and 1 in self.repsin:
                output1 = torch.cat((output21,output11), dim=1)
            elif 2 in self.repsin and 0 in self.repsin:
                output1 = torch.cat((output21,output01), dim=1)
            elif 1 in self.repsin and 0 in self.repsin:
                output1 = torch.cat((output11,output01), dim=1)
            elif 2 in self.repsin:
                output1 = output21
            elif 1 in self.repsin:
                output1 = output11
            elif 0 in self.repsin:
                output1 = output01
        if 0 in self.repsout:
            if 2 in self.repsin and 1 in self.repsin and 0 in self.repsin:
                output0 = torch.cat((output20,output10,output00), dim=1)
            elif 2 in self.repsin and 1 in self.repsin:
                output0 = torch.cat((output20,output10), dim=1)
            elif 2 in self.repsin and 0 in self.repsin:
                output0 = torch.cat((output20,output00), dim=1)
            elif 1 in self.repsin and 0 in self.repsin:
                output0 = torch.cat((output10,output00), dim=1)
            elif 2 in self.repsin:
                output0 = output20
            elif 1 in self.repsin:
                output0 = output10
            elif 0 in self.repsin:
                output0 = output00
               
        if 2 in self.repsout and 1 in self.repsout and 0 in self.repsout:
            return {'2':output2, '1':output1, '0':output0}
        elif 2 in self.repsout and 1 in self.repsout:
            return {'2':output2, '1':output1}
        elif 2 in self.repsout and 0 in self.repsout:
            return {'2':output2, '0':output0}
        elif 1 in self.repsout and 0 in self.repsout:
            return {'1':output1, '0':output0}
        elif 2 in self.repsout:
            return {'2':output2}
        elif 1 in self.repsout:
            return {'1':output1}
        elif 0 in self.repsout:
            return {'0':output0}
        else:
            pass
        

class equi_multi_local_redops(torch.nn.Module):
    """equivariant nn layer."""

    def __init__(self, input_depth, output_depth, device, subgroups, repsin, repsout, hops):
        super(equi_multi_local_redops, self).__init__()
        self.input_depth = input_depth
        self.output_depth = output_depth
        self.repsin = repsin
        self.repsout = repsout
        self.hops = hops
        self.device = device
        self.basis_dimension_22 = 15
        self.basis_dimension_21 = 5
        self.basis_dimension_12 = 5
        self.basis_dimension_20 = 2
        self.basis_dimension_02 = 2
        self.basis_dimension_11 = 2
        self.basis_dimension_10 = 1
        self.basis_dimension_01 = 1
        self.basis_dimension_00 = 1
       
        if 2 in repsin and 2 in repsout:
            coeffs_values_22 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_22), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs22 = torch.nn.ParameterDict()
#             for sub in subgroups:  ## This isn't working on multiple gpus due to a pytorch bug, have to hard code subgroups for now https://github.com/pytorch/pytorch/issues/36035
#                 self.coeffs22[f'{sub}'] = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_1 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_0 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_2 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_2 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_3 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_4 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.coeffs22_5 = torch.nn.Parameter(coeffs_values_22, requires_grad=True)
            self.diag_bias22 = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
            self.all_bias22 = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
       
        if 2 in repsin and 1 in repsout:
            coeffs_values_21 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_21), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs21 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs21[f'{sub}'] = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_1 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_0 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_2 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_3 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_4 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.coeffs21_5 = torch.nn.Parameter(coeffs_values_21, requires_grad=True)
            self.bias21 = torch.nn.Parameter(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
           
        if 1 in repsin and 2 in repsout:
            coeffs_values_12 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_12), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs12 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs12[f'{sub}'] = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_1 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_0 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_2 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_3 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_4 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.coeffs12_5 = torch.nn.Parameter(coeffs_values_12, requires_grad=True)
            self.bias12 = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
           
        if 2 in repsin and 0 in repsout:
            coeffs_values_20 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_20), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs20 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs20[f'{sub}'] = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_1 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_0 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_2 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_3 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_4 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.coeffs20_5 = torch.nn.Parameter(coeffs_values_20, requires_grad=True)
            self.bias20 = torch.nn.Parameter(torch.zeros((1, output_depth), dtype=torch.float32), requires_grad=True)
           
        if 0 in repsin and 2 in repsout:
            coeffs_values_02 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_02), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs02 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs02[f'{sub}'] = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_1 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_0 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_2 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_3 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_4 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.coeffs02_5 = torch.nn.Parameter(coeffs_values_02, requires_grad=True)
            self.bias02 = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
           
        if 1 in repsin and 1 in repsout:
            coeffs_values_11 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_11), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs11 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs11[f'{sub}'] = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_1 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_0 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_2 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_3 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_4 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.coeffs11_5 = torch.nn.Parameter(coeffs_values_11, requires_grad=True)
            self.bias11 = torch.nn.Parameter(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
           
        if 1 in repsin and 0 in repsout:
            coeffs_values_10 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_10), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs10 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs10[f'{sub}'] = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_1 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_0 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_2 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_3 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_4 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.coeffs10_5 = torch.nn.Parameter(coeffs_values_10, requires_grad=True)
            self.bias10 = torch.nn.Parameter(torch.zeros((1, output_depth), dtype=torch.float32), requires_grad=True)
           
        if 0 in repsin and 1 in repsout:
            coeffs_values_01 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_01), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs01 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs01[f'{sub}'] = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_1 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_0 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_2 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_3 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_4 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.coeffs01_5 = torch.nn.Parameter(coeffs_values_01, requires_grad=True)
            self.bias01 = torch.nn.Parameter(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
           
        if 0 in repsin and 0 in repsout:
            coeffs_values_00 = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension_00), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth)))#.to(device)#.cuda()
#             self.coeffs00 = torch.nn.ParameterDict()
#             for sub in subgroups:
#                 self.coeffs00[f'{sub}'] = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_1 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_0 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_2 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_3 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_4 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.coeffs00_5 = torch.nn.Parameter(coeffs_values_00, requires_grad=True)
            self.bias00 = torch.nn.Parameter(torch.zeros((1, output_depth), dtype=torch.float32), requires_grad=True)


    def ops_2_to_2(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x k x D x m x m
        diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x k x D x m
        sum_diag_part = torch.sum(diag_part, dim=3, keepdim=True)  # N x k x D x 1
        sum_of_rows = torch.sum(inputs, dim=4)  # N x k x D x m
        sum_of_cols = torch.sum(inputs, dim=3)  # N x k x D x m
        sum_all = torch.sum(sum_of_rows, dim=3)  # N x k x D

        # op1 - (1234) - extract diag
        op1 = torch.diag_embed(diag_part)  # N x k x D x m x m

        # op2 - (1234) + (12)(34) - place sum of diag on diag
        op2 = torch.diag_embed(sum_diag_part.repeat(1, 1, 1, dim))  # N x k x D x m x m

        # op3 - (1234) + (123)(4) - place sum of row i on diag ii
        op3 = torch.diag_embed(sum_of_rows)  # N x k x D x m x m

        # op4 - (1234) + (124)(3) - place sum of col i on diag ii
        op4 = torch.diag_embed(sum_of_cols)  # N x k x D x m x m

        # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
        op5 = torch.diag_embed(torch.unsqueeze(sum_all, dim=3).repeat(1, 1, 1, dim))  # N x k x D x m x m

        # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
        op6 = torch.unsqueeze(sum_of_cols, dim=4).repeat(1, 1, 1, 1, dim)  # N x k x D x m x m

        # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
        op7 = torch.unsqueeze(sum_of_rows, dim=4).repeat(1, 1, 1, 1, dim)  # N x k x D x m x m

        # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
        op8 = torch.unsqueeze(sum_of_cols, dim=3).repeat(1, 1, 1, dim, 1)  # N x k x D x m x m

        # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
        op9 = torch.unsqueeze(sum_of_rows, dim=3).repeat(1, 1, 1, dim, 1)  # N x k x D x m x m

        # op10 - (1234) + (14)(23) - identity
        op10 = inputs  # N x k x D x m x m

        # op11 - (1234) + (13)(24) - transpose
        op11 = inputs.permute(0, 1, 2, 4, 3)  # N x k x D x m x m

        # op12 - (1234) + (234)(1) - place ii element in row i
        op12 = torch.unsqueeze(diag_part, dim=4).repeat(1, 1, 1, 1, dim)  # N x k x D x m x m

        # op13 - (1234) + (134)(2) - place ii element in col i
        op13 = torch.unsqueeze(diag_part, dim=3).repeat(1, 1, 1, dim, 1)  # N x k x D x m x m

        # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
        op14 = torch.unsqueeze(sum_diag_part, dim=4).repeat(1, 1, 1, dim, dim)   # N x k x D x m x m

        # op15 - sum of all ops - place sum of all entries in all entries
        op15 = torch.unsqueeze(torch.unsqueeze(sum_all, dim=3), dim=4).repeat(1, 1, 1, dim, dim)  # N x k x D x m x m

        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)
                op3 = torch.div(op3, float_dim)
                op4 = torch.div(op4, float_dim)
                op5 = torch.div(op5, float_dim**2)
                op6 = torch.div(op6, float_dim)
                op7 = torch.div(op7, float_dim)
                op8 = torch.div(op8, float_dim)
                op9 = torch.div(op9, float_dim)
                op14 = torch.div(op14, float_dim)
                op15 = torch.div(op15, float_dim**2)

        return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]


   
    def ops_2_to_1(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D x m x m
        diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
        sum_diag_part = torch.sum(diag_part, dim=3, keepdim=True)  # N x D x 1
        sum_of_rows = torch.sum(inputs, dim=4)  # N x D x m
        sum_of_cols = torch.sum(inputs, dim=3)  # N x D x m
        sum_all = torch.sum(sum_of_rows, dim=3)  # N x D

        # op1 - (123) - extract diag
        op1 = diag_part  # N x D x m

        # op2 - (123) + (12)(3) - tile sum of diag part
        op2 = sum_diag_part.repeat(1, 1, 1, dim)  # N x D x m

        # op3 - (123) + (13)(2) - place sum of row i in element i
        op3 = sum_of_rows  # N x D x m

        # op4 - (123) + (23)(1) - place sum of col i in element i
        op4 = sum_of_cols  # N x D x m

        # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
        op5 = torch.unsqueeze(sum_all, dim=3).repeat(1, 1, 1, dim)  # N x D x m


        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)
                op3 = torch.div(op3, float_dim)
                op4 = torch.div(op4, float_dim)
                op5 = torch.div(op5, float_dim ** 2)

        return [op1, op2, op3, op4, op5]

   
    def ops_2_to_0(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D x m x m
        diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
        sum_diag_part = torch.sum(diag_part, dim=3)  # N x D
        sum_of_rows = torch.sum(inputs, dim=4)  # N x D x m
        sum_all = torch.sum(sum_of_rows, dim=3)  # N x D

        # op1 -
        op1 = sum_diag_part  # N x D

        # op2 -
        op2 = sum_all  # N x D


        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
                op1 = torch.div(op1, float_dim)
                op2 = torch.div(op2, float_dim ** 2)

        return [op1, op2]



    def ops_0_to_2(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D
       
        # op1 - (123) - place on diag
        op1 = torch.unsqueeze(inputs, dim=3).repeat(1, 1, 1, dim)  # N x D x m
        op1 = torch.diag_embed(op1)  # N x D x m x m
#         op1 = F.pad(op1, (0,dim-inputs.shape[-1],0,dim-inputs.shape[-1]))

        # op2 - (123) + (12)(3) - tile in all entries
        op2 = torch.unsqueeze(torch.unsqueeze(inputs, dim=3), dim=4).repeat(1, 1, 1, dim, dim)  # N x D x m x m

        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
#                 op1 = torch.div(op1, float_dim)
                op2 = torch.div(op2, float_dim ** 2)

        return [op1, op2]



    def ops_1_to_2(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D x m
        sum_all = torch.sum(inputs, dim=3, keepdims=True)  # N x D x 1

        # op1 - (123) - place on diag
        op1 = torch.diag_embed(inputs)  # N x D x m x m

        # op2 - (123) + (12)(3) - tile sum on diag
        op2 = torch.diag_embed(sum_all.repeat(1, 1, 1, dim))  # N x D x m x m

        # op3 - (123) + (13)(2) - tile element i in row i
        op3 = torch.unsqueeze(inputs, dim=3).repeat(1, 1, 1, dim, 1)  # N x D x m x m

        # op4 - (123) + (23)(1) - tile element i in col i
        op4 = torch.unsqueeze(inputs, dim=4).repeat(1, 1, 1, 1, dim)  # N x D x m x m

        # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
        op5 = torch.unsqueeze(sum_all, dim=4).repeat(1, 1, 1, dim, dim)  # N x D x m x m

        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)
                op5 = torch.div(op5, float_dim)

        return [op1, op2, op3, op4, op5]

   
    def ops_1_to_1(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D x m
        sum_all = torch.sum(inputs, dim=3, keepdims=True)  # N x D x 1

        # op1 - (12) - identity
        op1 = inputs  # N x D x m

        # op2 - (1)(2) - tile sum of all
        op2 = sum_all.repeat(1, 1, 1, dim)  # N x D x m

        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)

        return [op1, op2]

   
    def ops_1_to_0(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D x m
        sum_all = torch.sum(inputs, dim=3)  # N x D

        # op1 - (12) - identity
        op1 = sum_all  # N x D

        if normalization is not None:
            float_dim = degrees.type(torch.float32)
            if normalization is 'inf':
                op1 = torch.div(op1, float_dim)

        return [op1]


    def ops_0_to_1(self, inputs, dim, degrees, normalization='inf', normalization_val=1.0):  # N x D

        # op1
        op1 = torch.unsqueeze(inputs, dim=3).repeat(1, 1, 1, dim)  # N x D x m
#         op1 = torch.diag_embed(inputs)  # N x D x m
#         op1 = F.pad(op1, (0,dim-inputs.shape[-1]))

#         if normalization is not None:
#             float_dim = degrees.type(torch.float32)
#             if normalization is 'inf':
#                 op1 = torch.div(op1, float_dim)

        return [op1]

   
    def get_coeffs22(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs22_1)
            elif deg == 2:
                coeffs.append(self.coeffs22_2)
            elif deg == 3:
                coeffs.append(self.coeffs22_3)
            elif deg == 4:
                coeffs.append(self.coeffs22_4)
            elif deg == 5:
                coeffs.append(self.coeffs22_5)
            else:
                coeffs.append(self.coeffs22_0)
        return coeffs
           
    def get_coeffs21(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs21_1)
            elif deg == 2:
                coeffs.append(self.coeffs21_2)
            elif deg == 3:
                coeffs.append(self.coeffs21_3)
            elif deg == 4:
                coeffs.append(self.coeffs21_4)
            elif deg == 5:
                coeffs.append(self.coeffs21_5)
            else:
                coeffs.append(self.coeffs21_0)
        return coeffs
   
    def get_coeffs12(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs12_1)
            elif deg == 2:
                coeffs.append(self.coeffs12_2)
            elif deg == 3:
                coeffs.append(self.coeffs12_3)
            elif deg == 4:
                coeffs.append(self.coeffs12_4)
            elif deg == 5:
                coeffs.append(self.coeffs12_5)
            else:
                coeffs.append(self.coeffs12_0)
        return coeffs
   
    def get_coeffs11(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs11_1)
            elif deg == 2:
                coeffs.append(self.coeffs11_2)
            elif deg == 3:
                coeffs.append(self.coeffs11_3)
            elif deg == 4:
                coeffs.append(self.coeffs11_4)
            elif deg == 5:
                coeffs.append(self.coeffs11_5)
            else:
                coeffs.append(self.coeffs11_0)
        return coeffs
   
    def get_coeffs10(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs10_1)
            elif deg == 2:
                coeffs.append(self.coeffs10_2)
            elif deg == 3:
                coeffs.append(self.coeffs10_3)
            elif deg == 4:
                coeffs.append(self.coeffs10_4)
            elif deg == 5:
                coeffs.append(self.coeffs10_5)
            else:
                coeffs.append(self.coeffs10_0)
        return coeffs
   
    def get_coeffs01(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs01_1)
            elif deg == 2:
                coeffs.append(self.coeffs01_2)
            elif deg == 3:
                coeffs.append(self.coeffs01_3)
            elif deg == 4:
                coeffs.append(self.coeffs01_4)
            elif deg == 5:
                coeffs.append(self.coeffs01_5)
            else:
                coeffs.append(self.coeffs01_0)
        return coeffs
   
    def get_coeffs20(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs20_1)
            elif deg == 2:
                coeffs.append(self.coeffs20_2)
            elif deg == 3:
                coeffs.append(self.coeffs20_3)
            elif deg == 4:
                coeffs.append(self.coeffs20_4)
            elif deg == 5:
                coeffs.append(self.coeffs20_5)
            else:
                coeffs.append(self.coeffs20_0)
        return coeffs
   
    def get_coeffs02(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs02_1)
            elif deg == 2:
                coeffs.append(self.coeffs02_2)
            elif deg == 3:
                coeffs.append(self.coeffs02_3)
            elif deg == 4:
                coeffs.append(self.coeffs02_4)
            elif deg == 5:
                coeffs.append(self.coeffs02_5)
            else:
                coeffs.append(self.coeffs02_0)
        return coeffs
   
    def get_coeffs00(self, degree):
        coeffs = []
        for deg in degree:
            if deg == 1:
                coeffs.append(self.coeffs00_1)
            elif deg == 2:
                coeffs.append(self.coeffs00_2)
            elif deg == 3:
                coeffs.append(self.coeffs00_3)
            elif deg == 4:
                coeffs.append(self.coeffs00_4)
            elif deg == 5:
                coeffs.append(self.coeffs00_5)
            else:
                coeffs.append(self.coeffs00_0)
        return coeffs
    
    def get_mask(self, A, i, hops=1):
        indices = (A==1).nonzero(as_tuple=False)
#         print(f'indices : {indices}')
#         if hops==1:
#             indices = indices[indices[:,1]==i]
#         else:
#             indices1hop = indices[indices[:,1]==i]
# #             print(f'indices1hop : {indices1hop.shape}')
#             indices2hop = []
# #             print(f'ind shape : {indices1hop[0].shape}')
#             for ind in indices1hop:
#                 inds = indices[(indices[:,0]==ind[0]) & (indices[:,1]==ind[2])]
#                 for val in inds:
#                     indices2hop.append(val)
#             if hops==2:
#                 indices = torch.stack(indices2hop, dim=0)
#             else:
#                 indices2hop = torch.stack(indices2hop, dim=0)
                
#         degree = torch.unique(indices[:,0], return_counts=True)[1]
        
        return indices#, degree
   

    def forward(self, inputs, A, normalization='inf'):
#         print(f'A device : {A.device}')
#         print(f'coeffs22 : {self.coeffs22}')
#         print(self.coeffs22)

        m = torch.tensor(inputs['2'].shape[3], dtype=torch.int32, device=A.device)  # extract dimension
#         print(f'm device : {m.device}')

        if 2 in self.repsin and 2 in self.repsout:
            inputs_temp = inputs['2']  # N x S x m x m
#             print(f'inputs sum : {torch.sum(inputs_temp)}')
            output22 = torch.zeros((inputs_temp.shape[0],self.output_depth,inputs_temp.shape[2],inputs_temp.shape[2]), dtype=inputs_temp.dtype, device=inputs_temp.device)
#             indices, degree = self.get_mask(A[degree], i)
            indices = (A==1).nonzero(as_tuple=False)
#             degrees = torch.unique(indices[:,0], return_counts=True)[1]
#             print(degrees)
            for batch in range(inputs_temp.shape[0]):
                index_b = indices[indices[:,0]==batch]
#                 print(index_b)
                degrees = torch.unique(index_b[:,1], return_counts=True)[1]
                for i in range(inputs_temp.shape[-1]):
                    index = index_b[index_b[:,1]==i]
                    index = index[:,2]
#                     print(index)
                    subgraph = inputs_temp[batch]
                    subgraph = subgraph[:,index,:]
                    subgraph = subgraph[:,:,index]
#                     print(subgraph.shape)
                    subgraph = torch.unsqueeze(torch.unsqueeze(subgraph,dim=0),dim=0)
#                     print(subgraph.shape)
                    m = torch.tensor(subgraph.shape[-1], dtype=torch.int32, device=A.device)  # extract dimension
                    ops_out = self.ops_2_to_2(inputs=subgraph, dim=m, degrees=m, normalization=normalization)
                    ops_out = torch.stack(ops_out, dim=3)
                    ops_out = torch.squeeze(ops_out)
#                     print(ops_out.shape)
                    coeffs_temp = self.get_coeffs22([subgraph.shape[-1]])[0]
#                     print(coeffs_temp.shape)
                    output_temp = torch.einsum('dsb,dbij->sij', coeffs_temp, ops_out)  # S x m x m
                    for c, idx in enumerate(index):
                        output22[batch, :, idx, index] += output_temp[:,c,:]
                degrees = torch.diag_embed(degrees)
                degrees = torch.unsqueeze(degrees, dim=0)
                degrees[degrees==0] = 2
                output22[batch,:,:,:] /= degrees
#             print(output22)
#             print(f'output22 sum : {torch.sum(output22)}')

        if 2 in self.repsin and 1 in self.repsout:
            inputs_temp = inputs['2']  # N x S x m x m
#             print(f'inputs sum : {torch.sum(inputs_temp)}')
            output21 = torch.zeros((inputs_temp.shape[0],self.output_depth,inputs_temp.shape[2]), dtype=inputs_temp.dtype, device=inputs_temp.device)
#             indices, degree = self.get_mask(A[degree], i)
            indices = (A==1).nonzero(as_tuple=False)
#             degrees = torch.unique(indices[:,0], return_counts=True)[1]
#             print(degrees)
            for batch in range(inputs_temp.shape[0]):
                index_b = indices[indices[:,0]==batch]
#                 print(index_b)
                degrees = torch.unique(index_b[:,1], return_counts=True)[1]
                for i in range(inputs_temp.shape[-1]):
                    index = index_b[index_b[:,1]==i]
                    index = index[:,2]
#                     print(index)
                    subgraph = inputs_temp[batch]
                    subgraph = subgraph[:,index,:]
                    subgraph = subgraph[:,:,index]
#                     print(subgraph.shape)
                    subgraph = torch.unsqueeze(torch.unsqueeze(subgraph,dim=0),dim=0)
#                     print(subgraph.shape)
                    m = torch.tensor(subgraph.shape[-1], dtype=torch.int32, device=A.device)  # extract dimension
                    ops_out = self.ops_2_to_1(inputs=subgraph, dim=m, degrees=m, normalization=normalization)
                    ops_out = torch.stack(ops_out, dim=3)
                    ops_out = torch.squeeze(ops_out)
#                     print(ops_out.shape)
                    coeffs_temp = self.get_coeffs21([subgraph.shape[-1]])[0]
                    output_temp = torch.einsum('dsb,dbi->si', coeffs_temp, ops_out)  # N x S x m
                    output21[batch, :, index] += output_temp
                degrees = torch.unsqueeze(degrees, dim=0)
                degrees[degrees==0] = 2
                output21[batch,:,:] /= degrees
#             print(f'output21 sum : {torch.sum(output21)}')
            
        if 2 in self.repsin and 0 in self.repsout:
            inputs_temp = inputs['2']  # N x S x m x m
            output20 = torch.zeros((inputs_temp.shape[0],self.output_depth), dtype=inputs_temp.dtype, device=inputs_temp.device)
#             print(f'inputs sum : {torch.sum(inputs_temp)}')
#             indices, degree = self.get_mask(A[degree], i)
            indices = (A==1).nonzero(as_tuple=False)
#             degrees = torch.unique(indices[:,0], return_counts=True)[1]
#             print(degrees)
            for batch in range(inputs_temp.shape[0]):
                index_b = indices[indices[:,0]==batch]
#                 print(index_b)
                degrees = torch.unique(index_b[:,1], return_counts=True)[1]
                for i in range(inputs_temp.shape[-1]):
                    index = index_b[index_b[:,1]==i]
                    index = index[:,2]
#                     print(index)
                    subgraph = inputs_temp[batch]
                    subgraph = subgraph[:,index,:]
                    subgraph = subgraph[:,:,index]
#                     print(subgraph.shape)
                    subgraph = torch.unsqueeze(torch.unsqueeze(subgraph,dim=0),dim=0)
#                     print(subgraph.shape)
                    m = torch.tensor(subgraph.shape[-1], dtype=torch.int32, device=A.device)  # extract dimension
                    ops_out = self.ops_2_to_0(inputs=subgraph, dim=m, degrees=m, normalization=normalization)
                    ops_out = torch.stack(ops_out, dim=3)
                    ops_out = torch.squeeze(ops_out)
#                     print(ops_out.shape)
                    coeffs_temp = self.get_coeffs20([subgraph.shape[-1]])[0]
                    output_temp = torch.einsum('dsb,db->s', coeffs_temp, ops_out)  # N x S x m
                    output20[batch, :] += output_temp
            output20 /= torch.tensor(inputs_temp.shape[-1], dtype=torch.int32, device=inputs_temp.device)
            
#             print(f'output20 sum : {torch.sum(output20)}')
            
        if 1 in self.repsin and 2 in self.repsout:
            inputs_temp = inputs['1']  # N x S x m
#             print(f'inputs sum : {torch.sum(inputs_temp)}')
            output12 = torch.zeros((inputs_temp.shape[0],self.output_depth,inputs_temp.shape[2],inputs_temp.shape[2]), dtype=inputs_temp.dtype, device=inputs_temp.device)
#             indices, degree = self.get_mask(A[degree], i)
            indices = (A==1).nonzero(as_tuple=False)
#             degrees = torch.unique(indices[:,0], return_counts=True)[1]
#             print(degrees)
            for batch in range(inputs_temp.shape[0]):
                index_b = indices[indices[:,0]==batch]
#                 print(index_b)
                degrees = torch.unique(index_b[:,1], return_counts=True)[1]
                for i in range(inputs_temp.shape[-1]):
                    index = index_b[index_b[:,1]==i]
                    index = index[:,2]
#                     print(index)
                    subgraph = inputs_temp[batch]
                    subgraph = subgraph[:,index]
#                     print(subgraph.shape)
                    subgraph = torch.unsqueeze(torch.unsqueeze(subgraph,dim=0),dim=0)
#                     print(subgraph.shape)
                    m = torch.tensor(subgraph.shape[-1], dtype=torch.int32, device=A.device)  # extract dimension
                    ops_out = self.ops_1_to_2(inputs=subgraph, dim=m, degrees=m, normalization=normalization)
                    ops_out = torch.stack(ops_out, dim=3)
                    ops_out = torch.squeeze(ops_out)
#                     print(ops_out.shape)
                    coeffs_temp = self.get_coeffs12([subgraph.shape[-1]])[0]
#                     print(coeffs_temp.shape)
                    output_temp = torch.einsum('dsb,dbij->sij', coeffs_temp, ops_out)  # S x m x m
                    for c, idx in enumerate(index):
                        output12[batch, :, idx, index] += output_temp[:,c,:]
                degrees = torch.diag_embed(degrees)
                degrees = torch.unsqueeze(degrees, dim=0)
                degrees[degrees==0] = 2
                output12[batch,:,:,:] /= degrees
#             print(output22)
#             print(f'output12 sum : {torch.sum(output12)}')


        if 1 in self.repsin and 1 in self.repsout:
            inputs_temp = inputs['1']  # N x S x m
#             print(f'inputs sum : {torch.sum(inputs_temp)}')
            output11 = torch.zeros((inputs_temp.shape[0],self.output_depth,inputs_temp.shape[2]), dtype=inputs_temp.dtype, device=inputs_temp.device)
#             indices, degree = self.get_mask(A[degree], i)
            indices = (A==1).nonzero(as_tuple=False)
#             degrees = torch.unique(indices[:,0], return_counts=True)[1]
#             print(degrees)
            for batch in range(inputs_temp.shape[0]):
                index_b = indices[indices[:,0]==batch]
#                 print(index_b)
                degrees = torch.unique(index_b[:,1], return_counts=True)[1]
                for i in range(inputs_temp.shape[-1]):
                    index = index_b[index_b[:,1]==i]
                    index = index[:,2]
#                     print(index)
                    subgraph = inputs_temp[batch]
                    subgraph = subgraph[:,index]
#                     print(subgraph.shape)
                    subgraph = torch.unsqueeze(torch.unsqueeze(subgraph,dim=0),dim=0)
#                     print(subgraph.shape)
                    m = torch.tensor(subgraph.shape[-1], dtype=torch.int32, device=A.device)  # extract dimension
                    ops_out = self.ops_1_to_1(inputs=subgraph, dim=m, degrees=m, normalization=normalization)
                    ops_out = torch.stack(ops_out, dim=3)
                    ops_out = torch.squeeze(ops_out)
#                     print(ops_out.shape)
                    coeffs_temp = self.get_coeffs11([subgraph.shape[-1]])[0]
#                     print(coeffs_temp.shape)
                    output_temp = torch.einsum('dsb,dbi->si', coeffs_temp, ops_out)  # S x m
                    output11[batch, :, index] += output_temp
                degrees = torch.unsqueeze(degrees, dim=0)
                degrees[degrees==0] = 2
                output11[batch,:,:] /= degrees
#             print(output22)
#             print(f'output11 sum : {torch.sum(output11)}')
            
        if 1 in self.repsin and 0 in self.repsout:
            inputs_temp = inputs['1']  # N x S x m x m
            output10 = torch.zeros((inputs_temp.shape[0],self.output_depth), dtype=inputs_temp.dtype, device=inputs_temp.device)
#             print(f'inputs sum : {torch.sum(inputs_temp)}')
#             indices, degree = self.get_mask(A[degree], i)
            indices = (A==1).nonzero(as_tuple=False)
#             degrees = torch.unique(indices[:,0], return_counts=True)[1]
#             print(degrees)
            for batch in range(inputs_temp.shape[0]):
                index_b = indices[indices[:,0]==batch]
#                 print(index_b)
                degrees = torch.unique(index_b[:,1], return_counts=True)[1]
                for i in range(A.shape[-1]):
                    index = index_b[index_b[:,1]==i]
                    index = index[:,2]
#                     print(index)
                    subgraph = inputs_temp[batch]
                    subgraph = subgraph[:,index]
#                     print(subgraph.shape)
                    subgraph = torch.unsqueeze(torch.unsqueeze(subgraph,dim=0),dim=0)
#                     print(subgraph.shape)
                    m = torch.tensor(subgraph.shape[-1], dtype=torch.int32, device=A.device)  # extract dimension
                    ops_out = self.ops_1_to_0(inputs=subgraph, dim=m, degrees=m, normalization=normalization)
                    ops_out = torch.stack(ops_out, dim=3)
                    ops_out = torch.squeeze(torch.squeeze(ops_out, dim=0), dim=0)
#                     print(ops_out.shape)
                    coeffs_temp = self.get_coeffs10([subgraph.shape[-1]])[0]
                    output_temp = torch.einsum('dsb,db->s', coeffs_temp, ops_out)  # N x S x m
                    output10[batch, :] += output_temp
            output10 /= torch.tensor(inputs_temp.shape[-1], dtype=torch.int32, device=inputs_temp.device)
            
#             print(f'output10 sum : {torch.sum(output10)}')

            
        if 0 in self.repsin and 2 in self.repsout:
            inputs_temp = inputs['0']  # N x S
#             print(f'inputs sum : {torch.sum(inputs_temp)}')
            output02 = torch.zeros((inputs_temp.shape[0],self.output_depth,A.shape[-1],A.shape[-1]), dtype=inputs_temp.dtype, device=inputs_temp.device)
#             indices, degree = self.get_mask(A[degree], i)
            indices = (A==1).nonzero(as_tuple=False)
#             degrees = torch.unique(indices[:,0], return_counts=True)[1]
#             print(degrees)
            for batch in range(inputs_temp.shape[0]):
                index_b = indices[indices[:,0]==batch]
#                 print(index_b)
                degrees = torch.unique(index_b[:,1], return_counts=True)[1]
                for i in range(A.shape[-1]):
                    index = index_b[index_b[:,1]==i]
                    index = index[:,2]
#                     print(index)
                    subgraph = inputs_temp[batch]
#                     print(subgraph.shape)
                    subgraph = torch.unsqueeze(torch.unsqueeze(subgraph,dim=0),dim=0)
#                     print(subgraph.shape)
                    m = torch.tensor(len(index), dtype=torch.int32, device=A.device)  # extract dimension
                    ops_out = self.ops_0_to_2(inputs=subgraph, dim=m, degrees=m, normalization=normalization)
                    ops_out = torch.stack(ops_out, dim=3)
                    ops_out = torch.squeeze(ops_out)
#                     print(ops_out.shape)
                    coeffs_temp = self.get_coeffs02([len(index)])[0]
#                     print(coeffs_temp.shape)
                    output_temp = torch.einsum('dsb,dbij->sij', coeffs_temp, ops_out)  # S x m x m
                    for c, idx in enumerate(index):
                        output02[batch, :, idx, index] += output_temp[:,c,:]
                degrees = torch.diag_embed(degrees)
                degrees = torch.unsqueeze(degrees, dim=0)
                degrees[degrees==0] = 2
                output02[batch,:,:,:] /= degrees
#             print(output22)
#             print(f'output02 sum : {torch.sum(output02)}')

        if 0 in self.repsin and 1 in self.repsout:
            inputs_temp = inputs['0']  # N x S
#             print(f'inputs sum : {torch.sum(inputs_temp)}')
            output01 = torch.zeros((inputs_temp.shape[0],self.output_depth,A.shape[-1]), dtype=inputs_temp.dtype, device=inputs_temp.device)
#             indices, degree = self.get_mask(A[degree], i)
            indices = (A==1).nonzero(as_tuple=False)
#             degrees = torch.unique(indices[:,0], return_counts=True)[1]
#             print(degrees)
            for batch in range(inputs_temp.shape[0]):
                index_b = indices[indices[:,0]==batch]
#                 print(index_b)
                degrees = torch.unique(index_b[:,1], return_counts=True)[1]
                for i in range(A.shape[-1]):
                    index = index_b[index_b[:,1]==i]
                    index = index[:,2]
#                     print(index)
                    subgraph = inputs_temp[batch]
#                     print(subgraph.shape)
                    subgraph = torch.unsqueeze(torch.unsqueeze(subgraph,dim=0),dim=0)
#                     print(subgraph.shape)
                    m = torch.tensor(len(index), dtype=torch.int32, device=A.device)  # extract dimension
                    ops_out = self.ops_0_to_1(inputs=subgraph, dim=m, degrees=m, normalization=normalization)
                    ops_out = torch.stack(ops_out, dim=3)
                    ops_out = torch.squeeze(torch.squeeze(ops_out, dim=0), dim=0)
#                     print(ops_out.shape)
                    coeffs_temp = self.get_coeffs01([len(index)])[0]
#                     print(coeffs_temp.shape)
                    output_temp = torch.einsum('dsb,dbi->si', coeffs_temp, ops_out)  # S x m
                    output01[batch, :, index] += output_temp
                degrees = torch.unsqueeze(degrees, dim=0)
                degrees[degrees==0] = 2
                output01[batch,:,:] /= degrees
#             print(output22)
#             print(f'output01 sum : {torch.sum(output01)}')
            
        if 0 in self.repsin and 0 in self.repsout:
            inputs_temp = inputs['0']  # N x S
#             print(f'inputs sum : {torch.sum(inputs_temp)}')
            output00 = torch.zeros((inputs_temp.shape[0],self.output_depth), dtype=inputs_temp.dtype, device=inputs_temp.device)
#             indices, degree = self.get_mask(A[degree], i)
            indices = (A==1).nonzero(as_tuple=False)
#             degrees = torch.unique(indices[:,0], return_counts=True)[1]
#             print(degrees)
            for batch in range(inputs_temp.shape[0]):
                index_b = indices[indices[:,0]==batch]
#                 print(index_b)
                degrees = torch.unique(index_b[:,1], return_counts=True)[1]
                for i in range(A.shape[-1]):
                    index = index_b[index_b[:,1]==i]
                    index = index[:,2]
#                     print(index)
                    ops_out = inputs_temp[batch]
                    ops_out = torch.unsqueeze(ops_out, dim=1)
#                     print(ops_out.shape)
                    coeffs_temp = self.get_coeffs00([len(index)])[0]
#                     print(coeffs_temp.shape)
                    output_temp = torch.einsum('dsb,db->s', coeffs_temp, ops_out)  # S x m
                    output00[batch, :] += output_temp
                output00 /= torch.tensor(A.shape[-1], dtype=torch.int32, device=inputs_temp.device)
#             print(output22)
#             print(f'output00 sum : {torch.sum(output00)}')
                   
               
        if 2 in self.repsin and 2 in self.repsout:
            mat_diag_bias22 = self.diag_bias22.expand(-1,-1,inputs['2'].shape[3],inputs['2'].shape[3])
            mat_diag_bias22 = torch.mul(mat_diag_bias22, torch.eye(inputs['2'].shape[3], dtype=torch.float32, device=mat_diag_bias22.device))
            output22 = output22 + self.all_bias22 + mat_diag_bias22
        if 2 in self.repsin and 1 in self.repsout:
            output21 = output21 + self.bias21
        if 2 in self.repsin and 0 in self.repsout:
            output20 = output20 + self.bias20
        if 1 in self.repsin and 2 in self.repsout:
            output12 = output12 + self.bias12
        if 1 in self.repsin and 1 in self.repsout:
            output11 = output11 + self.bias11
        if 1 in self.repsin and 0 in self.repsout:
            output10 = output10 + self.bias10
        if 0 in self.repsin and 2 in self.repsout:
            output02 = output02 + self.bias02
        if 0 in self.repsin and 1 in self.repsout:
            output01 = output01 + self.bias01
        if 0 in self.repsin and 0 in self.repsout:
            output00 = output00 + self.bias00
       
        if 2 in self.repsout:
            if 2 in self.repsin and 1 in self.repsin and 0 in self.repsin:
                output2 = torch.cat((output22,output12,output02), dim=1)
            elif 2 in self.repsin and 1 in self.repsin:
                output2 = torch.cat((output22,output12), dim=1)
            elif 2 in self.repsin and 0 in self.repsin:
                output2 = torch.cat((output22,output02), dim=1)
            elif 1 in self.repsin and 0 in self.repsin:
                output2 = torch.cat((output12,output02), dim=1)
            elif 2 in self.repsin:
                output2 = output22
            elif 1 in self.repsin:
                output2 = output12
            elif 0 in self.repsin:
                output2 = output02
        if 1 in self.repsout:
            if 2 in self.repsin and 1 in self.repsin and 0 in self.repsin:
                output1 = torch.cat((output21,output11,output01), dim=1)
            elif 2 in self.repsin and 1 in self.repsin:
                output1 = torch.cat((output21,output11), dim=1)
            elif 2 in self.repsin and 0 in self.repsin:
                output1 = torch.cat((output21,output01), dim=1)
            elif 1 in self.repsin and 0 in self.repsin:
                output1 = torch.cat((output11,output01), dim=1)
            elif 2 in self.repsin:
                output1 = output21
            elif 1 in self.repsin:
                output1 = output11
            elif 0 in self.repsin:
                output1 = output01
        if 0 in self.repsout:
            if 2 in self.repsin and 1 in self.repsin and 0 in self.repsin:
                output0 = torch.cat((output20,output10,output00), dim=1)
            elif 2 in self.repsin and 1 in self.repsin:
                output0 = torch.cat((output20,output10), dim=1)
            elif 2 in self.repsin and 0 in self.repsin:
                output0 = torch.cat((output20,output00), dim=1)
            elif 1 in self.repsin and 0 in self.repsin:
                output0 = torch.cat((output10,output00), dim=1)
            elif 2 in self.repsin:
                output0 = output20
            elif 1 in self.repsin:
                output0 = output10
            elif 0 in self.repsin:
                output0 = output00
               
        if 2 in self.repsout and 1 in self.repsout and 0 in self.repsout:
            return {'2':output2, '1':output1, '0':output0}
        elif 2 in self.repsout and 1 in self.repsout:
            return {'2':output2, '1':output1}
        elif 2 in self.repsout and 0 in self.repsout:
            return {'2':output2, '0':output0}
        elif 1 in self.repsout and 0 in self.repsout:
            return {'1':output1, '0':output0}
        elif 2 in self.repsout:
            return {'2':output2}
        elif 1 in self.repsout:
            return {'1':output1}
        elif 0 in self.repsout:
            return {'0':output0}
        else:
            pass
        

class equi_2_to_2_local(torch.nn.Module):
    """equivariant nn layer."""

    def __init__(self, input_depth, output_depth, device, subgroups):
        super(equi_2_to_2_local, self).__init__()
        self.basis_dimension = 15
        self.device = device
#         self.coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
        self.coeffs_values = torch.mul(torch.randn(size=(input_depth, output_depth, self.basis_dimension), dtype=torch.float32), torch.sqrt(torch.tensor([2.]) / (input_depth + output_depth))).to(device)#.cuda()

        self.coeffs = {}
        for sub in subgroups:
            self.coeffs[f'{sub}'] = torch.nn.Parameter(self.coeffs_values, requires_grad=True)
#         self.coeffs = torch.nn.Parameter(self.coeffs_values, requires_grad=True)
        self.diag_bias = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float64), requires_grad=True)
        self.all_bias = torch.nn.Parameter(torch.zeros((1, output_depth, 1, 1), dtype=torch.float64), requires_grad=True)
        
    def ops_2_to_2(self, inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
#         print(f'input shape : {inputs.shape}')
        diag_part = torch.diagonal(inputs, dim1=-2, dim2=-1)   # N x D x m
#         print(f'diag_part shape : {diag_part.shape}')
        sum_diag_part = torch.sum(diag_part, dim=2, keepdim=True)  # N x D x 1
#         print(f'sum_diag_part shape : {sum_diag_part.shape}')
        sum_of_rows = torch.sum(inputs, dim=3)  # N x D x m
#         print(f'sum_of_rows shape : {sum_of_rows.shape}')
        sum_of_cols = torch.sum(inputs, dim=2)  # N x D x m
#         print(f'sum_of_cols shape : {sum_of_cols.shape}')
        sum_all = torch.sum(sum_of_rows, dim=2)  # N x D
#         print(f'sum_all shape : {sum_all.shape}')

        # op1 - (1234) - extract diag
        op1 = torch.diag_embed(diag_part)  # N x D x m x m

        # op2 - (1234) + (12)(34) - place sum of diag on diag
        op2 = torch.diag_embed(sum_diag_part.repeat(1, 1, dim))  # N x D x m x m

        # op3 - (1234) + (123)(4) - place sum of row i on diag ii
        op3 = torch.diag_embed(sum_of_rows)  # N x D x m x m

        # op4 - (1234) + (124)(3) - place sum of col i on diag ii
        op4 = torch.diag_embed(sum_of_cols)  # N x D x m x m

        # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
        op5 = torch.diag_embed(torch.unsqueeze(sum_all, dim=2).repeat(1, 1, dim))  # N x D x m x m

        # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
        op6 = torch.unsqueeze(sum_of_cols, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

        # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
        op7 = torch.unsqueeze(sum_of_rows, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

        # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
        op8 = torch.unsqueeze(sum_of_cols, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

        # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
        op9 = torch.unsqueeze(sum_of_rows, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

        # op10 - (1234) + (14)(23) - identity
        op10 = inputs  # N x D x m x m

        # op11 - (1234) + (13)(24) - transpose
        op11 = inputs.permute(0, 1, 3, 2)  # N x D x m x m

        # op12 - (1234) + (234)(1) - place ii element in row i
        op12 = torch.unsqueeze(diag_part, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

        # op13 - (1234) + (134)(2) - place ii element in col i
        op13 = torch.unsqueeze(diag_part, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

        # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
        op14 = torch.unsqueeze(sum_diag_part, dim=3).repeat(1, 1, dim, dim)   # N x D x m x m

        # op15 - sum of all ops - place sum of all entries in all entries
        op15 = torch.unsqueeze(torch.unsqueeze(sum_all, dim=2), dim=3).repeat(1, 1, dim, dim)  # N x D x m x m

        if normalization is not None:
            float_dim = dim.type(torch.FloatTensor)
            if normalization is 'inf':
                op2 = torch.div(op2, float_dim)
                op3 = torch.div(op3, float_dim)
                op4 = torch.div(op4, float_dim)
                op5 = torch.div(op5, float_dim**2)
                op6 = torch.div(op6, float_dim)
                op7 = torch.div(op7, float_dim)
                op8 = torch.div(op8, float_dim)
                op9 = torch.div(op9, float_dim)
                op14 = torch.div(op14, float_dim)
                op15 = torch.div(op15, float_dim**2)

        return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]
    
    def get_mask(self, A, i):
        indices = (A==1).nonzero(as_tuple=False)
        indices = indices[indices[:,1]==i]
#         print(f'indices : {indices}')

        mask = torch.ones(A.shape).to(self.device)
        for index in indices:
            mask[index[0],index[1],index[2]] = 0

        mask = torch.min(mask, torch.transpose(mask,-2,-1))
        mask = mask.bool()
        mask = torch.unsqueeze(mask, dim=1)
        
        degree = torch.unique(indices[:,0], return_counts=True)[1]
#         print(f'degree : {degree}')
        
        return mask, degree
        

    def forward(self, inputs, A, normalization='inf'):
        
        m = torch.tensor(inputs.shape[3], dtype=torch.int32, device=self.device)  # extract dimension

        for i in range(inputs.shape[-1]):
            inputs_temp = torch.clone(inputs)
            mask, degree = self.get_mask(A, i)
            inputs_temp.masked_fill_(mask, 0)
            ops_out = self.ops_2_to_2(inputs=inputs_temp, dim=m, normalization=normalization)
            ops_out = torch.stack(ops_out, dim=2)
            coeffs = [self.coeffs[f'{deg}'].double() for deg in degree]
            coeffs = torch.stack(coeffs, dim=0)
            if i==0:
                output = torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
            else:
                output += torch.einsum('ndsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m

        # bias
        mat_diag_bias = self.diag_bias.expand(-1,-1,inputs.shape[3],inputs.shape[3])
        mat_diag_bias = torch.mul(mat_diag_bias, torch.eye(inputs.shape[3], dtype=torch.double, device=self.device))
        output = output + self.all_bias + mat_diag_bias

        return output


# def equi_2_to_2(name, input_depth, output_depth, inputs, normalization='inf', normalization_val=1.0):
#     '''
#     :param name: name of layer
#     :param input_depth: D
#     :param output_depth: S
#     :param inputs: N x D x m x m tensor
#     :return: output: N x S x m x m tensor
#     '''
#     basis_dimension = 15

#     # initialization values for variables
#     coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
# #     coeffs_values = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32), tf.sqrt(2. / tf.to_float(input_depth + output_depth)))
#     #coeffs_values = tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32)
#     # define variables
#     coeffs = torch.autograd.Variable(coeffs_values, requires_grad=True)
# #     coeffs = tf.get_variable('coeffs', initializer=coeffs_values)

#     m = inputs.shape[3].type(torch.IntTensor)  # extract dimension
# #     m = tf.to_int32(tf.shape(inputs)[3])  # extract dimension

#     ops_out = ops_2_to_2(inputs, m, normalization=normalization)
#     ops_out = torch.stack(ops_out, dim=2)
# #     ops_out = tf.stack(ops_out, axis=2)

#     output = torch.einsum('dsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
# #     output = tf.einsum('dsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m

#     # bias
#     diag_bias = torch.autograd.Variable(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
# #     diag_bias = tf.get_variable('diag_bias', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
#     all_bias = torch.autograd.Variable(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
# #     all_bias = tf.get_variable('all_bias', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
#     mat_diag_bias = torch.matmul(torch.unsqueeze(torch.unsqueeze(torch.eye(inputs.shape[3].type(torch.IntTensor)), 0), 0), diag_bias)
# #     mat_diag_bias = tf.multiply(tf.expand_dims(tf.expand_dims(tf.eye(tf.to_int32(tf.shape(inputs)[3])), 0), 0), diag_bias)
#     output = output + all_bias + mat_diag_bias

#     return output


# def equi_2_to_1(name, input_depth, output_depth, inputs, normalization='inf', normalization_val=1.0):
#     '''
#     :param name: name of layer
#     :param input_depth: D
#     :param output_depth: S
#     :param inputs: N x D x m x m tensor
#     :return: output: N x S x m tensor
#     '''
#     basis_dimension = 5
# #     with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

#     # initialization values for variables
#     coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
# #     coeffs_values = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32), tf.sqrt(2. / tf.to_float(input_depth + output_depth)))
#     #coeffs_values = tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32)
#     # define variables
#     coeffs = torch.autograd.Variable(coeffs_values, requires_grad=True)
# #     coeffs = tf.get_variable('coeffs', initializer=coeffs_values)

#     m = inputs.shape[3].type(torch.IntTensor)  # extract dimension
# #     m = tf.to_int32(tf.shape(inputs)[3])  # extract dimension

#     ops_out = ops_2_to_1(inputs, m, normalization=normalization)
#     ops_out = torch.stack(ops_out, dim=2)
# #     ops_out = tf.stack(ops_out, axis=2)  # N x D x B x m

#     output = torch.einsum('dsb,ndbi->nsi', coeffs, ops_out)  # N x S x m x m
# #     output = tf.einsum('dsb,ndbi->nsi', coeffs, ops_out)  # N x S x m

#     # bias
#     bias = torch.autograd.Variable(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
# #     bias = tf.get_variable('bias', initializer=tf.zeros([1, output_depth, 1], dtype=tf.float32))
#     output = output + bias

#     return output


# def equi_1_to_2(name, input_depth, output_depth, inputs, normalization='inf', normalization_val=1.0):
#     '''
#     :param name: name of layer
#     :param input_depth: D
#     :param output_depth: S
#     :param inputs: N x D x m tensor
#     :return: output: N x S x m x m tensor
#     '''
#     basis_dimension = 5
# #     with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

#     # initialization values for variables
#     coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
# #     coeffs_values = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32), tf.sqrt(2. / tf.to_float(input_depth + output_depth)))
#     #coeffs_values = tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32)
#     # define variables
#     coeffs = torch.autograd.Variable(coeffs_values, requires_grad=True)
# #     coeffs = tf.get_variable('coeffs', initializer=coeffs_values)

#     m = inputs.shape[3].type(torch.IntTensor)  # extract dimension
# #     m = tf.to_int32(tf.shape(inputs)[2])  # extract dimension

#     ops_out = ops_1_to_2(inputs, m, normalization=normalization)
#     ops_out = torch.stack(ops_out, dim=2)
# #     ops_out = tf.stack(ops_out, axis=2)  # N x D x B x m x m

#     output = torch.einsum('dsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
# #     output = tf.einsum('dsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m

#     # bias
#     bias = torch.autograd.Variable(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
# #     bias = tf.get_variable('bias', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
#     output = output + bias

#     return output


# def equi_1_to_1(name, input_depth, output_depth, inputs, normalization='inf', normalization_val=1.0):
#     '''
#     :param name: name of layer
#     :param input_depth: D
#     :param output_depth: S
#     :param inputs: N x D x m tensor
#     :return: output: N x S x m tensor
#     '''
#     basis_dimension = 2
# #     with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

#     # initialization values for variables
#     coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
# #     coeffs_values = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32), tf.sqrt(2. / tf.to_float(input_depth + output_depth)))
#     #coeffs_values = tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32)
#     # define variables
#     coeffs = torch.autograd.Variable(coeffs_values, requires_grad=True)
# #     coeffs = tf.get_variable('coeffs', initializer=coeffs_values)

#     m = inputs.shape[3].type(torch.IntTensor)  # extract dimension
# #     m = tf.to_int32(tf.shape(inputs)[2])  # extract dimension

#     ops_out = ops_1_to_1(inputs, m, normalization=normalization)
#     ops_out = torch.stack(ops_out, dim=2)
# #     ops_out = tf.stack(ops_out, axis=2)  # N x D x B x m

#     output = torch.einsum('dsb,ndbi->nsi', coeffs, ops_out)  # N x S x m x m
# #     output = tf.einsum('dsb,ndbi->nsi', coeffs, ops_out)  # N x S x m

#     # bias
#     bias = torch.autograd.Variable(torch.zeros((1, output_depth, 1), dtype=torch.float32), requires_grad=True)
# #     bias = tf.get_variable('bias', initializer=tf.zeros([1, output_depth, 1], dtype=tf.float32))
#     output = output + bias

#     return output


# def equi_basic(name, input_depth, output_depth, inputs):
#     '''
#     :param name: name of layer
#     :param input_depth: D
#     :param output_depth: S
#     :param inputs: N x D x m x m tensor
#     :return: output: N x S x m x m tensor
#     '''
#     basis_dimension = 4
# #     with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

#     # initialization values for variables
#     coeffs_values = torch.matmul(torch.randn(size=(input_depth, output_depth, basis_dimension), dtype=torch.float32), torch.sqrt(2. / (input_depth + output_depth).type(torch.FloatTensor)))
# #     coeffs_values = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32), tf.sqrt(2. / tf.to_float(input_depth + output_depth)))
#     #coeffs_values = tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32)
#     # define variables
#     coeffs = torch.autograd.Variable(coeffs_values, requires_grad=True)
# #     coeffs = tf.get_variable('coeffs', initializer=coeffs_values)

#     m = inputs.shape[3].type(torch.IntTensor)  # extract dimension
# #     m = tf.to_int32(tf.shape(inputs)[3])  # extract dimension
#     float_dim = m.type(torch.FloatTensor)
# #     float_dim = tf.to_float(m)


#     # apply ops
#     ops_out = []
#     # w1 - identity
#     ops_out.append(inputs)
#     # w2 - sum cols
#     sum_of_cols = torch.divide(torch.sum(inputs, dim=2), float_dim)  # N x D x m
# #     sum_of_cols = tf.divide(tf.reduce_sum(inputs, axis=2), float_dim)  # N x D x m
#     ops_out.append(torch.unsqueeze(sum_of_cols, dim=2).repeat(1, 1, m, 1))  # N x D x m x m
# #     ops_out.append(tf.tile(tf.expand_dims(sum_of_cols, axis=2), [1, 1, m, 1]))  # N x D x m x m
#     # w3 - sum rows
#     sum_of_rows = torch.divide(torch.sum(inputs, dim=3), float_dim)  # N x D x m
# #     sum_of_rows = tf.divide(tf.reduce_sum(inputs, axis=3), float_dim)  # N x D x m
#     ops_out.append(torch.unsqueeze(sum_of_rows, dim=3).repeat(1, 1, 1, m))  # N x D x m x m
# #     ops_out.append(tf.tile(tf.expand_dims(sum_of_rows, axis=3), [1, 1, 1, m]))  # N x D x m x m
#     # w4 - sum all
#     sum_all = torch.divide(torch.sum(sum_of_rows, dim=2), torch.square(float_dim))  # N x D
# #     sum_all = tf.divide(tf.reduce_sum(sum_of_rows, axis=2), tf.square(float_dim))  # N x D
#     ops_out.append(torch.unsqueeze(torch.unsqueeze(sum_all, dim=2), dim=3).repeat(1, 1, m, m))  # N x D x m x m
# #     ops_out.append(tf.tile(tf.expand_dims(tf.expand_dims(sum_all, axis=2), axis=3), [1, 1, m, m]))  # N x D x m x m

#     ops_out = torch.stack(ops_out, dim=2)
# #     ops_out = tf.stack(ops_out, axis=2)
#     output = torch.einsum('dsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m
# #     output = tf.einsum('dsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m

#     # bias
#     bias = torch.autograd.Variable(torch.zeros((1, output_depth, 1, 1), dtype=torch.float32), requires_grad=True)
# #     bias = tf.get_variable('bias', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
#     output = output + bias

#     return output


# def ops_2_to_2(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
#     diag_part = torch.diagonal(inputs)   # N x D x m
#     sum_diag_part = torch.sum(diag_part, dim=2, keepdim=True)  # N x D x 1
#     sum_of_rows = torch.sum(inputs, dim=3)  # N x D x m
#     sum_of_cols = torch.sum(inputs, dim=2)  # N x D x m
#     sum_all = torch.sum(sum_of_rows, dim=2)  # N x D

#     # op1 - (1234) - extract diag
#     op1 = torch.diagonal(diag_part)  # N x D x m x m

#     # op2 - (1234) + (12)(34) - place sum of diag on diag
#     op2 = torch.diagonal(sum_diag_part.repeat(1, 1, dim))  # N x D x m x m

#     # op3 - (1234) + (123)(4) - place sum of row i on diag ii
#     op3 = torch.diagonal(sum_of_rows)  # N x D x m x m

#     # op4 - (1234) + (124)(3) - place sum of col i on diag ii
#     op4 = torch.diagonal(sum_of_cols)  # N x D x m x m

#     # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
#     op5 = torch.diagonal(torch.unsqueeze(sum_all, dim=2).repeat(1, 1, dim))  # N x D x m x m

#     # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
#     op6 = torch.unsqueeze(sum_of_cols, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

#     # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
#     op7 = torch.unsqueeze(sum_of_rows, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

#     # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
#     op8 = torch.unsqueeze(sum_of_cols, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

#     # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
#     op9 = torch.unsqueeze(sum_of_rows, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

#     # op10 - (1234) + (14)(23) - identity
#     op10 = inputs  # N x D x m x m

#     # op11 - (1234) + (13)(24) - transpose
#     op11 = inputs.permute(0, 1, 3, 2)  # N x D x m x m

#     # op12 - (1234) + (234)(1) - place ii element in row i
#     op12 = torch.unsqueeze(diag_part, dim=3).repeat(1, 1, 1, dim)  # N x D x m x m

#     # op13 - (1234) + (134)(2) - place ii element in col i
#     op13 = torch.unsqueeze(diag_part, dim=2).repeat(1, 1, dim, 1)  # N x D x m x m

#     # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
#     op14 = torch.unsqueeze(sum_diag_part, dim=3).repeat(1, 1, dim, dim)   # N x D x m x m

#     # op15 - sum of all ops - place sum of all entries in all entries
#     op15 = torch.unsqueeze(torch.unsqueeze(sum_all, dim=2), dim=3).repeat(1, 1, dim, dim)  # N x D x m x m

#     if normalization is not None:
#         float_dim = dim.type(torch.FloatTensor)
#         if normalization is 'inf':
#             op2 = torch.div(op2, float_dim)
#             op3 = torch.div(op3, float_dim)
#             op4 = torch.div(op4, float_dim)
#             op5 = torch.div(op5, float_dim**2)
#             op6 = torch.div(op6, float_dim)
#             op7 = torch.div(op7, float_dim)
#             op8 = torch.div(op8, float_dim)
#             op9 = torch.div(op9, float_dim)
#             op14 = torch.div(op14, float_dim)
#             op15 = torch.div(op15, float_dim**2)

#     return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]


# def ops_2_to_1(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
#     diag_part = tf.matrix_diag_part(inputs)  # N x D x m
#     sum_diag_part = tf.reduce_sum(diag_part, axis=2, keepdims=True)  # N x D x 1
#     sum_of_rows = tf.reduce_sum(inputs, axis=3)  # N x D x m
#     sum_of_cols = tf.reduce_sum(inputs, axis=2)  # N x D x m
#     sum_all = tf.reduce_sum(inputs, axis=(2, 3))  # N x D

#     # op1 - (123) - extract diag
#     op1 = diag_part  # N x D x m

#     # op2 - (123) + (12)(3) - tile sum of diag part
#     op2 = tf.tile(sum_diag_part, [1, 1, dim])  # N x D x m

#     # op3 - (123) + (13)(2) - place sum of row i in element i
#     op3 = sum_of_rows  # N x D x m

#     # op4 - (123) + (23)(1) - place sum of col i in element i
#     op4 = sum_of_cols  # N x D x m

#     # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
#     op5 = tf.tile(tf.expand_dims(sum_all, axis=2), [1, 1, dim])  # N x D x m


#     if normalization is not None:
#         float_dim = tf.to_float(dim)
#         if normalization is 'inf':
#             op2 = tf.divide(op2, float_dim)
#             op3 = tf.divide(op3, float_dim)
#             op4 = tf.divide(op4, float_dim)
#             op5 = tf.divide(op5, float_dim ** 2)

#     return [op1, op2, op3, op4, op5]


# def ops_1_to_2(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m
#     sum_all = tf.reduce_sum(inputs, axis=2, keepdims=True)  # N x D x 1

#     # op1 - (123) - place on diag
#     op1 = tf.matrix_diag(inputs)  # N x D x m x m

#     # op2 - (123) + (12)(3) - tile sum on diag
#     op2 = tf.matrix_diag(tf.tile(sum_all, [1, 1, dim]))  # N x D x m x m

#     # op3 - (123) + (13)(2) - tile element i in row i
#     op3 = tf.tile(tf.expand_dims(inputs, axis=2), [1, 1, dim, 1])  # N x D x m x m

#     # op4 - (123) + (23)(1) - tile element i in col i
#     op4 = tf.tile(tf.expand_dims(inputs, axis=3), [1, 1, 1, dim])  # N x D x m x m

#     # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
#     op5 = tf.tile(tf.expand_dims(sum_all, axis=3), [1, 1, dim, dim])  # N x D x m x m

#     if normalization is not None:
#         float_dim = tf.to_float(dim)
#         if normalization is 'inf':
#             op2 = tf.divide(op2, float_dim)
#             op5 = tf.divide(op5, float_dim)

#     return [op1, op2, op3, op4, op5]


# def ops_1_to_1(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m
#     sum_all = tf.reduce_sum(inputs, axis=2, keepdims=True)  # N x D x 1

#     # op1 - (12) - identity
#     op1 = inputs  # N x D x m

#     # op2 - (1)(2) - tile sum of all
#     op2 = tf.tile(sum_all, [1, 1, dim])  # N x D x m

#     if normalization is not None:
#         float_dim = tf.to_float(dim)
#         if normalization is 'inf':
#             op2 = tf.divide(op2, float_dim)

#     return [op1, op2]

