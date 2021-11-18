# import tensorflow as tf
import torch
import torch.nn.functional as F

class ReLU(torch.nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        
    def forward(self, x, A):
        out = {}
        for key in x:
            out[key] = torch.nn.functional.relu(x[key])
        return out
    
class eLU(torch.nn.Module):
    def __init__(self):
        super(eLU, self).__init__()
        
    def forward(self, x, A):
        out = {}
        for key in x:
            out[key] = torch.nn.functional.elu(x[key])
        return out

class SiLU(torch.nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()
        
    def forward(self, x, A):
        out = {}
        for key in x:
#             out[key] = torch.nn.functional.silu(x[key])
            out[key] = x[key] * torch.sigmoid(x[key])
        return out

class GroupNorm(torch.nn.Module):
    def __init__(self, groups):
        super(GroupNorm, self).__init__()
        self.groups = groups
        
    def forward(self, x, A):
        out = {}
        for key in x:
            out[key] = torch.nn.functional.group_norm(x[key], num_groups=self.groups)
        return out

class Bilinear(torch.nn.Module):
    def __init__(self, fin, fout, reps):
        super(Bilinear, self).__init__()
        self.reps = reps
        if 0 in reps:
            self.bilinear0 = torch.nn.Bilinear(fin, fin, fout)
        if 1 in reps:
            self.bilinear1 = torch.nn.Bilinear(fin, fin, fout)
        if 2 in reps:
            self.bilinear2 = torch.nn.Bilinear(fin, fin, fout)
        
    def forward(self, x, A):
        out = {}
        for key in x:
#             print(f'{key} : {x[key].shape}')
            if key == '0':
                xin = x[key]
#                 print(f'xin : {xin.shape}')
#                 xin = torch.transpose(xin, 1, 3)
#                 print(f'xin : {xin.shape}')
#                 xin_shape = xin.shape
#                 xin = xin.reshape(xin_shape[0], -1, xin_shape[-1])
                xout = self.bilinear0(xin, xin)
#                 xout = xout.reshape(xin_shape[0], xin_shape[1], xin_shape[2], -1)
                out[key] = xout
            if key == '1':
                xin = x[key]
#                 print(f'xin : {xin.shape}')
                xin = torch.transpose(xin, 1, 2)
                xin_shape = xin.shape
                xin = xin.reshape(xin_shape[0], -1, xin_shape[-1]) .contiguous()
#                 print(f'xin : {xin.shape}')
                xout = self.bilinear1(xin, xin)
                xout = xout.reshape(xin_shape[0], xin_shape[1], -1)
                out[key] = torch.transpose(xout, 1, 2)
            if key == '2':
                xin = x[key]
#                 print(f'xin : {xin.shape}')
                xin = torch.transpose(xin, 1, 3)
#                 print(f'xin : {xin.shape}')
                xin_shape = xin.shape
                xin = xin.reshape(xin_shape[0], -1, xin_shape[-1])
                xout = self.bilinear2(xin, xin)
                xout = xout.reshape(xin_shape[0], xin_shape[1], xin_shape[2], -1)
                out[key] = torch.transpose(xout, 1, 3)
            
        return out

class diag_offdiag_maxpool(torch.nn.Module):
    """diag_offdiag_maxpool"""

    def __init__(self):
        super(diag_offdiag_maxpool, self).__init__()
#         self.ones = torch.nn.Parameter(torch.ones(1))

    def forward(self, inputs, A):
        inputs2 = inputs['2']
        max_diag = torch.max(torch.diagonal(inputs2, dim1=-2, dim2=-1), dim=2)[0] #BxS
        max_val = torch.max(max_diag)
        min_val = torch.max(torch.mul(inputs2, -1))
        val = torch.abs(max_val+min_val)
        min_mat = torch.unsqueeze(torch.unsqueeze(torch.diagonal(torch.add(torch.mul(torch.diag_embed(inputs2[0][0]),0),val)), dim=0), dim=0)
        max_offdiag = torch.max(torch.max(torch.sub(inputs2, min_mat), dim=2)[0], dim=2)[0]
        
        inputs1 = inputs['1']
        pool1 = torch.max(inputs1, dim=-1)[0]

        return torch.cat((max_diag, max_offdiag, pool1, inputs['0']), dim=1) #output BxSx4
    
# def diag_offdiag_maxpool(input): #input.shape BxSxNxN
#     max_diag = torch.max(torch.diagonal(input), dim=2) #BxS
# #     max_diag = tf.reduce_max(tf.matrix_diag_part(input), axis=2) #BxS

#     max_val = torch.max(max_diag)
# #     max_val = tf.reduce_max(max_diag)

#     min_val = torch.max(torch.mul(input, torch.ones(1)*-1))
# #     min_val = tf.reduce_max(tf.multiply(input, tf.constant(-1.)))
#     val = torch.abs(max_val+min_val)
# #     val = tf.abs(max_val+min_val)
#     min_mat = torch.unsqueeze(torch.unsqueeze(torch.diagonal(torch.add(torch.mul(torch.diag_embed(input[0][0]),0),val)), dim=0), dim=0)
# #     min_mat = tf.expand_dims(tf.expand_dims(tf.matrix_diag(tf.add(tf.multiply(tf.matrix_diag_part(input[0][0]),0),val)), axis=0), axis=0)
#     max_offdiag = torch.max(torch.max(torch.subtract(input, min_mat), dim=2), dim=2)
# #     max_offdiag = tf.reduce_max(tf.subtract(input, min_mat), axis=[2, 3])

#     return torch.cat((max_diag, max_offdiag), dim=1) #output BxSx2
# #     return tf.concat([max_diag, max_offdiag], axis=1) #output BxSx2


def spatial_dropout(x, keep_prob, is_training, seed=1234):
    if is_training:
        output = spatial_dropout_imp(x, keep_prob, seed)
    else:
        output = x
#     output = tf.cond(is_training, lambda: spatial_dropout_imp(x, keep_prob, seed), lambda: x)
    return output

def spatial_dropout_imp(x, keep_prob, seed=1234):
    drop = keep_prob + torch.rand((1, x.shape[1], 1, 1), seed=seed)
#     drop = keep_prob + tf.random_uniform(shape=[1, tf.shape(x)[1], 1, 1], minval=0, maxval=1, seed=seed)
    drop = torch.floor(drop)
#     drop = tf.floor(drop)
    return torch.divide(torch.matmul(drop, x), keep_prob)
#     return tf.divide(tf.multiply(drop, x), keep_prob)



class fully_connected(torch.nn.Module):
    """One-layer fully-connected RELU net with batch norm."""

    def __init__(self, n_in, n_hid, do_prob=0., activation_fn='relu'):
        super(fully_connected, self).__init__()
        self.fc1 = torch.nn.Linear(n_in, n_hid)
        self.do = torch.nn.Dropout(p=do_prob, inplace=True)
        self.dropout_prob = do_prob
        self.activation_fn = activation_fn

        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
                
                
    def forward(self, inputs):
        """ Fully connected layer with non-linear operation.

        Args:
          inputs: 2-D tensor BxN
          num_outputs: int

        Returns:
          Variable tensor of size B x num_outputs.
        """
        inputs = inputs.float()
        if self.activation_fn == None:
            x = self.fc1(inputs)
        else:
            x = F.relu(self.fc1(inputs))
#         x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.do(x)

        return x

# def fully_connected(inputs,
#                     num_outputs,
#                     scope,
#                     activation_fn=tf.nn.relu):
#     """ Fully connected layer with non-linear operation.

#     Args:
#       inputs: 2-D tensor BxN
#       num_outputs: int

#     Returns:
#       Variable tensor of size B x num_outputs.
#     """
#     with tf.variable_scope(scope) as sc:
#         num_input_units = inputs.get_shape()[-1].value
#         initializer = tf.contrib.layers.xavier_initializer()
#         weights = tf.get_variable("weights", shape=[num_input_units, num_outputs],
#                                   initializer=initializer, dtype=tf.float32)

#         outputs = tf.matmul(inputs, weights)
#         biases = tf.get_variable('biases', [num_outputs],
#                                   initializer=tf.constant_initializer(0.))

#         outputs = tf.nn.bias_add(outputs, biases)

#         if activation_fn is not None:
#             outputs = activation_fn(outputs)
#         return outputs