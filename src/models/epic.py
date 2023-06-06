from typing import Mapping, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.nn.utils.weight_norm as weight_norm

from src.models.modules import DenseNetwork

# from weight_std import Linear_wstd  ## !! DOESN'T WORK PROPERLY YET !! ##


class weight_norm(nn.Module):
    append_g = "_g"
    append_v = "_v"

    def __init__(self, module, weights):
        super(weight_norm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()

    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)

            # construct g,v such that w = g/||v|| * v
            g = torch.norm(w)
            v = w / g.expand_as(w)
            g = nn.Parameter(g.data)
            v = nn.Parameter(v.data)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v

            # remove w from parameter list
            del self.module._parameters[name_w]

            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)

    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            w = v * (g / torch.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class EPiC_layer(nn.Module):
    def __init__(self, local_in_dim, hid_dim, latent_dim):
        super(EPiC_layer, self).__init__()

        fc_global1 = nn.Linear(int(2 * hid_dim) + latent_dim, hid_dim)
        self.fc_global1 = weight_norm(fc_global1, ["weight"])

        fc_global2 = nn.Linear(hid_dim, latent_dim)
        self.fc_global2 = weight_norm(fc_global2, ["weight"])

        fc_local1 = nn.Linear(local_in_dim + latent_dim, hid_dim)
        self.fc_local1 = weight_norm(fc_local1, ["weight"])

        fc_local2 = nn.Linear(hid_dim, hid_dim)
        self.fc_local2 = weight_norm(fc_local2, ["weight"])

    def forward(
        self, x_global, x_local
    ):  # shapes: x_global[b,latent], x_local[b,n,latent_local]
        batch_size, n_points, latent_local = x_local.size()
        latent_global = x_global.size(1)

        x_pooled_mean = x_local.mean(1, keepdim=False)
        x_pooled_sum = x_local.sum(1, keepdim=False)
        x_pooledCATglobal = torch.cat([x_pooled_mean, x_pooled_sum, x_global], 1)
        x_global1 = F.leaky_relu(
            self.fc_global1(x_pooledCATglobal)
        )  # new intermediate step
        x_global = F.leaky_relu(
            self.fc_global2(x_global1) + x_global
        )  # with residual connection before AF

        x_global2local = x_global.view(-1, 1, latent_global).repeat(
            1, n_points, 1
        )  # first add dimension, than expand it
        x_localCATglobal = torch.cat([x_local, x_global2local], 2)
        x_local1 = F.leaky_relu(
            self.fc_local1(x_localCATglobal)
        )  # with residual connection before AF
        x_local = F.leaky_relu(self.fc_local2(x_local1) + x_local)

        return x_global, x_local


class EPiC_layer_mask(nn.Module):
    """Definition of the EPIC layer"""

    def __init__(
        self,
        local_in_dim: int,
        hid_dim: int,
        latent_dim: int,
        sum_scale: float = 1e-2,
        dropout: float = 0.25,
    ):
        """Initialise EPiC layer

        Parameters
        ----------
        local_in_dim : int
            Dimension of local features
        hid_dim : int
            Dimension of hidden layer
        latent_dim : int
            Dimension of latent space
        sum_scale : float, optional
            Scale factor for the result of the sum pooling operation, by default 1e-2
        """
        super().__init__()
        # self.fc_global1 = weight_norm(nn.Linear(int(2 * hid_dim) + latent_dim, hid_dim))
        # self.fc_global2 = weight_norm(nn.Linear(hid_dim, latent_dim))
        # self.fc_local1 = weight_norm(nn.Linear(local_in_dim + latent_dim, hid_dim))
        # self.fc_local2 = weight_norm(nn.Linear(hid_dim, hid_dim))
        fc_global1 = nn.Linear(int(2 * hid_dim) + latent_dim, hid_dim)
        self.fc_global1 = weight_norm(fc_global1, ["weight"])
        fc_global2 = nn.Linear(hid_dim, latent_dim)
        self.fc_global2 = weight_norm(fc_global2, ["weight"])
        fc_local1 = nn.Linear(local_in_dim + latent_dim, hid_dim)
        self.fc_local1 = weight_norm(fc_local1, ["weight"])
        fc_local2 = nn.Linear(hid_dim, hid_dim)
        self.fc_local2 = weight_norm(fc_local2, ["weight"])

        self.sum_scale = sum_scale
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_global, x_local, mask):
        """Definition of the EPiC layer forward pass

        Parameters
        ----------
        x_global : torch.tensor
            Global features of shape [batch_size, dim_latent_global]
        x_local : torch.tensor
            Local features of shape [batch_size, N_points, dim_latent_local]
        mask : torch.tensor
            Mask of shape [batch_size, N_points, 1]. All non-padded values are
            "True", padded values are "False".
            This allows to exclude zero-padded points from the sum/mean aggregation
            functions

        Returns
        -------
        x_global
            Global features after the EPiC layer transformation
        x_local
            Local features after the EPiC layer transformation
        """
        batch_size, n_points, latent_local = x_local.size()
        latent_global = x_global.size(1)  # get number of global features

        # calculate the mean along the axis that represents the sets
        # communication between points is masked
        x_local = x_local * mask
        x_pooled_sum = x_local.sum(1, keepdim=False)
        x_pooled_mean = x_pooled_sum / mask.sum(1)
        x_pooled_sum = x_pooled_sum * self.sum_scale

        x_pooledCATglobal = torch.cat([x_pooled_mean, x_pooled_sum, x_global], 1)
        # new intermediate step
        x_global1 = F.leaky_relu(self.dropout(self.fc_global1(x_pooledCATglobal)))
        # with residual connection before AF
        x_global = F.leaky_relu(self.dropout(self.fc_global2(x_global1) + x_global))

        # point wise function does not need to be masked
        # first add dimension, than expand it
        x_global2local = x_global.view(-1, 1, latent_global).repeat(1, n_points, 1)
        x_localCATglobal = torch.cat([x_local, x_global2local], 2)
        # with residual connection before AF
        x_local1 = F.leaky_relu(self.dropout(self.fc_local1(x_localCATglobal)))
        x_local = F.leaky_relu(self.dropout(self.fc_local2(x_local1) + x_local))

        return x_global, x_local


# EPiC layer
class EPiC_layer_cond_mask(nn.Module):
    def __init__(self, local_in_dim, hid_dim, latent_dim, cond_feats=1, sum_scale=1e-2):
        super().__init__()
        # self.fc_global1 = weight_norm(
        #     nn.Linear(int(2 * hid_dim) + latent_dim + cond_feats, hid_dim)
        # )
        # self.fc_global2 = weight_norm(nn.Linear(hid_dim, latent_dim))
        # self.fc_local1 = weight_norm(nn.Linear(local_in_dim + latent_dim, hid_dim))
        # self.fc_local2 = weight_norm(nn.Linear(hid_dim, hid_dim))
        fc_global1 = nn.Linear(int(2 * hid_dim) + latent_dim + cond_feats, hid_dim)
        self.fc_global1 = weight_norm(fc_global1, ["weight"])
        fc_global2 = nn.Linear(hid_dim, latent_dim)
        self.fc_global2 = weight_norm(fc_global2, ["weight"])
        fc_local1 = nn.Linear(local_in_dim + latent_dim, hid_dim)
        self.fc_local1 = weight_norm(fc_local1, ["weight"])
        fc_local2 = nn.Linear(hid_dim, hid_dim)
        self.fc_local2 = weight_norm(fc_local2, ["weight"])

        self.sum_scale = sum_scale

    def forward(self, x_global, x_local, cond_tensor, mask):  # shapes:
        # - x_global[b,latent]
        # - x_local[b,n,latent_local]
        # - points_tensor [b,cond_feats]
        # - mask[B,N,1]
        # mask: all non-padded values = True      all zero padded = False
        batch_size, n_points, latent_local = x_local.size()
        latent_global = x_global.size(1)

        # communication between points is masked
        x_local = x_local * mask.view(x_local.shape[0], x_local.shape[1], 1)
        x_pooled_sum = x_local.sum(1, keepdim=False)
        x_pooled_mean = x_pooled_sum / mask.sum(1).view(x_pooled_sum.shape[0], 1)
        x_pooled_sum = x_pooled_sum * self.sum_scale
        x_pooledCATglobal = torch.cat(
            [x_pooled_mean, x_pooled_sum, x_global, cond_tensor], 1
        )
        # new intermediate step
        x_global1 = F.leaky_relu(self.fc_global1(x_pooledCATglobal))
        # with residual connection before AF
        x_global = F.leaky_relu(self.fc_global2(x_global1) + x_global)

        # point wise function does not need to be masked
        # first add dimension, than expand it
        x_global2local = x_global.view(-1, 1, latent_global).repeat(1, n_points, 1)
        x_localCATglobal = torch.cat([x_local, x_global2local], 2)
        # with residual connection before AF
        x_local1 = F.leaky_relu(self.fc_local1(x_localCATglobal))
        x_local = F.leaky_relu(self.fc_local2(x_local1) + x_local)

        return x_global, x_local


#### MASKED EPIC SQUASH MODEL ####


# inspired by FFJORD https://github.com/rtqichen/ffjord/blob/master/lib/layers/diffeq_layers/basic.py
class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super().__init__()
        self._layer = weight_norm(nn.Linear(dim_in, dim_out))
        self._hyper_bias = weight_norm(nn.Linear(dim_ctx, dim_out, bias=False))
        self._hyper_gate = weight_norm(nn.Linear(dim_ctx, dim_out))

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret


class ConcatSquashLinear_2inputs(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx1, dim_ctx2):
        super().__init__()
        self._layer = weight_norm(nn.Linear(dim_in, dim_out))
        self._hyper_bias1 = weight_norm(nn.Linear(dim_ctx1, dim_out, bias=False))
        self._hyper_gate1 = weight_norm(nn.Linear(dim_ctx1, dim_out))
        self._hyper_bias2 = weight_norm(nn.Linear(dim_ctx2, dim_out, bias=False))
        self._hyper_gate2 = weight_norm(nn.Linear(dim_ctx2, dim_out))

    def forward(self, ctx1, ctx2, x):
        gate1 = torch.sigmoid(self._hyper_gate1(ctx1))
        bias1 = self._hyper_bias1(ctx1)
        gate2 = torch.sigmoid(self._hyper_gate2(ctx2))
        bias2 = self._hyper_bias2(ctx2)
        ret = self._layer(x) * gate1 * gate2 + bias1 + bias2
        return ret


class EPiC_ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx, sum_scale=1e-4):
        super().__init__()
        self.sum_scale = sum_scale
        self.act = nn.LeakyReLU()
        self._layer_ctx = ConcatSquashLinear_2inputs(dim_ctx, dim_ctx, dim_in, dim_in)
        self.layer = ConcatSquashLinear(dim_in, dim_out, dim_ctx)

    def forward(self, ctx, x, mask):
        x = x * mask
        x_sum = (x * mask).sum(1, keepdim=True)  # B,1,d
        x_mean = x_sum / mask.sum(1, keepdim=True)  # B,1,d
        x_sum = x_sum * self.sum_scale

        ctx = self.act(self._layer_ctx(x_sum, x_mean, ctx))  # B,1,c
        ret = self.act(self.layer(ctx, x))  # B,N,d
        return ctx, ret


class EPiC_ConcatSquashLinear_noAct(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx, sum_scale=1e-4):
        super().__init__()
        self.sum_scale = sum_scale
        self._layer_ctx = ConcatSquashLinear_2inputs(dim_ctx, dim_ctx, dim_in, dim_in)
        self.layer = ConcatSquashLinear(dim_in, dim_out, dim_ctx)

    def forward(self, ctx, x, mask):
        x = x * mask
        x_sum = (x * mask).sum(1, keepdim=True)  # B,1,d
        x_mean = x_sum / mask.sum(1, keepdim=True)  # B,1,d
        x_sum = x_sum * self.sum_scale

        ctx = self._layer_ctx(x_sum, x_mean, ctx)  # B,1,c
        ret = self.layer(ctx, x)  # B,N,d
        return ctx, ret


class EPiC_Encoder(nn.Module):
    def __init__(
        self,
        hid_d: int,
        feats: int,
        equiv_layers: int,
        latent: int,
        sum_scale: float = 1e-2,
        cond_dim: int = None,
    ) -> None:
        super().__init__()

        self.hid_d = hid_d
        self.feats = feats
        self.equiv_layers = equiv_layers
        self.latent = latent
        self.cond_feats = cond_dim
        self.sum_scale = sum_scale

        fc_l1 = nn.Linear(self.feats, self.hid_d)
        self.fc_l1 = weight_norm(fc_l1, ["weight"])

        fc_l2 = nn.Linear(self.hid_d, self.hid_d)
        self.fc_l2 = weight_norm(fc_l2, ["weight"])

        fc_g1 = nn.Linear(int(2 * self.hid_d + self.cond_feats), self.hid_d)
        self.fc_g1 = weight_norm(fc_g1, ["weight"])

        fc_g2 = nn.Linear(self.hid_d, self.latent)
        self.fc_g2 = weight_norm(fc_g2, ["weight"])

        output_dense = nn.Linear(self.hid_d, self.feats)
        self.output_dense = weight_norm(output_dense, ["weight"])

        self.nn_list = nn.ModuleList()

        if self.cond_feats == 0:
            for _ in range(self.equiv_layers):
                self.nn_list.append(
                    EPiC_layer_mask(
                        self.hid_d,
                        self.hid_d,
                        self.latent,
                        sum_scale=self.sum_scale,
                    )
                )
        else:
            for _ in range(self.equiv_layers):
                self.nn_list.append(
                    EPiC_layer_cond_mask(
                        self.hid_d,
                        self.hid_d,
                        self.latent,
                        self.cond_feats,
                        sum_scale=self.sum_scale,
                    )
                )

        # self.fc_g3 = weight_norm(
        #     nn.Linear(int(2 * self.hid_d + self.latent + self.cond_feats), self.hid_d)
        # )
        # self.fc_g4 = weight_norm(nn.Linear(self.hid_d, self.hid_d))
        # self.out = weight_norm(nn.Linear(self.hid_d, 1))

    def _noncond_forward(self, x, mask):
        """Forward propagation through the network
        Parameters
        ----------
        x : torch.tensor
            Input tensor of shape [batch_size, N_points, N_features]
        mask : torch.tensor
            Mask of shape [batch_size, N_points, 1]
            This allows to exclude zero-padded points from the sum/mean aggregation
            functions
        Returns
        -------
        x
            Output of the network
        """
        # local encoding
        x_local = F.leaky_relu(self.dropout(self.fc_l1(x)))
        x_local = F.leaky_relu(self.dropout(self.fc_l2(x_local) + x_local))

        # global features: masked
        x_local = x_local * mask
        x_sum = x_local.sum(1, keepdim=False)
        x_mean = x_sum / mask.sum(1)
        x_sum = x_sum * self.sum_scale

        x_global = torch.cat([x_mean, x_sum], 1)
        x_global = F.leaky_relu(self.dropout(self.fc_g1(x_global)))
        x_global = F.leaky_relu(
            self.dropout(self.fc_g2(x_global))
        )  # projecting down to latent size

        # equivariant connections
        x_global_in = x_global.clone()
        x_local_in = x_local.clone()
        for i in range(self.epic_layers):
            # contains residual connection
            x_global, x_local = self.nn_list[i](x_global, x_local, mask)
            x_global, x_local = x_global + x_global_in, x_local + x_local_in

        return x_local, x_global

    def _cond_forward(self, x, cond_tensor, mask):
        x_local = F.leaky_relu(self.fc_l1(x))
        x_local = F.leaky_relu(self.fc_l2(x_local) + x_local)

        # global features: masked
        x_local = x_local * mask.view(x_local.shape[0], x_local.shape[1], 1)
        x_sum = x_local.sum(1, keepdim=False)
        x_mean = x_sum / mask.sum(1).view(x_sum.shape[0], 1)
        x_sum = x_sum * self.sum_scale
        x_global = torch.cat([x_mean, x_sum, cond_tensor], 1)
        x_global = F.leaky_relu(self.fc_g1(x_global))
        x_global = F.leaky_relu(self.fc_g2(x_global))  # projecting down to latent size

        # equivariant connections
        for i in range(self.equiv_layers):
            # contains residual connection
            x_global, x_local = self.nn_list[i](x_global, x_local, cond_tensor, mask)

        return x_local, x_global

    def forward(self, x, cond_tensor=None, mask=None):
        if cond_tensor is None:
            x_local, x_global = self._noncond_forward(x, mask)
        else:
            x_local, x_global = self._cond_forward(x, cond_tensor, mask)

        x_local = self.output_dense(x_local)
        return x_local, x_global


class EPiC2_Encoder(nn.Module):
    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        equiv_layers: int = 6,
        sum_scale: float = 1e-2,
        latent: int = 32,
        hid_d: int = 128,
    ) -> None:
        super().__init__()

        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        self.equiv_layers = equiv_layers
        self.sum_scale = sum_scale
        self.latent = latent
        self.hid_d = hid_d

        fc_l1 = nn.Linear(self.inpt_dim, self.hid_d)
        self.fc_l1 = weight_norm(fc_l1, ["weight"])

        fc_l2 = nn.Linear(self.hid_d, self.hid_d)
        self.fc_l2 = weight_norm(fc_l2, ["weight"])

        fc_g1 = nn.Linear(int(2 * self.hid_d + self.cond_feats), self.hid_d)
        self.fc_g1 = weight_norm(fc_g1, ["weight"])

        fc_g2 = nn.Linear(self.hid_d, self.latent)
        self.fc_g2 = weight_norm(fc_g2, ["weight"])

        output_dense = nn.Linear(self.hid_d, self.outp_dim)
        self.output_dense = weight_norm(output_dense, ["weight"])

        self.nn_list = nn.ModuleList()

        for _ in range(self.equiv_layers):
            self.nn_list.append(
                EPiC_layer_cond_mask(
                    self.hid_d,
                    self.hid_d,
                    self.latent,
                    self.ctxt_dim,
                    sum_scale=self.sum_scale,
                )
            )
    
    def forward(self, x, mask, cond_tensor):
        x_local = F.leaky_relu(self.fc_l1(x))
        x_local = F.leaky_relu(self.fc_l2(x_local) + x_local)

        # global features: masked
        x_local = x_local * mask.view(x_local.shape[0], x_local.shape[1], 1)
        x_sum = x_local.sum(1, keepdim=False)
        x_mean = x_sum / mask.sum(1).view(x_sum.shape[0], 1)
        x_sum = x_sum * self.sum_scale
        x_global = torch.cat([x_mean, x_sum, cond_tensor], 1)
        x_global = F.leaky_relu(self.fc_g1(x_global))
        x_global = F.leaky_relu(self.fc_g2(x_global))  # projecting down to latent size

        # equivariant connections
        for i in range(self.equiv_layers):
            # contains residual connection
            x_global, x_local = self.nn_list[i](x_global, x_local, cond_tensor, mask)

        return x_local, x_global