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

    def forward(self, x, cond_tensor, mask):
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
        x_local = self.output_dense(x_local)
        return x_local


class Local_EPiC_layer_cond_mask(nn.Module):
    def __init__(self, local_in_dim, hid_dim, latent_dim, cond_feats=1, sum_scale=1e-2):
        super().__init__()

        fc_global1 = nn.Linear(int(2 * hid_dim) + latent_dim + cond_feats, hid_dim)
        self.fc_global1 = weight_norm(fc_global1, ["weight"])
        fc_global2 = nn.Linear(hid_dim + cond_feats, latent_dim)
        self.fc_global2 = weight_norm(fc_global2, ["weight"])
        fc_local1 = nn.Linear(local_in_dim + latent_dim + cond_feats, hid_dim)
        self.fc_local1 = weight_norm(fc_local1, ["weight"])
        fc_local2 = nn.Linear(hid_dim + cond_feats, hid_dim)
        self.fc_local2 = weight_norm(fc_local2, ["weight"])

        self.sum_scale = sum_scale

    def forward(self, x_global, x_local, cond_tensor, mask):  # shapes:
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
        x_global1 = torch.cat([x_global1, cond_tensor], axis=1)
        x_global = F.leaky_relu(self.fc_global2(x_global1) + x_global)

        # point wise function does not need to be masked
        # first add dimension, than expand it
        x_global2local = x_global.view(-1, 1, latent_global).repeat(1, n_points, 1)
        x_cond = cond_tensor.view(-1, 1, cond_tensor.shape[-1]).repeat(1, n_points, 1)
        x_localCATglobal = torch.cat([x_local, x_global2local, x_cond], 2)
        # with residual connection before AF
        x_local1 = F.leaky_relu(self.fc_local1(x_localCATglobal))
        x_local1 = torch.cat([x_local1, x_cond], axis=2)
        x_local = F.leaky_relu(self.fc_local2(x_local1) + x_local)

        return x_global, x_local


class Local_EPiC_Encoder(nn.Module):
    def __init__(
        self,
        hid_d: int,
        feats: int,
        ctxt_dim: int = 0,
        equiv_layers: int = 6,
        sum_scale: float = 1e-2,
        latent: int = 32,
        cond_dim: int = None,
    ) -> None:
        super().__init__()
        
        self.hid_d = hid_d
        self.feats = feats
        self.ctxt_dim = ctxt_dim
        self.equiv_layers = equiv_layers
        self.sum_scale = sum_scale
        self.cond_feats = cond_dim
        self.latent = latent
        self.hid_d = hid_d

        fc_l1 = nn.Linear(self.feats + self.cond_feats, self.hid_d)
        self.fc_l1 = weight_norm(fc_l1, ["weight"])

        fc_l2 = nn.Linear(self.hid_d + self.cond_feats, self.hid_d)
        self.fc_l2 = weight_norm(fc_l2, ["weight"])

        fc_g1 = nn.Linear(int(2 * self.hid_d) + self.cond_feats, self.hid_d)
        self.fc_g1 = weight_norm(fc_g1, ["weight"])

        fc_g2 = nn.Linear(self.hid_d + self.cond_feats, self.latent)
        self.fc_g2 = weight_norm(fc_g2, ["weight"])

        output_dense = nn.Linear(self.hid_d + self.cond_feats, self.feats)
        self.output_dense = weight_norm(output_dense, ["weight"])

        self.nn_list = nn.ModuleList()

        for _ in range(self.equiv_layers):
            self.nn_list.append(
                Local_EPiC_layer_cond_mask(
                    self.hid_d,
                    self.hid_d,
                    self.latent,
                    self.cond_feats,
                    sum_scale=self.sum_scale,
                )
            )

    def forward(self, x, mask, cond_tensor):
        n_points = x.shape[1]
        x_cond = cond_tensor.view(-1, 1, cond_tensor.shape[-1]).repeat(1, n_points, 1)
        x_local_cat = torch.cat([x, x_cond], 2)
        x_local = F.leaky_relu(self.fc_l1(x_local_cat))
        x_local_cat = torch.cat([x_local, x_cond], 2)
        x_local = F.leaky_relu(self.fc_l2(x_local_cat) + x_local)

        # global features: masked
        x_local = x_local * mask.view(x_local.shape[0], x_local.shape[1], 1)
        x_sum = x_local.sum(1, keepdim=False)
        x_mean = x_sum / mask.sum(1).view(x_sum.shape[0], 1)
        x_sum = x_sum * self.sum_scale
        x_global = torch.cat([x_mean, x_sum, cond_tensor], 1)
        x_global = F.leaky_relu(self.fc_g1(x_global))
        x_global = torch.cat([x_global, cond_tensor], axis=1)
        x_global = F.leaky_relu(self.fc_g2(x_global))  # projecting down to latent size

        # equivariant connections
        for i in range(self.equiv_layers):
            # contains residual connection
            x_global, x_local = self.nn_list[i](x_global, x_local, cond_tensor, mask)

        # Concat the cond tensor to the local features and pass through a dense layer
        x_local = torch.cat([x_local, x_cond], axis=2)
        return self.output_dense(x_local)
