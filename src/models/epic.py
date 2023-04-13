import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm
# from weight_std import Linear_wstd  ## !! DOESN'T WORK PROPERLY YET !! ##


class EPiC_layer(nn.Module):
    def __init__(self, local_in_dim, hid_dim, latent_dim):
        super(EPiC_layer, self).__init__()
        self.fc_global1 = weight_norm(nn.Linear(int(2*hid_dim)+latent_dim, hid_dim)) 
        self.fc_global2 = weight_norm(nn.Linear(hid_dim, latent_dim)) 
        self.fc_local1 = weight_norm(nn.Linear(local_in_dim+latent_dim, hid_dim))
        self.fc_local2 = weight_norm(nn.Linear(hid_dim, hid_dim))

    def forward(self, x_global, x_local):   # shapes: x_global[b,latent], x_local[b,n,latent_local]
        batch_size, n_points, latent_local = x_local.size()
        latent_global = x_global.size(1)

        x_pooled_mean = x_local.mean(1, keepdim=False)
        x_pooled_sum = x_local.sum(1, keepdim=False)
        x_pooledCATglobal = torch.cat([x_pooled_mean, x_pooled_sum, x_global], 1)
        x_global1 = F.leaky_relu(self.fc_global1(x_pooledCATglobal))  # new intermediate step
        x_global = F.leaky_relu(self.fc_global2(x_global1) + x_global) # with residual connection before AF

        x_global2local = x_global.view(-1,1,latent_global).repeat(1,n_points,1) # first add dimension, than expand it
        x_localCATglobal = torch.cat([x_local, x_global2local], 2)
        x_local1 = F.leaky_relu(self.fc_local1(x_localCATglobal))  # with residual connection before AF
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
        self.fc_global1 = weight_norm(nn.Linear(int(2 * hid_dim) + latent_dim, hid_dim))
        self.fc_global2 = weight_norm(nn.Linear(hid_dim, latent_dim))
        self.fc_local1 = weight_norm(nn.Linear(local_in_dim + latent_dim, hid_dim))
        self.fc_local2 = weight_norm(nn.Linear(hid_dim, hid_dim))
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


class EPiC_discriminator_mask(nn.Module):
    """EPiC classifier"""

    def __init__(self, args):
        """Initialise the EPiC classifier

        Parameters
        ----------
        args : keyword argruments
            Expects:
                hid_d = dimension of the hidden layers in the phi MLPs
                feats = number of local features
                epic_layers = number of epic layers
                latent = dimension of the latent space (in the networks that act
                         on the point clouds)
        """
        super().__init__()
        self.hid_d = args.hid_d
        self.feats = args.feats
        self.epic_layers = args.epic_layers
        self.latent = args.latent  # used for latent size of equiv concat
        self.sum_scale = args.sum_scale
        self.dropout_value = args.dropout_value

        self.fc_l1 = weight_norm(nn.Linear(self.feats, self.hid_d))
        self.fc_l2 = weight_norm(nn.Linear(self.hid_d, self.hid_d))

        self.fc_g1 = weight_norm(nn.Linear(int(2 * self.hid_d), self.hid_d))
        self.fc_g2 = weight_norm(nn.Linear(self.hid_d, self.latent))

        self.nn_list = nn.ModuleList()
        for _ in range(self.epic_layers):
            self.nn_list.append(
                EPiC_layer_mask(
                    self.hid_d, self.hid_d, self.latent, sum_scale=self.sum_scale, dropout=self.dropout_value
                )
            )

        self.fc_g3 = weight_norm(
            nn.Linear(int(2 * self.hid_d + self.latent), self.hid_d)
        )
        self.fc_g4 = weight_norm(nn.Linear(self.hid_d, self.hid_d))
        self.out = weight_norm(nn.Linear(self.hid_d, 1))

        self.dropout = nn.Dropout(self.dropout_value)

    def forward(self, x, mask):
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
        x_global = F.leaky_relu(self.dropout(self.fc_g2(x_global)))  # projecting down to latent size

        # equivariant connections
        x_global_in = x_global.clone()
        x_local_in = x_local.clone()
        for i in range(self.epic_layers):
            # contains residual connection
            x_global, x_local = self.nn_list[i](x_global, x_local, mask)
            x_global, x_local = x_global + x_global_in, x_local + x_local_in

        # again masking global features
        x_local = x_local * mask
        x_sum = x_local.sum(1, keepdim=False)
        x_mean = x_sum / mask.sum(1)
        x_sum = x_sum * self.sum_scale
        x = torch.cat([x_mean, x_sum, x_global], 1)

        x = F.leaky_relu(self.dropout(self.fc_g3(x)))
        x = F.leaky_relu(self.dropout(self.fc_g4(x) + x))
        x = self.out(x)
        return x


######################################################################################################
############################## CONDITIONAL MASKED EPIC CLASSIFER #####################################
######################################################################################################


# EPiC layer
class EPiC_layer_cond_mask(nn.Module):
    def __init__(self, local_in_dim, hid_dim, latent_dim, cond_feats=1, sum_scale=1e-2):
        super().__init__()
        self.fc_global1 = weight_norm(
            nn.Linear(int(2 * hid_dim) + latent_dim + cond_feats, hid_dim)
        )
        self.fc_global2 = weight_norm(nn.Linear(hid_dim, latent_dim))
        self.fc_local1 = weight_norm(nn.Linear(local_in_dim + latent_dim, hid_dim))
        self.fc_local2 = weight_norm(nn.Linear(hid_dim, hid_dim))
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
        x_local = x_local * mask
        x_pooled_sum = x_local.sum(1, keepdim=False)
        x_pooled_mean = x_pooled_sum / mask.sum(1)
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


# EPIC classifer
class EPiC_discriminator_cond_mask(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hid_d = args["hid_d"]
        self.feats = args["feats"]
        self.equiv_layers = args["equiv_layers_discriminator"]
        self.latent = args["latent"]  # used for latent size of equiv concat
        self.cond_feats = args["cond_feats"]
        self.sum_scale = args["sum_scale"]

        self.fc_l1 = self.weight_norm(nn.Linear(self.feats, self.hid_d))
        self.fc_l2 = self.weight_norm(nn.Linear(self.hid_d, self.hid_d))

        self.fc_g1 = self.weight_norm(
            nn.Linear(int(2 * self.hid_d + self.cond_feats), self.hid_d)
        )
        self.fc_g2 = self.weight_norm(nn.Linear(self.hid_d, self.latent))

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

        self.fc_g3 = self.weight_norm(
            nn.Linear(int(2 * self.hid_d + self.latent + self.cond_feats), self.hid_d)
        )
        self.fc_g4 = self.weight_norm(nn.Linear(self.hid_d, self.hid_d))
        self.out = self.weight_norm(nn.Linear(self.hid_d, 1))

    def forward(self, x, cond_tensor, mask):
        # x [B,N,F]    cond_tensor B,C     mask B,N,1
        # local encoding
        x_local = F.leaky_relu(self.fc_l1(x))
        x_local = F.leaky_relu(self.fc_l2(x_local) + x_local)

        # global features: masked
        x_local = x_local * mask
        x_sum = x_local.sum(1, keepdim=False)
        x_mean = x_sum / mask.sum(1)
        x_sum = x_sum * self.sum_scale
        x_global = torch.cat([x_mean, x_sum, cond_tensor], 1)
        x_global = F.leaky_relu(self.fc_g1(x_global))
        x_global = F.leaky_relu(self.fc_g2(x_global))  # projecting down to latent size

        # equivariant connections
        for i in range(self.equiv_layers):
            # contains residual connection
            x_global, x_local = self.nn_list[i](x_global, x_local, cond_tensor, mask)

        # again masking global features
        x_local = x_local * mask
        x_sum = x_local.sum(1, keepdim=False)
        x_mean = x_sum / mask.sum(1)
        x_sum = x_sum * self.sum_scale
        x = torch.cat([x_mean, x_sum, x_global, cond_tensor], 1)

        x = F.leaky_relu(self.fc_g3(x))
        x = F.leaky_relu(self.fc_g4(x) + x)
        x = self.out(x)
        return x




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

        ctx = self.act(self._layer_ctx(x_sum, x_mean, ctx)) # B,1,c
        ret = self.act(self.layer(ctx, x)) # B,N,d
        return ctx, ret




    
class EPiC_discriminator_mask_squash(nn.Module):
    """EPiC classifier with epic squash layers ONLY (no concat)"""

    def __init__(self, args):
        """Initialise the EPiC classifier

        Parameters
        ----------
        args : keyword argruments
            Expects:
                hid_d = dimension of the hidden layers in the phi MLPs
                feats = number of local features
                epic_layers = number of epic layers
                latent = dimension of the latent space (in the networks that act
                         on the point clouds)
        """
        super().__init__()
        self.hid_d = args.hid_d
        self.feats = args.feats
        self.epic_layers = args.epic_layers
        self.latent = args.latent  # used for latent size of equiv concat
        self.sum_scale = args.sum_scale

        self.fc_l1 = weight_norm(nn.Linear(self.feats, self.hid_d))

        self.fc_g1 = weight_norm(nn.Linear(self.hid_d, self.latent))
        self.fc_g2 = weight_norm(nn.Linear(self.hid_d, self.latent))
        self.fc_g3 = weight_norm(nn.Linear(self.latent+self.latent, self.latent))

        self.nn_list = nn.ModuleList()
        for _ in range(self.epic_layers):
            self.nn_list.append(
                EPiC_ConcatSquashLinear(
                    dim_in=self.hid_d, dim_out=self.hid_d, dim_ctx=self.latent, sum_scale=self.sum_scale
                )
            )

        self.fc_g4 = weight_norm(nn.Linear(self.latent, self.hid_d))
        self.fc_g5 = weight_norm(nn.Linear(self.hid_d, self.hid_d))
        self.out = weight_norm(nn.Linear(self.hid_d, 1))

    def forward(self, x, mask):
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
        x_local = F.leaky_relu(self.fc_l1(x))

        # global features: masked
        x = x * mask
        x_sum = (x * mask).sum(1, keepdim=True)  # B,1,d
        x_mean = x_sum / mask.sum(1, keepdim=True)  # B,1,d
        x_sum = x_sum * self.sum_scale

        x_mean = self.fc_g1(x_mean) # B,1,C
        x_sum = self.fc_g2(x_sum) # B,1,C

        x_global = torch.cat([x_mean, x_sum], -1) # B,1,C+C
        x_global = F.leaky_relu(self.fc_g3(x_global))

        x_global_in, x_local_in = x_global.clone(), x_local.clone()
        # equivariant connections
        for i in range(self.epic_layers):
            # contains residual connection
            x_global, x_local = self.nn_list[i](x_global, x_local, mask)
            x_global, x_local = x_global+x_global_in, x_local+x_local_in   # skip connection to sampled input

        x = F.leaky_relu(self.fc_g4(x_global))
        x = F.leaky_relu(self.fc_g5(x) + x)
        x = self.out(x)
        return x
    


############################################################


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

        ctx = self._layer_ctx(x_sum, x_mean, ctx) # B,1,c
        ret = self.layer(ctx, x) # B,N,d
        return ctx, ret



class EPiC_discriminator_mask_squash_res(nn.Module):
    """EPiC classifier with epic squash layers ONLY (no concat) and residual connections"""

    def __init__(self, args):
        """Initialise the EPiC classifier

        Parameters
        ----------
        args : keyword argruments
            Expects:
                hid_d = dimension of the hidden layers in the phi MLPs
                feats = number of local features
                epic_layers = number of epic layers
                latent = dimension of the latent space (in the networks that act
                         on the point clouds)
        """
        super().__init__()
        self.hid_d = args.hid_d
        self.feats = args.feats
        self.epic_layers = args.epic_layers
        self.latent = args.latent  # used for latent size of equiv concat
        self.sum_scale = args.sum_scale

        self.fc_l1 = weight_norm(nn.Linear(self.feats, self.hid_d))

        self.fc_g1 = weight_norm(nn.Linear(self.hid_d, self.latent))
        self.fc_g2 = weight_norm(nn.Linear(self.hid_d, self.latent))
        self.fc_g3 = weight_norm(nn.Linear(self.latent+self.latent, self.latent))

        self.nn_list = nn.ModuleList()
        for _ in range(self.epic_layers):
            self.nn_list.append(
                EPiC_ConcatSquashLinear_noAct(
                    dim_in=self.hid_d, dim_out=self.hid_d, dim_ctx=self.latent, sum_scale=self.sum_scale
                )
            )

        self.fc_g4 = weight_norm(nn.Linear(self.latent, self.hid_d))
        self.fc_g5 = weight_norm(nn.Linear(self.hid_d, self.hid_d))
        self.out = weight_norm(nn.Linear(self.hid_d, 1))

    def forward(self, x, mask):
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
        x_local = F.leaky_relu(self.fc_l1(x))

        # global features: masked
        x = x * mask
        x_sum = (x * mask).sum(1, keepdim=True)  # B,1,d
        x_mean = x_sum / mask.sum(1, keepdim=True)  # B,1,d
        x_sum = x_sum * self.sum_scale

        x_mean = self.fc_g1(x_mean) # B,1,C
        x_sum = self.fc_g2(x_sum) # B,1,C

        x_global = torch.cat([x_mean, x_sum], -1) # B,1,C+C
        x_global = F.leaky_relu(self.fc_g3(x_global))

        # equivariant connections
        for i in range(self.epic_layers):
            # contains residual connection
            x_global_new, x_local_new = self.nn_list[i](x_global, x_local, mask)
            # residual connection
            x_global = F.leaky_relu(x_global_new+x_global)   
            x_local = F.leaky_relu(x_local_new+x_local)

        x = F.leaky_relu(self.fc_g4(x_global))
        x = F.leaky_relu(self.fc_g5(x) + x)
        x = self.out(x)
        return x
    

