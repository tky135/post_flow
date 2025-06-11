import torch
import torch.nn as nn
import math
from diffusers.models.embeddings import Timesteps

# Generic MLP model
class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU, linear_layer=nn.Linear):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.linear_layer = linear_layer
        for i in range(len(layer_sizes) - 1):
            self.layers.append(linear_layer(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(activation())

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        return x


# Model for velocity field v(x, t)
class MLPVelocity(nn.Module):
    def __init__(self, dim, hidden_sizes=[128, 128, 128], output_dim=None, input_dim=None):
        super().__init__()
        output_dim = output_dim or dim
        input_dim = input_dim or dim + 1
        self.mlp = MLP([input_dim, *hidden_sizes, output_dim])

    def forward(self, *args):
        return self.mlp(torch.cat(args, dim=-1))


# Model for velocity field v(x,t) with label conditioning
class MLPVelocityConditioned(nn.Module):
    def __init__(self, dim, hidden_sizes=[128, 128, 128], output_dim=None, label_dim=1):
        super().__init__()
        output_dim = output_dim or dim
        self.label_dim = label_dim
        self.mlp = MLP([dim + 1 + label_dim, *hidden_sizes, output_dim])

    def forward(self, x, t, labels=None):
        t = t.squeeze().view(t.shape[0], -1)
        x = x.view(x.shape[0], -1)
        if labels is not None:
            labels = labels.float().view(labels.shape[0], -1)
            x_t = torch.cat((x, t, labels), dim=1)
        else:
            x_t = torch.cat((x, t), dim=1)
        return self.mlp(x_t)

def get_1d_sincos_pos_embed(
        embed_dim, pos, min_period=1e-3, max_period=10):
    if embed_dim % 2 != 0:
        raise ValueError('embed_dim must be divisible by 2')
    half_dim = embed_dim // 2
    period = torch.logspace(
        math.log(min_period), math.log(max_period), half_dim, base=math.e,
        dtype=pos.dtype, device=pos.device)
    out = pos.unsqueeze(-1) * (2 * math.pi / period)
    emb = torch.cat(
        [torch.sin(out), torch.cos(out)],
        dim=-1)
    return emb


def get_2d_sincos_pos_embed(
        embed_dim, crd, min_period=1e-3, max_period=10):
    """
    Args:
        embed_dim (int)
        crd (torch.Tensor): Shape (bs, 2)
    """
    if embed_dim % 2 != 0:
        raise ValueError('embed_dim must be divisible by 2')
    bs = crd.size(0)
    emb = get_1d_sincos_pos_embed(
        embed_dim // 2, crd.flatten(), min_period, max_period)  # (bs * 2, embed_dim // 2)
    return emb.reshape(bs, embed_dim)


class SinCos2DPosEmbed(nn.Module):
    def __init__(self, num_channels=256, min_period=1e-3, max_period=10):
        super().__init__()
        self.num_channels = num_channels
        self.min_period = min_period
        self.max_period = max_period

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (torch.Tensor): Shape (B, 2)

        Returns:
            torch.Tensor: Shape (B, num_channels)
        """
        return get_2d_sincos_pos_embed(self.num_channels, hidden_states, self.min_period, self.max_period)


class GMFlowMLP2DDenoiser(nn.Module):
    def __init__(
            self,
            num_gaussians=32,
            pos_min_period=5e-3,
            pos_max_period=50,
            embed_dim=256,
            hidden_dim=512,
            constant_logstd=None,
            num_layers=5):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.constant_logstd = constant_logstd
        


        hidden_dim = 128 * 3
        self.vnetd = VNetD(data_dim=2, depth=1, hidden_num=hidden_dim)
        self.out_means = nn.Linear(hidden_dim, num_gaussians * 2)
        self.out_logweights = nn.Linear(hidden_dim, num_gaussians)
        if constant_logstd is None:
            self.out_logstds = nn.Linear(hidden_dim, 1)


        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out_logweights.weight)
        if self.constant_logstd is None:
            nn.init.zeros_(self.out_logstds.weight)

    def forward(self, hidden_states, timestep):
        
        shape = hidden_states.shape
        bs = shape[0]
        extra_dims = shape[2:]

        # t_emb = self.time_proj(timestep).to(hidden_states)
        # pos_emb = self.pos_emb(hidden_states.reshape(bs, 2)).to(hidden_states)
        # embeddings = torch.cat([t_emb, pos_emb], dim=-1)
        t = self.vnetd.time_mlp(timestep)
        xt = self.vnetd.data_mlp(hidden_states)
        
        feat = torch.cat([xt, t], dim=1)   # N x 2*D*dim

        feat = self.vnetd.fc1(feat)
        feat = self.vnetd.act(feat)
        feat = self.vnetd.fc2(feat)
        feat = self.vnetd.act(feat)

        means = self.out_means(feat).reshape(bs, self.num_gaussians, 2, *extra_dims)
        logweights = self.out_logweights(feat).log_softmax(dim=-1).reshape(bs, self.num_gaussians, 1, *extra_dims)
        if self.constant_logstd is None:
            logstds = self.out_logstds(feat).reshape(bs, 1, 1, *extra_dims)
        else:
            logstds = torch.full(
                (bs, 1, 1, *extra_dims), self.constant_logstd,
                dtype=hidden_states.dtype, device=hidden_states.device)
        return dict(
            means=means,
            logweights=logweights,
            logstds=logstds)

class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
        assert(self.dim%2==0)

    def forward(self,x):
        device = x.device
        half_dim = self.dim//2
        emb = math.log(10000)/(half_dim-1)
        emb = torch.exp(torch.arange(half_dim,device=device)*(-emb))
        if x.dim()==1:
            emb = x[...,None]*emb[None,:]
        elif x.dim()==2:
            emb = x[...,None]*emb[None,None,:]
        elif x.dim()==3:
            emb = x[...,None]*emb[None,None,None,:]
        else:
            assert(False)
        emb = torch.cat((emb.sin(),emb.cos()),dim=-1)
        return emb

class VNetD(torch.nn.Module):
    def __init__(self, data_dim=2, depth=2, hidden_num=128, output_dim=None):
        super().__init__()
        dim = self.dim = 64

        self.time_mlp = torch.nn.Sequential(
            SinusoidalPosEmb(dim),
            torch.nn.Linear(dim,dim),
            torch.nn.Flatten(),
            torch.nn.GELU(),
            torch.nn.Linear(depth*dim,depth*dim),
            # torch.nn.LayerNorm(dim),
        )

        self.data_mlp = torch.nn.Sequential(
            SinusoidalPosEmb(dim),
            torch.nn.Linear(dim,dim),
            torch.nn.Flatten(),
            torch.nn.GELU(),
            torch.nn.Linear(data_dim*depth*dim,depth*dim),
            # torch.nn.LayerNorm(dim),
        )

        self.fc1 = torch.nn.Linear(2*depth*dim, 2*depth*hidden_num, bias=True)
        self.fc2 = torch.nn.Linear(2*depth*hidden_num, depth*hidden_num, bias=True)
        if output_dim is None:
            self.fc3 = torch.nn.Linear(depth*hidden_num, data_dim, bias=True)
        else:
            self.fc3 = torch.nn.Linear(depth*hidden_num, output_dim, bias=True)
        self.act = torch.nn.GELU()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, xt, t, *args):
        """
        xt: (batch_size, depth, data_dim)
        t: (batch_size, depth)
        """
        if len(args) > 0:
            xt = torch.cat([xt, *args], dim=-1)
        t = self.time_mlp(t)
        xt = self.data_mlp(xt)
        
        x = torch.cat([xt, t], dim=1)   # N x 2*D*dim
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        return x

