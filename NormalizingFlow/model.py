import torch
import torch.nn as nn
import numpy as np
import FrEIA.framework as Ff
import FrEIA.modules as Fm


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2 * dims_in), nn.ReLU(), nn.Linear(2 * dims_in, dims_out))


def flow_model(args, in_channels):
    coder = Ff.SequenceINN(in_channels)
    print('Normalizing Flow => Feature Dimension: ', in_channels)
    for k in range(args.coupling_layers):
        coder.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, affine_clamping=args.clamp_alpha,
                     global_affine_type='SOFTPLUS', permute_soft=False)
    return coder


def conditional_flow_model(args, in_channels):
    coder = Ff.SequenceINN(in_channels)
    print('Conditional Normalizing Flow => Feature Dimension: ', in_channels)
    for k in range(args.coupling_layers):  # 8
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(args.pos_embed_dim,), subnet_constructor=subnet_fc,
                     affine_clamping=args.clamp_alpha,
                     global_affine_type='SOFTPLUS', permute_soft=False)
    return coder


def get_flow_model(args, in_channels):
    if args.flow_arch == 'flow_model':
        model = flow_model(args, in_channels)
    elif args.flow_arch == 'conditional_flow_model':
        model = conditional_flow_model(args, in_channels)
    else:
        raise NotImplementedError('{} is not supported Normalizing Flow!'.format(args.flow_arch))

    return model
