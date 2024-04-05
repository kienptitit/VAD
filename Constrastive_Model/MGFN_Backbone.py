import torch
from torch import nn, einsum
from einops import rearrange


def exists(val):
    return val is not None


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


def FeedForward(dim, repe=4, dropout=0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv1d(dim, dim * repe, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv1d(dim * repe, dim, 1)
    )


class GLANCE(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            dim_head=64,
            dropout=0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)
        self.attn = 0

    def forward(self, x):
        x = self.norm(x)
        shape, h = x.shape, self.heads
        x = rearrange(x, 'b c ... -> b c (...)')
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        self.attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', self.attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n', h=h)
        out = self.to_out(out)

        return out.view(*shape)


class FOCUS(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            dim_head=64,
            local_aggr_kernel=5
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = nn.BatchNorm1d(dim)
        self.to_v = nn.Conv1d(dim, inner_dim, 1, bias=False)
        self.rel_pos = nn.Conv1d(heads, heads, local_aggr_kernel, padding=local_aggr_kernel // 2, groups=heads)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)  # (b*crop,c,t)
        b, c, *_, h = *x.shape, self.heads
        v = self.to_v(x)  # (b*crop,c,t)
        v = rearrange(v, 'b (c h) ... -> (b c) h ...', h=h)  # (b*ten*64,c/64,32)
        out = self.rel_pos(v)
        out = rearrange(out, '(b c) h ... -> b (c h) ...', b=b)
        return self.to_out(out)


class Backbone(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            heads,
            mgfn_type='gb',
            kernel=5,
            dim_headnumber=64,
            ff_repe=4,
            dropout=0.,
            attention_dropout=0.
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            if mgfn_type == 'fb':
                attention = FOCUS(dim, heads=heads, dim_head=dim_headnumber, local_aggr_kernel=kernel)
            elif mgfn_type == 'gb':
                attention = GLANCE(dim, heads=heads, dim_head=dim_headnumber, dropout=attention_dropout)
            else:
                raise ValueError('unknown mhsa_type')

            self.layers.append(nn.ModuleList([
                nn.Conv1d(dim, dim, 3, padding=1),
                attention,
                FeedForward(dim, repe=ff_repe, dropout=dropout),
            ]))

    def forward(self, x):
        for scc, attention, ff in self.layers:
            x = scc(x) + x
            x = attention(x) + x
            x = ff(x) + x

        return x


class mgfn(nn.Module):
    def __init__(
            self,
            *,
            classes=0,
            dims=(256, 512, 1024),
            depths=(3, 3, 2),
            mgfn_types=('gb', 'fb', 'fb'),
            lokernel=5,
            channels=1024,
            ff_repe=4,
            dim_head=64,
            dropout=0.,
            attention_dropout=0.,
            batch_size=16,
            dropout_rate=0.7
    ):
        super().__init__()
        init_dim, *_, last_dim = dims
        self.to_tokens = nn.Conv1d(channels, init_dim, kernel_size=3, stride=1, padding=1)

        mgfn_types = tuple(map(lambda t: t.lower(), mgfn_types))

        self.stages = nn.ModuleList([])

        for ind, (depth, mgfn_types) in enumerate(zip(depths, mgfn_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]
            heads = stage_dim // dim_head

            self.stages.append(nn.ModuleList([
                Backbone(
                    dim=stage_dim,
                    depth=depth,
                    heads=heads,
                    mgfn_type=mgfn_types,
                    ff_repe=ff_repe,
                    dropout=dropout,
                    attention_dropout=attention_dropout
                ),
                nn.Sequential(
                    LayerNorm(stage_dim),
                    nn.Conv1d(stage_dim, dims[ind + 1], 1, stride=1),
                ) if not is_last else None
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(last_dim)
        )
        self.batch_size = batch_size
        self.fc = nn.Linear(last_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(dropout_rate)

        self.to_mag = nn.Conv1d(1, init_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, video):
        k = 3
        bs, ncrops, t, c = video.size()  # 32,10,32,1025
        x = video.view(bs * ncrops, t, c).permute(0, 2, 1)  # 320,1025,32
        x_f = x[:, :1024, :]
        x_m = x[:, 1024:, :]
        x_f = self.to_tokens(x_f)  # 320,64,32
        x_m = self.to_mag(x_m)  # 320,64,32
        x_f = x_f + 0.1 * x_m
        for backbone, conv in self.stages:
            x_f = backbone(x_f)
            if exists(conv):
                x_f = conv(x_f)

        x_f = x_f.permute(0, 2, 1).reshape(bs, ncrops, t, -1)
        return x_f


if __name__ == '__main__':
    model = mgfn()
    inp = torch.randn(16, 1, 32, 1025)
    print(model(inp).shape)
