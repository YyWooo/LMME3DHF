from torch import nn, einsum
import torch
from einops.layers.torch import Rearrange
from einops import rearrange, repeat


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FSAttention(nn.Module):
    """Factorized Self-Attention"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class FSATransformerEncoder(nn.Module):
    """Factorized Self-Attention Transformer Encoder"""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, channels, nt, nh, nw, dropout=0.,sample_num=8):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.nt = nt
        self.nh = nh
        self.nw = nw

        for _ in range(depth):
            self.layers.append(nn.ModuleList(
                [PreNorm(dim, FSAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                 PreNorm(dim, FSAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                 ]))
        self.temporal_conv = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=[sample_num,3,3], stride=[1,1,1], padding=[0,1,1], bias=True)

    def forward(self, x):

        b = x.shape[0] # [b t h*w c]
        x = torch.flatten(x, start_dim=0, end_dim=1)  # extract spatial tokens from x [b*t h*w d]

        for sp_attn, temp_attn, ff in self.layers:
            sp_attn_x = sp_attn(x) + x  # Spatial attention

            # Reshape tensors for temporal attention
            sp_attn_x = rearrange(sp_attn_x, '(b nt) nhnw d -> (b nhnw) nt d', b=b) # [b*h*w t d]

            temp_attn_x = temp_attn(sp_attn_x) + sp_attn_x  # Temporal attention

            x = ff(temp_attn_x) + temp_attn_x  # MLP # [b*h*w t d]

            # Again reshape tensor for spatial attention
            x = rearrange(temp_attn_x, '(b nhnw) nt d -> (b nt) nhnw d', b=b)# [b*t,h*w,c]

        # Reshape vector to [b, nt*nh*nw, dim]
        # x = rearrange(x, '(b nt) (nh nw) d -> b nt nh nw d', b=b, nh=self.nh)
        # x = x.mean(dim=1)
        x = rearrange(x, '(b nt) (nh nw) d -> b d nt nh nw', b=b, nh=self.nh)
        x = self.temporal_conv(x)
        # x = rearrange(x, 'b nh nw d -> b d nh nw', nh=self.nh).unsqueeze(dim=2)

        return x




class ViViTBackbone(nn.Module):
    """ Model-3 backbone of ViViT """

    def __init__(self, t, h, w, patch_t, patch_h, patch_w,  dim, depth, heads, mlp_dim, dim_head=3,
                 channels=3, mode='tubelet', emb_dropout=0., dropout=0.1, device='cuda',sample_num=8):
        super().__init__()

        assert t % patch_t == 0 and h % patch_h == 0 and w % patch_w == 0, "Video dimensions should be divisible by " \
                                                                           "tubelet size "

        self.T = t
        self.H = h
        self.W = w
        self.channels = channels
        self.t = patch_t
        self.h = patch_h
        self.w = patch_w
        self.mode = mode

        self.nt = self.T // self.t
        self.nh = self.H // self.h
        self.nw = self.W // self.w

        tubelet_dim = self.t * self.h * self.w * channels

        self.to_tubelet_embedding = nn.Sequential(
            Rearrange('b c (t pt) (h ph) (w pw) -> b t (h w) (pt ph pw c)', pt=self.t, ph=self.h, pw=self.w),
            nn.Linear(tubelet_dim, dim)
        )

        # repeat same spatial position encoding temporally
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.nh * self.nw, dim)).repeat(1, self.nt, 1, 1).to(device)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = FSATransformerEncoder(dim, depth, heads, dim_head, mlp_dim, self.channels,
                                                     self.nt, self.nh, self.nw, dropout,sample_num=sample_num)


    def forward(self, x):
        """ x is a video: (b, C, T, H, W) """

        tokens = self.to_tubelet_embedding(x) # torch.Size([1, 8, 169, 1152])

        tokens += self.pos_embedding
        tokens = self.dropout(tokens)

        x = self.transformer(tokens)
        # x = x.mean(dim=1)

        # x = self.to_latent(x)
        return x


if __name__ == '__main__':
    device = torch.device('cpu')
    x = torch.rand(32, 3, 32, 64, 64).to(device)

    vivit = ViViTBackbone(32, 64, 64, 8, 4, 4, 10, 512, 6, 10, 8, model=4).to(device)
    out = vivit(x)
    print(out)
