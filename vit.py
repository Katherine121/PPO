import timm.models
import torch
from torch import nn
import torchvision.models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from thop import profile


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


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


class Attention(nn.Module):
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
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(0, depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size,
                 dim, depth, heads, dim_head, mlp_dim,
                 dropout=0., emb_dropout=0.,
                 pool='cls', channels=3):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.height = image_height

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.extractor_dim = dim
        self.mlp_head = nn.Sequential(
            nn.Linear(2 * self.extractor_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Tanh()
        )

    def forward(self, img):
        if len(img.shape) == 4:
            x = self.to_patch_embedding(img)
            b, n, _ = x.shape

            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)

            x = self.transformer(x)

            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            img = x.reshape(-1, 2 * self.extractor_dim)

        else:
            b = img.size(0)
            img = img.reshape(b * 2, 3, self.height, self.height)

            x = self.to_patch_embedding(img)
            _, n, _ = x.shape

            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b * 2)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)

            x = self.transformer(x)

            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            img = x.reshape(b, 2 * self.extractor_dim)

        return self.mlp_head(img)


class SURFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor_dim = 640
        self.mlp_head = nn.Sequential(
            nn.Linear(2 * self.extractor_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Tanh()
        )

    def forward(self, img):
        # input surf features
        if len(img.shape) == 2:
            img = img.reshape(-1, 2 * self.extractor_dim)
        else:
            b = img.size(0)
            img = img.reshape(b, 2 * self.extractor_dim)

        return self.mlp_head(img)


class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.mobilenet_v3_small()
        self.backbone.classifier = nn.Identity()
        self.extractor_dim = 576
        self.mlp_head = nn.Sequential(
            nn.Linear(2 * self.extractor_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Tanh()
        )

    def forward(self, img):
        if len(img.shape) == 4:
            x = self.backbone(img)
            img = x.reshape(-1, 2 * self.extractor_dim)
        else:
            b = img.size(0)
            x = self.backbone(img.view(b * 2, 3, 256, 256))
            img = x.reshape(b, 2 * self.extractor_dim)

        return self.mlp_head(img)


class Actor(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.actor = ViT(image_size=256,
                         patch_size=32,
                         dim=64,
                         depth=4,
                         heads=2,
                         dim_head=64,
                         mlp_dim=128,
                         dropout=0.,
                         emb_dropout=0.)
        # self.actor = MobileNet()
        # print(self.actor)
        # flops 51.21M, params 0.41M
        flops, params = profile(self.actor, (torch.randn(2, 3, 256, 256), ))
        print('flops: ', flops, 'params: ', params)
        print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
        if checkpoint_path is not None:
            state_dict1 = torch.load(checkpoint_path)
            state_dict1 = state_dict1["state_dict"]
            state_dict2 = {}
            for name, param in state_dict1.items():
                state_dict2[name[len("module:"):]] = param
            self.actor.load_state_dict(state_dict2)

    def forward(self, img):
        return self.actor(img)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = ViT(image_size=256,
                          patch_size=32,
                          dim=64,
                          depth=4,
                          heads=2,
                          dim_head=64,
                          mlp_dim=128,
                          dropout=0.,
                          emb_dropout=0.)
        # self.critic = MobileNet()
        self.critic.mlp_head = nn.Sequential(
            nn.Linear(2 * self.critic.extractor_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, img):
        return self.critic(img)
