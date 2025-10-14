import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.embed_size = 128  # embed_size
        self.hidden_size = 256  # hidden_size
        self.pre_length = configs.pred_len
        self.feature_size = configs.enc_in  # channels
        self.seq_length = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        # embedding
        self.embeddings = nn.Parameter(torch.randn(1, 1, 1, self.embed_size))

        # weights initialization (Xavier)
        self.r1 = nn.Parameter(torch.empty(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(torch.empty(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(torch.empty(self.embed_size))
        self.ib1 = nn.Parameter(torch.empty(self.embed_size))
        self.r2 = nn.Parameter(torch.empty(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(torch.empty(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(torch.empty(self.embed_size))
        self.ib2 = nn.Parameter(torch.empty(self.embed_size))

        self._reset_parameters()

        # normalization & dropout
        self.norm = nn.LayerNorm(self.embed_size)
        self.drop = nn.Dropout(0.2)

        # fully connected projection (كما في الورقة)
        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.pre_length)
        )

    def _reset_parameters(self):
        for w in [self.r1, self.i1, self.r2, self.i2]:
            nn.init.xavier_uniform_(w)
        for b in [self.rb1, self.ib1, self.rb2, self.ib2]:
            nn.init.zeros_(b)

    # dimension extension
    def tokenEmb(self, x):
        # x: [B, T, N]
        x = x.permute(0, 2, 1)  # [B, N, T]
        x = x.unsqueeze(-1)     # [B, N, T, 1]
        return x * self.embeddings  # [B, N, T, D]

    # frequency-domain MLP
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        # x: complex tensor [..., D]
        xr = x.real
        xi = x.imag

        o1_real = F.relu(
            torch.einsum('...d,dk->...k', xr, r)
            - torch.einsum('...d,dk->...k', xi, i)
            + rb
        )

        o1_imag = F.relu(
            torch.einsum('...d,dk->...k', xi, r)
            + torch.einsum('...d,dk->...k', xr, i)
            + ib
        )

        y = torch.complex(o1_real, o1_imag)
        # sparsity regularization
        stacked = torch.stack([y.real, y.imag], dim=-1)
        shrunk = F.softshrink(stacked, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(shrunk)
        return y

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # x: [B, N, T, D]
        x = x.permute(0, 2, 1, 3)  # [B, T, N, D]
        x = self.norm(x)
        x = self.drop(x)

        # full FFT (not rfft)
        Xf = torch.fft.fft(x, dim=2, norm='ortho')
        Yf = self.FreMLP(B, L, N, Xf, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.ifft(Yf, n=self.feature_size, dim=2, norm='ortho').real

        x = x.permute(0, 2, 1, 3)  # [B, N, T, D]
        return x

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # x: [B, N, T, D]
        x = self.norm(x)
        x = self.drop(x)

        Xf = torch.fft.fft(x, dim=2, norm='ortho')
        Yf = self.FreMLP(B, N, L, Xf, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.ifft(Yf, n=self.seq_length, dim=2, norm='ortho').real

        return x

    def forward(self, x):
        # x: [B, T, N]
        B, T, N = x.shape
        x = self.tokenEmb(x)  # [B, N, T, D]
        bias = x.clone()

        if self.channel_independence == '1':
            x = x + self.MLP_channel(x, B, N, T)
        x = x + self.MLP_temporal(x, B, N, T)

        # residual
        x = x + bias

        # projection
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)  # [B, pred_len, N]
        return x
