import torch
from torch import nn


class ConvBlock(nn.Conv2d):
    """Simple Conv2d block with optional GroupNorm + SiLU."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation_fn=None,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        bias=True,
        gn=False,
        gn_groups=8,
    ):
        if padding == "same":
            padding = kernel_size // 2 * dilation

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.activation_fn = nn.SiLU() if activation_fn else None
        self.group_norm = nn.GroupNorm(gn_groups, out_channels) if gn else None

    def forward(self, x, r_embedding=None, t_embedding=None, residual=False):
        if residual:
            x = x + t_embedding + r_embedding
            y = x
            x = super().forward(x)
            y = y + x
        else:
            y = super().forward(x)
        y = self.group_norm(y) if self.group_norm is not None else y
        y = self.activation_fn(y) if self.activation_fn is not None else y
        return y


class MeanFlowModel(nn.Module):
    """MeanFlow model used in notebook 02."""

    def __init__(self, image_resolution, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        _, _, img_c = image_resolution

        self.in_project = ConvBlock(img_c, hidden_dims[0], kernel_size=7)
        self.time_project_t = nn.Sequential(
            ConvBlock(1, hidden_dims[0], kernel_size=1, activation_fn=True),
            ConvBlock(hidden_dims[0], hidden_dims[1], kernel_size=1),
        )
        self.time_project_r = nn.Sequential(
            ConvBlock(1, hidden_dims[0], kernel_size=1, activation_fn=True),
            ConvBlock(hidden_dims[0], hidden_dims[1], kernel_size=1),
        )
        self.convs = nn.ModuleList(
            [ConvBlock(in_channels=hidden_dims[0], out_channels=hidden_dims[0], kernel_size=3)]
        )
        for idx in range(1, len(hidden_dims)):
            self.convs.append(
                ConvBlock(
                    hidden_dims[idx - 1],
                    hidden_dims[idx],
                    kernel_size=3,
                    dilation=3 ** ((idx - 1) // 2),
                    activation_fn=True,
                    gn=True,
                    gn_groups=8,
                )
            )
        self.out_project = ConvBlock(hidden_dims[-1], out_channels=img_c, kernel_size=3)

    def get_z_t(self, x, t, e):
        return (1 - t) * x + t * e

    def get_instantaneous_velocity_v(self, e, x):
        return e - x

    def forward(self, z_t, r, t):
        r_embedding = self.time_project_r(r)
        t_embedding = self.time_project_t(t)
        y = self.in_project(z_t)
        for block in self.convs:
            y = block(y, r_embedding, t_embedding, residual=True)
        return self.out_project(y)

    @torch.no_grad()
    def sample(self, shape, n_steps=1, device="cpu"):
        b = shape[0]
        make_time = lambda v: torch.full((b, 1, 1, 1), v, device=device)
        t_list = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
        z = torch.randn(shape, device=device)

        for i in range(n_steps):
            t_i = make_time(float(t_list[i]))
            r_i = make_time(float(t_list[i + 1]))
            u_t = self(z, r=r_i, t=t_i)
            d_t = t_list[i] - t_list[i + 1]
            z = z - u_t * d_t
        return z
