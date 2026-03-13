import torch
from torch import nn

class ConvBlock(nn.Module):
    """
        Conv2D Block
            Args:
                x: (N, C_in, H, W)
            Returns:
                y: (N, C_out, H, W)
    """

    def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
                    stride=1, padding='same', dilation=1, groups=1, bias=True, gn=False, gn_groups=8):

        if padding == 'same':
            padding = kernel_size // 2 * dilation

        super(ConvBlock, self).__init__(in_channels, out_channels, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation,
                                            groups=groups, bias=bias)

        self.activation_fn = nn.SiLU() if activation_fn else None
        self.group_norm = nn.GroupNorm(gn_groups, out_channels) if gn else None

    def forward(self, x, time_embedding=None, residual=False):

        if residual:
            # timestep embedding was only applied to residual blocks of U-Net
            x = x + time_embedding
            y = x
            x = super(ConvBlock, self).forward(x)
            y = y + x
        else:
            y = super(ConvBlock, self).forward(x)
        y = self.group_norm(y) if self.group_norm is not None else y
        y = self.activation_fn(y) if self.activation_fn is not None else y

        return y

class ConditionalFlowMatching(nn.Module):
    def __init__(self, image_resolution, hidden_dims=[256, 256], sigma_min=0.):
        super(ConditionalFlowMatching, self).__init__()

        _, _, img_C = image_resolution

        self.in_project = ConvBlock(img_C, hidden_dims[0], kernel_size=7)

        self.time_project = nn.Sequential(
                                 ConvBlock(1, hidden_dims[0], kernel_size=1, activation_fn=True),
                                 ConvBlock(hidden_dims[0], hidden_dims[1], kernel_size=1))

        self.convs = nn.ModuleList([ConvBlock(in_channels=hidden_dims[0], out_channels=hidden_dims[0], kernel_size=3)])

        for idx in range(1, len(hidden_dims)):
            self.convs.append(ConvBlock(hidden_dims[idx-1], hidden_dims[idx], kernel_size=3, dilation=3**((idx-1)//2),
                                                    activation_fn=True, gn=True, gn_groups=8))

        self.out_project = ConvBlock(hidden_dims[-1], out_channels=img_C, kernel_size=3)
        self.sigma_min = sigma_min

    def get_velocity(self, x_0, x_1):
        """
            Target velocity (or vector field).
                This vector field generates conditional probabilistic path
                    that maps a sampled noise from Gaussian normal to target data distribution
        """
        return (x_1 - (1-self.sigma_min) * x_0)

    def interpolate(self, x_0, x_1, t):
        """
            Conditional flow, psai_t(x_0) that corresponds to u_t(x | x_1).
                Since we use Optimal Transport conditional VFs,
                    we simply use linear interpolation between noise(x_0) and target sample(x_1).
        """
        return (1 - (1-self.sigma_min) * t) *  x_0 + t * x_1


    def forward(self, x, t):
        """
            Estimate vector field given x_t and time embedding.
        """

        time_embedding = self.time_project(t)
        y = self.in_project(x)

        for i in range(len(self.convs)):
            y = self.convs[i](y, time_embedding, residual=True)

        v_t = self.out_project(y)
        return v_t

    @torch.no_grad()
    def sample(self, t_steps, shape, DEVICE):

        """
            move x_0 (located in noise distribution) to x_1 (located in target data distribution)
        """

        B, C, W, H = shape

        x_0 = torch.randn(size=shape, device=DEVICE)
        t_vals = torch.linspace(0, 1, t_steps, device=DEVICE)
        delta = 1.0 / (t_steps - 1)

        x_1_hat = x_0
        for i in range(t_steps - 1):
            t_cur = t_vals[i].view(-1, 1, 1, 1)
            velocity_pred = self(x_1_hat, t_cur)
            x_1_hat = x_1_hat + velocity_pred * delta

        return x_1_hat