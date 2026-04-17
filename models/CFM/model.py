import torch
from torch import nn


class ConvBlock(nn.Conv2d):
    """
    A simple Conv2d block with optional GroupNorm + SiLU.

    Note that a plain convolution block is just a linear operation. Stacking only linear layers collapses into another linear function.
    That's why we need to add non-linearity (e.g. SiLU) and normalization layers (e.g. GroupNorm) to make it a non-linear operation.
    """

    def __init__(
        self,
        in_channels,            # number of input feature maps, e.g. 3 for RGB images
        out_channels,           # number of output feature maps, e.g. 64 for 64 feature maps
        kernel_size,            # size of the convolutional kernel, e.g. 3 for 3x3 kernel
        activation_fn=None,     # boolean for an activation function
        stride=1,               # stride of the convolutional kernel (controls downsampling or upsampling)
        padding="same",         # padding of the convolutional kernel
        dilation=1,
        groups=1,               # number of groups for the convolutional kernel
        bias=True,              # boolean for the bias in the convolutional kernel
        gn=False,               # boolean for the GroupNorm (type of normalization)
        gn_groups=8,            # number of groups for the GroupNorm
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
        self.group_norm = nn.GroupNorm(gn_groups, out_channels) if gn else None
        self.activation_fn = nn.SiLU() if activation_fn else None

    def forward(self, x, time_embedding=None, residual=False):
        '''
        We use the convolution to extract features, we use normalization to stabilize the training, activation to make them expressive,
        time embedding to tell the network what to do at each time step, residual connections to make learning easier and more stable,
        and broadcast-add to inject global per-channel bias across spatial maps.
        '''
        if residual:
            '''
            In the case of residual connections, we wish to make the network behave differently depending on time step t (e.g. noise level).
            
            We inject the time information before the convolution so that the convolution block acts as a function of both the input x and the time step t.
            This embedding allows the filters to react differently to different noise levels.
            It also allows the conditioning to happen inside a feature space extraction, not just after (because if it were only after, the convolution would be a linear function).
            '''
            x = x + time_embedding
            y = x
            x = super().forward(x)
            y = y + x
        else:
            y = super().forward(x)
        y = self.group_norm(y) if self.group_norm is not None else y            # first we clean up and normalize the output
        y = self.activation_fn(y) if self.activation_fn is not None else y      # then we apply the non-linearity to enhance expressivity
        return y


class CFMModel(nn.Module):
    """
    Conditional Flow Matching model implementation using a simple convolutional network (CNN) with time embedding.
    """

    def __init__(self, image_resolution, hidden_dims=None, sigma_min=0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]    

        _, _, img_c = image_resolution      # expected format is (height, width, channels), e.g. (256, 256, 3) for 256x256 RGB images
        
        
        self.in_project = ConvBlock(img_c, hidden_dims[0], kernel_size=7)   # first convolution layer, no dilation
        # it maps the raw image to the feature space
        # kernel size 7 is a common choice for image processing tasks; large receptive field early helps capture global information


        self.time_project = nn.Sequential(  # time embedding projection
            ConvBlock(1, hidden_dims[0], kernel_size=1, activation_fn=True),  # this is a 1x1 convolution layer (per-channel linear projection)
            # it expands time embedding into the feature space, meaning that each channel in the feature space is a linear combination of the time embedding
            # shape: (B, hidden_dims[0], 1, 1)
            ConvBlock(hidden_dims[0], hidden_dims[1], kernel_size=1),         # again, a 1x1 convolution layer, but without activation function
            # shape: (B, hidden_dims[1], 1, 1)
            # time is now a feature map-like tensor
        )
        # time t is typically of shape (B, 1), where B is the batch size, but for Conv2D we need to expand it to (B, 1, 1, 1);
        # so that it can be broadcasted across the spatial dimensions.

        # Here, instead of MLP, we use a sequence of conv layers to embed the time into the feature space.
        # This is a common practice in diffusion models to keep the model architecture simple and efficient.


        self.convs = nn.ModuleList(  # a list of layers that PyTorch will track and optimize
            [ConvBlock(in_channels=hidden_dims[0], out_channels=hidden_dims[0], kernel_size=3)]
            # the first of the hidden layers, keeps the same channel size, does basic feature processing
        )
        for idx in range(1, len(hidden_dims)):  # for each of the hidden layers
            self.convs.append(
                ConvBlock(
                    hidden_dims[idx - 1],       # output channels from the previous layer
                    hidden_dims[idx],           # input channels for the current layer, connecting to the previous layer's output
                    kernel_size=3,
                    dilation=3 ** ((idx - 1) // 2),     # dilation for the conv kernel, controls the receptive field size; grows like 1, 1, 3, 3, 9, 9, ...
                    activation_fn=True,                 # activation function for the current layer
                    gn=True,                            # boolean for the GroupNorm
                    gn_groups=8,                        # number of groups for the GroupNorm
                )
            )
            # the current layer's output is connected to the next layer's input, and so on, creating a chain of layers
            # a cool trick is that the receptive field grows exponentially, without downsampling (downsampling would mean losing information by averaging out the features)

        self.out_project = ConvBlock(hidden_dims[-1], out_channels=img_c, kernel_size=3)  # final convolution layer
        # it maps the feature space back to the image space
        # shape: (B, img_c, H, W), which produces v_theta(x_t, t), which is the same shape as the input image x_t


        self.sigma_min = sigma_min     # a hyperparameter that controls minimum noise level in FM models

    def get_velocity(self, x_0, x_1):
        # this is the definition of the conditional variable u_t(x|z) = x_1 - x_0, 
        # with the assumption that q(z) = q(x_0)*q(x_1) (I-CFM)
        return x_1 - (1 - self.sigma_min) * x_0

    def interpolate(self, x_0, x_1, t):
        '''
        This is the definition of the probability path mean:  μ_t(z) = t*x_1 + (1-t)*x_0:
        the output is the linear interpolation between x_0 ~ p_0 (e.g. x_0 ~ N(0,I))and target data sample x_1 ~ p_1 (e.g. x_1 ~ MNIST) at time t.

        We use this as target instead of simulating actual μ_t.
        Note: once sigma is ~ 0, this means that the Gaussian of our probability path is almost a Dirac delta function.
        '''
        
        return (1 - (1 - self.sigma_min) * t) * x_0 + t * x_1

    def forward(self, x, t):
        time_embedding = self.time_project(t)       # calling the time embedding projection
        y = self.in_project(x)                      # the first convolution layer
        for block in self.convs:
            y = block(y, time_embedding, residual=True)  # calling the hidden convolution layers
        return self.out_project(y)                       # the final convolution layer

    @torch.no_grad()
    def sample(self, t_steps, shape, device):
        '''
        Using the Euler method to (numerically, not symbolically) integrate the ODE, and hence generate a sample.
        '''
        x_0 = torch.randn(size=shape, device=device)  # sample a noise from "the noise distribution" p_0
        t_vals = torch.linspace(0, 1, t_steps, device=device)  # sample t_steps number of t values from 0 to 1 ( t ~ U(0,1))
        delta = 1.0 / max(t_steps - 1, 1)   # literally the time differential dt; an ("infinitesimally") small increment of time

        x_1_hat = x_0
        for i in range(t_steps - 1):
            t_cur = t_vals[i].view(1, 1, 1, 1).expand(shape[0], 1, 1, 1)  # current time t
            velocity_pred = self(x_1_hat, t_cur)  # calling the model to estimate the velocity field; equivalent to `model(x_t, t)`
            x_1_hat = x_1_hat + velocity_pred * delta
        return x_1_hat
