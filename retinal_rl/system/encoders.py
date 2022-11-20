"""
retina_rl library

"""
from torch import nn
from torchvision.transforms import Grayscale

from sample_factory.algorithms.appo.model_utils import register_custom_encoder, EncoderBase, get_obs_shape
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements


### Utils ###


def activator(cfg) -> nn.Module:
    if cfg.activation == "elu":
        return nn.ELU(inplace=True)
    elif cfg.activation == "relu":
        return nn.ReLU(inplace=True)
    elif cfg.activation == "tanh":
        return nn.Tanh()
    elif cfg.activation == "linear":
        return nn.Identity(inplace=True)
    else:
        raise Exception("Unknown activation function")


### Retinal-VVS Model ###


class LindseyEncoderBase(EncoderBase):

    def __init__(self, cfg, obs_space, timing):

        super().__init__(cfg, timing)

        # Saved config variables

        self.kernel_size = cfg.kernel_size
        self.gsbl = cfg.greyscale
        self.encoder_out_size = cfg.hidden_size
        self.activation = activator(cfg)

        # Preparing Network

        self.nl_fc = self.activation

        indm = 3
        if self.gsbl: indm = 1

        nchns = cfg.global_channels
        btlchns = cfg.retinal_bottleneck
        vvsdpth = cfg.vvs_depth
        retstrd = cfg.retinal_stride # only for first conv layer
        krnsz = self.kernel_size

        conv_layers = []

        for i in range(vvsdpth+2): # +2 for the first 'retinal' layers

            if i == 0: # 'bipolar cells' ('global channels')
                conv_layers.extend([nn.Conv2d(indm, nchns, krnsz, stride=retstrd), self.activation])
            elif i == 1: # 'ganglion cells' ('retinal bottleneck')
                conv_layers.extend([nn.Conv2d(nchns, btlchns, krnsz, stride=1), self.activation])
            elif i == 2: # 'V1' ('global channels')
                conv_layers.extend([nn.Conv2d(btlchns, nchns, krnsz, stride=1), self.activation])
            else: # 'vvs layers'
                conv_layers.extend([nn.Conv2d(nchns, nchns, krnsz, stride=1), self.activation])

        # Storing network
        self.conv_head = nn.Sequential(*conv_layers)

        obs_shape = get_obs_shape(obs_space)
        if self.gsbl:
            obs_shape['obs'] = (1, obs_shape['obs'][1], obs_shape['obs'][2])

        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)

        self.fc1 = nn.Linear(self.conv_head_out_size,self.encoder_out_size)

    def forward(self, x):

        # we always work with dictionary observations. Primary observation is available with the key 'obs'
        if self.gsbl:

            gs = Grayscale(num_output_channels=1) # change compared to vanilla Lindsey
            x = gs.forward(x) # change compared to vanilla Lindsey

        x = self.conv_head(x)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.nl_fc(self.fc1(x))
        return x

class LindseyEncoder(LindseyEncoderBase):

    def __init__(self, cfg, obs_space,timing):

        super().__init__(cfg,obs_space,timing)
        self.base_encoder = LindseyEncoderBase(cfg,obs_space,timing)
        print(self)

    def forward(self, obs_dict):
        # we always work with dictionary observations. Primary observation is available with the key 'obs'
        main_obs = obs_dict['obs']

        # forward pass through configurable fully connected blocks immediately after the encoder
        x = self.base_encoder(main_obs)
        return x

def register_encoders():
    register_custom_encoder('lindsey', LindseyEncoder)
