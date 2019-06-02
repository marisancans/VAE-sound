import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn

class Encoder(torch.nn.Module):
    def __init__(self, args):    
        super().__init__()

        self.init_densenet()
        self.args = args
       
        self.mu = torch.nn.Linear(in_features=self.densenet_out, out_features=self.args.z_size)
        self.sigma = torch.nn.Linear(in_features=self.densenet_out, out_features=self.args.z_size)

    def init_densenet(self):
        pretrained_model = models.densenet161(pretrained=True)
        features_pretrained = next(iter(pretrained_model.children()))

        self.seq = torch.nn.Sequential()
        
        for name, module in features_pretrained.named_children():
            if name == 'norm5':
                self.densenet_out = module.num_features
            self.seq.add_module(name, module)

        self.seq.add_module('avg_pool', torch.nn.AdaptiveAvgPool2d(output_size=1))


    def forward(self, x):
        x = self.seq(x)
        bs = x.shape[0]
        features = x.shape[1]
        x = x.view(bs, features)
       

        # VAE
        z_mu = self.mu.forward(x)
        z_sigma = self.sigma.forward(x)
        eps = torch.randn(bs, self.args.z_size).to(self.args.device) * z_sigma + z_mu # Sampling epsilon from normal distributions
        
        
        z_vector = z_mu + z_sigma * eps # z ~ Q(z|X)
        return z_vector, z_mu, z_sigma

    
class Decoder(torch.nn.Module):
    def __init__(self, args):    
        super().__init__()

        self.args = args
        in_channels = self.args.z_size

        orig_height = self.args.image_size
        orig_width = self.args.image_size

        h_out = 0
        w_out = 0

        h_in = 1
        w_in = 1

        stride = (2, 2)
        kernel = (3, 3)

        self.seq = nn.Sequential()
        i = 0

        while True:
            h_out = stride[0] * (h_in - 1) + kernel[0]
            w_out = stride[1] * (w_in - 1) + kernel[1]
            h_in = h_out
            w_in = w_out

            is_last_layer = h_out == orig_height and w_out == orig_width

            if is_last_layer:
                out_channels = 3
            else:
                if i == 0:
                    out_channels = in_channels * 100 // 2
                else:
                    out_channels = in_channels // 2
                

            self.seq.add_module(f'deconv_{i}', nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride))
            
            
            in_channels = out_channels

            if not is_last_layer:
                self.seq.add_module(f'relu_{i}', nn.ReLU())
                self.seq.add_module (f'bn_{i}', nn.BatchNorm2d(num_features=in_channels))

            print(f'layer: {i} | h_out: {h_out} | w_out: {h_out} | c_in: {in_channels} --> c_out: {out_channels}')

            if is_last_layer:
                break

            i += 1
            
            
    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x = self.seq(x)
        x = torch.sigmoid(x)

        return x