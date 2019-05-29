import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn

class Encoder(torch.nn.Module):
    def __init__(self):    
        super().__init__()

        pretrained_model = models.densenet161(pretrained=True)
        features_pretrained = next(iter(pretrained_model.children()))

        self.encoder = torch.nn.Sequential()
        
        for name, module in features_pretrained.named_children():
            if name == 'norm5':
                self.out_size = module.num_features
            self.encoder.add_module(name, module)

        self.encoder.add_module('avg_pool', torch.nn.AdaptiveAvgPool2d(output_size=1))

        # self.encoder_out = torch.nn.Linear(in_features=self.out_size, out_features=self.args.conv_encoder_layer_out)

    def forward(self, x):
        x = self.encoder(x)

        return x

    
class Decoder(torch.nn.Module):
    def __init__(self, orig_height, orig_width, in_channels):    
        super().__init__()

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
                out_channels = int(in_channels // 2)

            self.seq.add_module(f'deconv_{i}', nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride))
            in_channels = out_channels

            if not is_last_layer:
                self.seq.add_module(f'relu_{i}', nn.ReLU())

            print(f'layer: {i} | h_out: {h_out} | w_out: {h_out} | c_in: {in_channels} --> c_out: {out_channels}')
            i += 1

            if is_last_layer:
                break
            
            
    


    def forward(self, x):
        x = self.seq(x)
        x = torch.sigmoid(x)

        return x