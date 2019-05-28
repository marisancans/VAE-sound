import torch
import torch.nn.functional as F
import torchvision.models as models

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
    def __init__(self):    
        super().__init__()

        self.deconv_1 = torch.nn.ConvTranspose2d(in_channels=20, out_channels=10, kernel_size=3)
        self.deconv_2 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=3, kernel_size=3)


    def forward(self, x):
        x = self.deconv_1.forward(x)
        x = F.relu(x)
        x = self.deconv_2.forward(x)
        x = torch.sigmoid(x)

        return x