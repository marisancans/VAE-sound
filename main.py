import cv2, torch, os, datetime
import numpy as np

import multiprocessing
from os import getpid
import argparse

from models import Encoder, Decoder, CustomDataset


def arg_to_bool(x): return str(x).lower() == 'true'
parser = argparse.ArgumentParser()
parser.add_argument('-debug', default=True, type=arg_to_bool)
parser.add_argument('-device', default='cpu')
parser.add_argument('-z_size', default=2, type=int)

parser.add_argument('-lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-epoch', default=10000, type=int)

parser.add_argument('-buffer_size', default=100, type=int, help='How many images are generated and stored on device')
parser.add_argument('-beta', default=1.0, type=float, help='Beta hyperparameter in beta VAE')
parser.add_argument('-image_size', default=63, type=int, help='Aviable sizes = 3, 7, 15, 31, 63, 127, 255, 511')
parser.add_argument('-show', default=True, type=arg_to_bool)

parser.add_argument('-load', default='', help='Specify save folder name and the last epoch will be tanken')
args = parser.parse_args()

print('Using device:', args.device)

encoder = Encoder(args).to(args.device)
decoder = Decoder(args).to(args.device)

now = datetime.datetime.now()
dir_name = now.strftime("%B_%d_at_%H_%M_%p")
save_dir = './save/' + dir_name

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, args.lr)
reconstruction_loss_fn = torch.nn.BCELoss()

def show(truth_t, pred_t):
    img_t = torch.cat((truth_t, pred_t), dim=2)
    img_t = img_t.permute(1, 2, 0)
   
    img = img_t.cpu().detach().numpy()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(1)

def save(encoder, decoder, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(encoder.state_dict(), save_dir + f'/{epoch}_encoder.pth')
    torch.save(decoder.state_dict(), save_dir + f'/{epoch}_decoder.pth')

def load(path):
    if not os.path.exists(path):
        print('Load path doesnt exist!')
    else:
        files = [f for f in os.listdir(path) if os.isfile(os.path.join(path, f))].sort()
    
    return files



dataset = CustomDataset(args)
dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)



for epoch in range(1, args.epoch):
    epoch_loss_rec = []
    epoch_loss_kl = []

    for batch_t in dataset_loader:
        z_vector, z_mu, z_sigma = encoder.forward(batch_t)
        decoded = decoder.forward(z_vector)

        if args.show:
            show(batch_t[0], decoded[0])
        
        if args.load:
            continue

        loss_recunstruction = reconstruction_loss_fn(decoded, batch_t)

        loss_kl = args.beta * 0.5 * (1.0 + torch.log(z_sigma**2) - z_mu**2 - z_sigma**2)
        loss_kl = torch.mean(loss_kl)
        loss = loss_recunstruction - loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss_kl.append(float(loss_kl))
        epoch_loss_rec.append(float(loss_recunstruction))



    epoch_loss_kl = np.average(np.array(epoch_loss_kl))
    epoch_loss_rec = np.average(np.array(epoch_loss_rec))
    print(f'epoch: {epoch}   |    kl: {epoch_loss_kl:.4}    |    rec: {epoch_loss_rec:.4}')        

    if epoch % 50 == 0:
        save(encoder, decoder, epoch)


if __name__ == '__main__':
    main()