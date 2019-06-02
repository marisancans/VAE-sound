import cv2, torch
import numpy as np
import random as random
import multiprocessing
from os import getpid
import argparse


from models import Encoder, Decoder


def arg_to_bool(x): return str(x).lower() == 'true'
parser = argparse.ArgumentParser()
parser.add_argument('-debug', default=True, type=arg_to_bool)
parser.add_argument('-device', default='cpu')
parser.add_argument('-z_size', default=2, type=int)
parser.add_argument('-image_size', default=63, type=int, help='Aviable sizes = 3, 7, 15, 31, 63, 127, 255, 511')
parser.add_argument('-show', default=True, type=arg_to_bool)
args = parser.parse_args()

print('Using device:', args.device)

encoder = Encoder(args).to(args.device)
decoder = Decoder(args).to(args.device)

def rnd_color():
    f = lambda : random.randint(0, 255)
    return f(), f(), f()

def create_images(size, buffer_size):
    buffer = []

    for nth_img in range(buffer_size):
        r, g, b = rnd_color()
            
        img = np.ones((size, size, 3), np.uint8)
        img[:] = 255, 0, 0

        radius = random.randint(int(size/5), int(size/2))
        rnd_x = random.randint(0, size)
        cv2.circle(img, (int(size/2), rnd_x), radius, (0, 0, r), -1)

        buffer.append(img)

        if (nth_img + 1)  % (buffer_size // 10) == 0:
            print(f'{((nth_img + 1)/buffer_size) * 100}%')
    
    return buffer




###### PARAMS
LR = 0.01
batch_size = 32
buffer_size = 1000
BETA = 250

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, LR)
loss_fn = torch.nn.L1Loss()

def show(truth_t, pred_t):
    img_t = torch.cat((truth_t, pred_t), dim=2)
    img_t = img_t.permute(1, 2, 0)
   
    img = img_t.cpu().detach().numpy()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(1)

def save(encoder, decoder):
    torch.save()


print(f'Generating samples...')
buffer = create_images(args.image_size, buffer_size)

print('Movng to device...')
buffer_t = torch.FloatTensor(buffer).to(args.device)
buffer_t = buffer_t.permute(0, 3, 1, 2) # (B, H, W, C) -->  (B, C, H, W)


for i in range(1000000):
    rnd_idxs = random.sample(range(0, len(buffer_t)), batch_size)
    batch_t = buffer_t[rnd_idxs]

    z_vector, z_mu, z_sigma = encoder.forward(batch_t)
    decoded = decoder.forward(z_vector)
    
    loss_recunstruction = loss_fn(decoded, batch_t)

    loss_kl = BETA * 0.5 * torch.mean(1.0 + torch.log(z_sigma**2) - z_mu**2 - z_sigma**2)
    loss = loss_recunstruction + loss_kl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'{float(loss_recunstruction):.3}  {float(loss_kl):.3}')
    if args.show:
        show(batch_t[0], decoded[0])

    




    x = 1


if __name__ == '__main__':
    main()