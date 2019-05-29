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
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', DEVICE)

batch_size = 2

# Aviable sizes
# [3 3] [7 7] [15 15] [31 31] [63 63] [127 127] [255 255] [511 511]
WIDTH = 31
HEIGHT = 31

encoder = Encoder().to(DEVICE)
decoder = Decoder(HEIGHT, WIDTH, encoder.out_size).to(DEVICE)

def rnd_color():
    f = lambda : random.random()
    return f(), f(), f()

def create_image(id=0):
    r, g, b = rnd_color()
        
    img = np.ones((HEIGHT, WIDTH, 3), np.uint8)
    img[:] = r, 0, 0

    radius = random.randint(int(WIDTH/3), int(WIDTH/2))
    cv2.circle(img, (int(HEIGHT/2), int(WIDTH/2)), 5, (255, 0, 0), -1)
    
    return img

def make_batch_multithreaded(batch_size):   
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
    batch = pool.map(create_image, range(5))

    return batch

def make_batch_debug(batch_size):
    return [create_image() for _ in range(batch_size)]




###### PARAMS
LR = 0.01
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, LR)
loss_fn = torch.nn.MSELoss()

def show(truth_t, pred_t):
    img = torch.cat((truth_t, pred_t), dim=2)
    img = img.permute(1, 2, 0)
    img = img.cpu().detach().numpy()

    img = truth_t

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(1)

for _ in range(100000):
    if args.debug:
        batch = make_batch_debug(batch_size)
    else:
        batch = make_batch_multithreaded(batch_size)

    batch_t = torch.FloatTensor(batch).to(DEVICE)
    batch_t = batch_t.permute(0, 3, 1, 2) # (B, H, W, C) -->  (B, C, H, W)

    z_vector = encoder.forward(batch_t)
    pred_t = decoder.forward(z_vector)
    loss = loss_fn(pred_t, batch_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(float(loss))

    truth = batch_t[0]
    pred = pred_t[0]

    show(truth, pred)


    x = 1


if __name__ == '__main__':
    main()