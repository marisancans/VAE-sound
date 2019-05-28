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


encoder = Encoder().to(DEVICE)
decoder = Decoder().to(DEVICE)

batch_size = 32
WIDTH = 60
HEIGHT = 64


def rnd_color():
    f = lambda : random.randint(0, 255)
    return f(), f(), f()

def create_image(id=0):
    r, g, b = rnd_color()
        
    img = np.ones((HEIGHT, WIDTH, 3), np.uint8)
    img[:] = r, 0, 0

    # radius = random.randint(int(w/3), int(w/2))
    cv2.circle(img, (int(HEIGHT/2), int(WIDTH/2)), int(r/5), (255, 0, 0), -1)
    
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

for _ in range(100000):
    if args.debug:
        batch = make_batch_debug(batch_size)
    else:
        batch = make_batch_multithreaded(batch_size)

    batch_t = torch.FloatTensor(batch).to(DEVICE)
    batch_t = batch_t.permute(0, 3, 1, 2) # (B, H, W, C) -->  (B, C, H, W)

    z_vector = encoder.forward(batch_t)
    pred = decoder.forward(z_vector)
    loss = loss_fn(pred, batch_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(float(loss))

   



    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image',img)
    # cv2.waitKey(250)

    x = 1


if __name__ == '__main__':
    main()