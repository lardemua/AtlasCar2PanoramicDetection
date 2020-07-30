import json
import os

import urllib.request
from urllib.request import urlretrieve, urlopen

from PIL import Image
import numpy as np
from tqdm import tqdm

from multiprocessing import Pool

objects = json.load(open('export-2020-07-01T13_25_05.648Z.json'))

LABELS = {
    'sidewalk': 1, 'road': 0, 'car': 13,
}


def create_if_not_exists(dir):
    if os.path.exists(dir):
        return
    os.mkdir(dir)


create_if_not_exists('images')
create_if_not_exists('labels')


def load_image(id, o):
    image_file = 'images/{:08d}.jpg'.format(id)
    label_file = 'labels/{:08d}.png'.format(id)

    urlretrieve(o['Labeled Data'], image_file)

    image = Image.open(image_file)
    width, height = image.size

    labels = o['Label']['objects']

    mask = np.zeros((height, width)).astype('uint8')
    mask[:, :] = 255

    for label in labels:
        f = urlopen(label['instanceURI'])
        value = label['value']
        label_id = LABELS[value]
        image = Image.open(f).convert('L')
        image = np.array(image)
        mask[image == 255] = label_id

    mask = Image.fromarray(mask)
    mask.save(label_file)


with Pool(8) as pool:
    pool.starmap(load_image, enumerate(objects))
