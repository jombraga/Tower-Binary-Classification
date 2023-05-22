import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np


def load_data(dir, categories, img_size=(150, 150, 3)):
    flat_data_arr = []
    target_arr = []

    for category in categories:

        print(f'Loading... category: {category}')
        path = os.path.join(dir, category)

        for img in os.listdir(path):

            img_array = imread(os.path.join(path, img))
            img_resized = resize(img_array, img_size)
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(categories.index(category))

        print(f'Loaded category: {category} successfully')

    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)
    df = pd.DataFrame(flat_data)
    df['Target'] = target
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return x, y
