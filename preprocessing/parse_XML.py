import xmltodict
import numpy as np
import pandas as pd
from skimage.transform import rotate
import matplotlib.pyplot as plt
import os
import re
import matplotlib.image as mpimage

# def to_dict(path):
#     l = ''
#     for line in open(path):
#         l += line
#
#     d = xmltodict.parse(l)
#     image = {'name': d['annotation']['source']['image'], 'objects': {
#         i: {
#
#         }
#
#         for i, car in enumerate(d['annotation']['object'])
#         }}
#
#     for i, car in enumerate(d['annotation']['object']):
#         image['objects'][i]['name'] = car['name']
#         for key in car['bndbox']:
#             image['objects'][i][key] = car['bndbox'][key]
#
#     # return image
# def r(x, y, angle, center):
#     x_, y_ = [], []
#     for i in range(len(x)):
#         newX = center[0] + (x[i] - center[0]) * np.cos(angle * np.pi / 180) - (y[i] - center[1]) * np.sin(
#             angle * np.pi / 180)
#         newY = center[1] + (x[i] - center[0]) * np.sin(angle * np.pi / 180) + (y[i] - center[1]) * np.cos(
#             angle * np.pi / 180)
#         x_.append(newX)
#         y_.append(newY)
#     return x_, y_
#
#
# def rotate_crop(image, angle, center, size, i):
#     c = r([center[0]], [center[1]], angle, [len(image[0]) / 2, len(image[0]) / 2])
#     new_center = (c[0][0], c[1][0])
#     new_image = rotate(image,-angle)
#     return new_image


BASE_DIR = '../train/'
CROP_DIR = '../train/'


def code(annotation):
    # print('code : ' ,re.findall('4K0G[0-9]+', annotation)[0])
    return re.findall('4K0G[0-9]+', annotation)[0]


def info(path):
    com_path = BASE_DIR + path
    file = open(com_path)
    cars = []
    for line in file:
        if not (line[0] == '#' or line[0] == '@'):
            line = list(map(float, line.split()[1:]))

            cars.append({
                'type': int(line[0]),
                'center.x': line[1],
                'center.y': line[2],
                'size_width': line[3],
                'size_height': line[4],
                'angle': line[5]
            })
    return cars


def parse_file(images_name, annotation):
    images_annotation = {
        image_name: [info(path) for path in annotation if code(path) in image_name]
        for image_name in images_name
        }
    for key in images_annotation.keys():
        l = []
        for i in range(len(images_annotation[key])):
            for j in range(len(images_annotation[key][i])):
                l.append(images_annotation[key][i][j])
        images_annotation[key] = l
    return images_annotation


def dict_to_data_frame(d):
    df = pd.DataFrame(data=np.zeros((len(d), len(d[0].keys()))), columns=d[0].keys())
    for i, car in enumerate(d):
        for col in df.columns:
            df.set_value(i, col, d[i][col])
    return df


def rotate_(x, y, angle, center):
    x_, y_ = [], []
    for i in range(len(x)):
        newX = center[0] + (x[i] - center[0]) * np.cos(angle * np.pi / 180) - (y[i] - center[1]) * np.sin(
            angle * np.pi / 180)
        newY = center[1] + (x[i] - center[0]) * np.sin(angle * np.pi / 180) + (y[i] - center[1]) * np.cos(
            angle * np.pi / 180)
        x_.append(newX)
        y_.append(newY)
    return x_, y_


def magic(image, images_cars, kernel_size, r_stride, c_stride):
    row = 0
    image_dim_0, image_dim_1 = np.shape(image)[0], np.shape(image)[1]
    cropped = []
    while row + kernel_size < image_dim_0:
        col = 0
        while col + kernel_size < image_dim_1:

            cars_in_this_sub_image = []

            for car in range(len(images_cars)):
                if col <= images_cars.iloc[car]['center.x'] <= col + kernel_size and row <= images_cars.iloc[car]['center.y'] <= row + kernel_size:

                    cars_in_this_sub_image.append({
                        'type': int(images_cars.iloc[car]['type']),
                        'center.x': images_cars.iloc[car]['center.x'] - col,
                        'center.y': images_cars.iloc[car]['center.y'] - row,
                        'size_width': images_cars.iloc[car]['size_width'],
                        'size_height': images_cars.iloc[car]['size_height'],
                        'angle': images_cars.iloc[car]['angle']
                    })
            cropped.append((image[row:row + kernel_size, col:col + kernel_size, :], cars_in_this_sub_image))
            col += c_stride
        row += r_stride
    return cropped


def plot(image, annotation):
    plt.imshow(image)
    x, y = annotation['center.x'].values, annotation['center.y'].values
    plt.scatter(x, y)
    plt.show()


images_name = [name for name in os.listdir('../train/') if 'JPG' in name]
annotation = [name for name in os.listdir('../train/') if 'samp' in name]
f = parse_file(images_name, annotation)
df = dict_to_data_frame(f[images_name[0]])

image = mpimage.imread(BASE_DIR+images_name[0])
plot(image, df)
cropped = magic(image, df, kernel_size=960, r_stride=100, c_stride=100)
for sub in cropped :
    if len(sub[1])>0:
        df  = dict_to_data_frame(sub[1])
        plot(sub[0],df)
        _270 = rotate(sub[0],270)
        _df = df
        _df['center.x'] , _df['center.y'] =  rotate_(df['center.x'].values,df['center.y'].values,-270, [960 / 2,960 / 2])
        plot(_270,_df)
