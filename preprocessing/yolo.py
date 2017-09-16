import tensorflow as tf
import numpy as np
import pandas as pd
import itertools
import math
from tensorflow.contrib.layers import xavier_initializer_conv2d
import matplotlib.pyplot as plt
from termcolor import colored
import os
from shapely.geometry import box
from shapely.affinity import rotate as r
import matplotlib.image as mpimage
import re
from skimage.transform import rotate
from bs4 import BeautifulSoup
import random


# _____________________ D A T A ______________________
class RotatedRect:
    def __init__(self, cx, cy, w, h, angle):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle
        self.b = None

    def get_contour(self):
        w = self.w
        h = self.h
        c = box(self.cx - w, self.cy - h, self.cx + w, self.cy + h)
        rc = r(c, self.angle)
        self.b = rc
        return rc

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())

    def area(self):
        return self.b.area


def code(annotation):
    # print('code : ' ,re.findall('4K0G[0-9]+', annotation)[0])
    return re.findall('4K0G[0-9]+', annotation)[0]


def info(path):
    com_path = annotation_dir + path
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
    df['type'] = np.array(df['type'], dtype=str)
    df['pose'] = np.array(df['pose'], dtype=str)

    for i, car in enumerate(d):
        for col in df.columns:
            df.set_value(i, col, d[i][col])

    return df


def rotate_(x, y, angle, center):
    x_ = center[0] + (x - center[0]) * np.cos(angle * np.pi / 180) - (y - center[1]) * np.sin(angle * np.pi / 180)
    y_ = center[1] + (x - center[0]) * np.sin(angle * np.pi / 180) + (y - center[1]) * np.cos(angle * np.pi / 180)
    return x_, y_


def in_image(car_index, data, row, col, kernel_size, epsilon):
    try:
        image_box = RotatedRect(col + kernel_size / 2, row + kernel_size / 2, kernel_size / 2, kernel_size / 2, 0)
        bndbox = RotatedRect(data.iloc[car_index]['center.x'], data.iloc[car_index]['center.y'],
                             data.iloc[car_index]['size_width'],
                             data.iloc[car_index]['size_height'], data.iloc[car_index]['angle'])
        area = (bndbox.intersection(image_box).area / bndbox.area())
        # fig = pyplot.figure(1, figsize=(10, 4))
        # ax = fig.add_subplot(121)
        # ax.set_xlim(-5616, 5616)
        # ax.set_ylim(-3744, 3744)
        # ax.add_patch(PolygonPatch(bndbox.get_contour(), fc='#990000', alpha=0.7))
        # ax.add_patch(PolygonPatch(image_box.get_contour(), fc='#000099', alpha=0.7))
        # plt.pause(0.05)
        # total_area = bndbox.area()
        # print(colored(area,'blue'),colored(total_area,'green'))
        if area > epsilon:
            return True
        else:
            return False
    except ZeroDivisionError:

        print(car_index)
        plt.imshow(image)
        plt.scatter([data.iloc[car_index]['center.x']], [data.iloc[car_index]['center.y']], c='r', s=100)
        plt.show()
        print('x', data.iloc[car_index]['center.x'], )
        print('y', data.iloc[car_index]['center.y'], )
        print('w', data.iloc[car_index]['size_width'], )
        print('h', data.iloc[car_index]['size_height'], )
        print('angle', data.iloc[car_index]['angle'], )
        exit()


def magic(image, images_cars, kernel_size, r_stride, c_stride, epsilon=1.):
    images_cars['type'] = images_cars['type'].apply(lambda x: x.strip())
    row = 0
    image_dim_0, image_dim_1 = np.shape(image)[0], np.shape(image)[1]
    cropped = []
    while row + kernel_size < image_dim_0:
        col = 0
        while col + kernel_size < image_dim_1:

            cars_in_this_sub_image = []

            for car in range(len(images_cars)):
                if in_image(car, images_cars, row, col, kernel_size, epsilon):
                    if images_cars.iloc[car]['type'] == 'van' or images_cars.iloc[car]['type'] == 'pkw':
                        t = 'pkw'
                    elif images_cars.iloc[car]['type'] == 'truck':
                        t = 'truck'
                    else:
                        t = None

                    if t is not None:
                        cars_in_this_sub_image.append({
                            'type': t,
                            'center.x': images_cars.iloc[car]['center.x'] - col,
                            'center.y': images_cars.iloc[car]['center.y'] - row,
                            'size_width': images_cars.iloc[car]['size_width'],
                            'size_height': images_cars.iloc[car]['size_height'],
                            'angle': images_cars.iloc[car]['angle'],
                            'difficult': images_cars.iloc[car]['difficult'],
                            'truncated': images_cars.iloc[car]['truncated'],
                            'pose': images_cars.iloc[car]['pose'],
                            'posenum': images_cars.iloc[car]['posenum']
                        })

            cropped.append((image[row:row + kernel_size, col:col + kernel_size, :], cars_in_this_sub_image))
            col += c_stride
        cars_in_this = []
        for car in range(len(images_cars)):
            if in_image(car, images_cars, row, image_dim_1 - kernel_size, kernel_size, epsilon):
                if images_cars.iloc[car]['type'] == 'van' or images_cars.iloc[car]['type'] == 'pkw':
                    t = 'pkw'
                elif images_cars.iloc[car]['type'] == 'truck':
                    t = 'truck'
                else:
                    t = None
                if t is not None:
                    cars_in_this.append({
                        'type': t,
                        'center.x': images_cars.iloc[car]['center.x'] - (image_dim_1 - kernel_size),
                        'center.y': images_cars.iloc[car]['center.y'] - row,
                        'size_width': images_cars.iloc[car]['size_width'],
                        'size_height': images_cars.iloc[car]['size_height'],
                        'angle': images_cars.iloc[car]['angle'],
                        'difficult': images_cars.iloc[car]['difficult'],
                        'truncated': images_cars.iloc[car]['truncated'],
                        'pose': images_cars.iloc[car]['pose'],
                        'posenum': images_cars.iloc[car]['posenum']
                    })
        cropped.append((image[row:row + kernel_size, image_dim_1 - kernel_size:, :], cars_in_this))
        row += r_stride
    col = 0
    while col + kernel_size < image_dim_1:

        cars_in = []

        for car in range(len(images_cars)):
            if in_image(car, images_cars, image_dim_0 - kernel_size, col, kernel_size, epsilon):
                if images_cars.iloc[car]['type'] == 'van' or images_cars.iloc[car]['type'] == 'pkw':
                    t = 'pkw'
                elif images_cars.iloc[car]['type'] == 'truck':
                    t = 'truck'
                else:
                    t = None
                if t is not None:
                    cars_in.append({
                        'type': t,
                        'center.x': images_cars.iloc[car]['center.x'] - col,
                        'center.y': images_cars.iloc[car]['center.y'] - (image_dim_0 - kernel_size),
                        'size_width': images_cars.iloc[car]['size_width'],
                        'size_height': images_cars.iloc[car]['size_height'],
                        'angle': images_cars.iloc[car]['angle'],
                        'difficult': images_cars.iloc[car]['difficult'],
                        'truncated': images_cars.iloc[car]['truncated'],
                        'pose': images_cars.iloc[car]['pose'],
                        'posenum': images_cars.iloc[car]['posenum']
                    })
        cropped.append((image[image_dim_0 - kernel_size:, col:col + kernel_size, :], cars_in))
        col += c_stride
    return cropped


def plot(image, annotation):
    plt.imshow(image)
    x, y = annotation['center.x'].values, annotation['center.y'].values
    plt.scatter(x, y)


def draw_bounding_box(x, y, w, h, angle, color):
    x1 = x - w
    x2 = x + w
    y1 = y - h
    y2 = y + h
    x_, y_ = rotate_(np.array([x1, x1, x2, x2]), np.array([y1, y2, y2, y1]), -angle, [x, y])
    plt.plot([x_[0], x_[1]], [y_[0], y_[1]], c=color)
    plt.plot([x_[1], x_[2]], [y_[1], y_[2]], c=color)
    plt.plot([x_[2], x_[3]], [y_[2], y_[3]], c=color)
    plt.plot([x_[3], x_[0]], [y_[3], y_[0]], c=color)


def read_xml(annotation):
    objects = annotation.find_all("object")
    data = []
    for obj in objects:
        name = obj.find('name').text
        pose = obj.find('pose').text
        pose_num = int(obj.find('posenum').text)
        truncated = int(obj.find('truncated').text)
        difficult = int(obj.find('difficult').text)
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        xmax = int(bbox.find("xmax").text)
        ymin = int(bbox.find("ymin").text)
        ymax = int(bbox.find("ymax").text)
        angle = float(bbox.find('angle').text)
        data.append([name, pose, pose_num, truncated, difficult, xmin, xmax, ymin, ymax, angle])

    df = pd.DataFrame(data=data,
                      columns=['type', 'pose', 'posenum', 'truncated', 'difficult', 'center.x', 'size_width',
                               'center.y', 'size_height', 'angle'])

    return df


def main_name(name):
    return name[:name.index('.')]


def read_all_data(image_path, annotation_path, batch_size):
    image_names = os.listdir(image_path)

    annotation_names = os.listdir(annotation_path)

    annotations = {main_name(name): pd.read_csv(annotation_path + name) for name in annotation_names}

    images = {main_name(name): mpimage.imread(image_path + name) for name in image_names}

    data = []

    for name in images.keys():
        data.append((images[name], annotations[name]))

    for j in range(len(image_names) % batch_size):
        index = np.random.randint(0, len(list(images.keys())) - 2)
        data.append(data[index])

    images, annotations = [data[i][0] for i in range(len(data))], [data[i][1] for i in range(len(data))]

    for batch_num in range(int(len(images) / batch_size)):
        yield images[batch_num * batch_size:(batch_num + 1) * batch_size], \
              annotations[batch_num * batch_size:(batch_num + 1) * batch_size]


def rotate_annotation(annotation, angle, image_size):
    new_annotation = annotation
    new_annotation['angle'] += angle
    rotated_position = {
        90: {
            'north': 'west',
            'northwest': 'southwest',
            'west': 'south',
            'southwest': 'southeast',
            'south': 'east',
            'southeast': 'northeast',
            'east': 'north',
            'northeast': 'northwest'
        },
        180: {
            'north': 'south',
            'northwest': 'southeast',
            'west': 'east',
            'southwest': 'northeast',
            'south': 'north',
            'southeast': 'northwest',
            'east': 'west',
            'northeast': 'southwest'
        },
        270: {
            'north': 'east',
            'northwest': 'northeast',
            'west': 'north',
            'southwest': 'northwest',
            'south': 'west',
            'southeast': 'southwest',
            'east': 'south',
            'northeast': 'southeast'
        }
    }
    new_annotation['center.x'], new_annotation['center.y'] = \
        rotate_(new_annotation['center.x'].values,
                new_annotation['center.y'].values,
                -angle,
                [image_size / 2, image_size / 2])

    new_annotation['pose'] = new_annotation['pose'].apply(lambda x: x.strip())

    for i in range(len(new_annotation)):
        new_annotation.set_value(i, 'pose', rotated_position[angle][new_annotation.iloc[i]['pose']])

    return new_annotation


def write_to_xml(annotation, folder_name, name, save_path):
    xml = '<annotation>\n'
    xml += '<folder>' + folder_name + '</folder>\n'
    xml += '<filename>' + name + '.xml' + '</filename>\n'
    xml += '<source>\n'
    xml += '<database> Munich Aerial Imagery </database>\n'
    xml += '<annotation> PASCAL VOC2007 </annotation>\n'
    xml += '<size>\n'
    xml += '<width> 5616 </width>\n'
    xml += '<height> 3744 </height>\n'
    xml += '<depth> 3 </depth>\n'
    xml += '</size>\n'
    xml += '<image>' + name + '.jpeg' + '</image>\n'
    xml += '</source>\n'

    for i in range(len(annotation)):
        xml += '<object>\n'
        xml += '<name>' + annotation.iloc[i]['type'] + '</name>\n'
        xml += '<pose>' + annotation.iloc[i]['pose'] + '</pose>\n'
        xml += '<posenum>' + str(annotation.iloc[i]['posenum']) + '</posenum>\n'
        xml += '<truncated>' + str(annotation.iloc[i]['truncated']) + '</truncated>\n'
        xml += '<difficult>' + str(annotation.iloc[i]['difficult']) + '</difficult>\n'
        xml += '<bndbox>\n'
        xml += '<xmin>' + str(annotation.iloc[i]['center.x']) + '</xmin>\n'
        xml += '<ymin>' + str(annotation.iloc[i]['center.y']) + '</ymin>\n'
        xml += '<xmax>' + str(annotation.iloc[i]['size_width']) + '</xmax>\n'
        xml += '<ymax>' + str(annotation.iloc[i]['size_height']) + '</ymax>\n'
        xml += '<angle>' + str(annotation.iloc[i]['angle']) + '</angle>\n'
        xml += '</bndbox>\n'
        xml += '</object>\n\n'

    xml += '</annotation>'
    to_xml = open(save_path + name + '.xml', 'w')
    for line in xml:
        to_xml.write(line)
    to_xml.close()


def draw_grid(ng, image):
    x = 0
    max_y = len(image[0])
    plt.imshow(image)
    for i in range(0, ng):
        plt.plot([x, x], [0, max_y], c='g')
        plt.plot([0, max_y], [x, x], c='g')
        x += max_y / ng

    plt.show()


def get_related_xml(name):
    xmls = os.listdir('xml/')
    for n in xmls:
        xml = ''
        for line in open('xml/' + n):
            xml += line
        xml = BeautifulSoup(xml, 'lxml')
        if xml.find('image').text == name:
            return xml


# ____________________________________________________
# _____________________U T I L S______________________
# ____________________________________________________
def lrelu(x, leak):
    return tf.maximum(x, leak * x)


def conv2d(input, filter_shape, strides=2, padding='SAME', name='conv'):
    with tf.variable_scope(name):
        w = tf.get_variable(name='w',
                            shape=filter_shape,
                            dtype=tf.float32,
                            initializer=xavier_initializer_conv2d())

        tf.summary.histogram('w_' + name, w)

        b = tf.get_variable(name='b',
                            shape=[filter_shape[-1]],
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer())

        tf.summary.histogram('b_' + name, b)

        conv = tf.nn.conv2d(input=input,
                            filter=w,
                            strides=[1, strides, strides, 1],
                            padding=padding,
                            use_cudnn_on_gpu=True)
        return lrelu(conv + b, 0.02)


def multiply(x, shape, name='multiply', activation='lrelu'):
    with tf.variable_scope(name):
        w = tf.get_variable(name='w',
                            dtype=tf.float32,
                            initializer=xavier_initializer_conv2d(),
                            shape=shape)
        b = tf.get_variable(name='b',
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer(),
                            shape=[shape[-1]])
        output = tf.nn.xw_plus_b(x, w, b)
        if activation == 'sigmoid':
            output = tf.nn.sigmoid(output)
        elif activation == 'lrelu':
            output = lrelu(output, 0.02)

        return output


def avgpool(x, kernel=2, stride=2):
    return tf.nn.avg_pool(x, [1, kernel, kernel, 1], [1, stride, stride, 1], padding='SAME')


def sample_noise(shape):
    return tf.random_uniform(shape=shape, minval=-1, maxval=1)


def add_noise(input, shape):
    return input + sample_noise(shape)


def read_image(address, format):
    print('reading from : ', colored(address, 'blue'))
    image_dir = [address + name for name in os.listdir(address) if format in name.lower()]
    print(image_dir)
    images = [plt.imread(name) for name in image_dir]
    print(colored('reading finished.', 'green'))
    return np.array(images)


# __________________________________________________________________
# __________________________________________________________________
# __________________________________________________________________

def read_one_batch(names, batch_size, batch_num):
    images = [mpimage.imread(image_dir + names[i][0]) for i in
              range(batch_size * batch_num, (batch_num + 1) * batch_size)]
    annotations = [pd.read_csv(annotation_dir + names[i][1]) for i in
                   range(batch_size * batch_num, (batch_num + 1) * batch_size)]
    return images, annotations


def get_training_data(annotation):
    return get_cell_obj_info(annotation)


def get_cell_obj_info(objects):
    grid = [[None for _ in range(ngrid)] for _ in range(ngrid)]

    for i in range(len(objects)):

        cell_x, cell_y = get_cell([objects.iloc[i]['center.x'], objects.iloc[i]['center.y']], image_dim, image_dim)

        if grid[cell_x][cell_y] is None:
            grid[cell_x][cell_y] = [[objects.iloc[i]['center.x'],
                                     objects.iloc[i]['center.y'],
                                     objects.iloc[i]['size_width'],
                                     objects.iloc[i]['size_height'],
                                     convert_class[objects.iloc[i]['type']],
                                     objects.iloc[i]['angle']]]

    return grid


def get_cell(point, width, height):
    col = int(math.floor(point[0] / width * (ngrid - 1)))
    row = int(math.floor(point[1] / height * (ngrid - 1)))

    return [row, col]


def reshape(bbox, original_width, original_height, new_width, new_height):
    w_ratio = new_width / original_width
    h_ratio = new_height / original_height

    return [bbox[0] * w_ratio,
            bbox[1] * h_ratio,
            bbox[2] * w_ratio,
            bbox[3] * h_ratio]


class Yolo:
    def __init__(self):
        self.weights_file = '../log/model/yolo.ckpt'
        self.alpha = 0.1
        self.threshold = 0.2
        self.iou_threshold = 0.5
        self.classes = ['pkw', 'truck']
        self.keep_prob = tf.placeholder(tf.float32)
        self.lambdacoord = 5.0
        self.lambdanoobj = 0.5
        self.log_step = 2
        self.show_result = 1
        self.save_step = 40
        self.epoch = tf.placeholder(tf.int32)
        self.training_step = 1000
        self.len_data = 100000
        self.sess = tf.Session()
        self.summaries_path = '../log/summaries/'

        self.x = tf.placeholder(shape=[None, image_dim, image_dim, 4], name='images', dtype='float')
        self.x_ = tf.placeholder(tf.float32, [None, ngrid, ngrid, nbox])
        self.y_ = tf.placeholder(tf.float32, [None, ngrid, ngrid, nbox])
        self.w_ = tf.placeholder(tf.float32, [None, ngrid, ngrid, nbox])
        self.h_ = tf.placeholder(tf.float32, [None, ngrid, ngrid, nbox])
        self.C_ = tf.placeholder(tf.float32, [None, ngrid, ngrid, nbox])
        self.p_ = tf.placeholder(tf.float32, [None, ngrid, ngrid, nclass])
        self.obj = tf.placeholder(tf.float32, [None, ngrid, ngrid, nbox])
        self.objI = tf.placeholder(tf.float32, [None, ngrid, ngrid])
        self.noobj = tf.placeholder(tf.float32, [None, ngrid, ngrid, nbox])
        self.o_ = tf.placeholder(tf.float32, [None, ngrid, ngrid, nbox])
        self.batch_size = 64

    def feed(self, x):
        with tf.variable_scope('yolo'):
            # VGG 16
            # 512 512 3
            output = conv2d(x, [3, 3, 4, 64], strides=1, name='conv1')
            output = conv2d(output, [3, 3, 64, 64], strides=1, name='conv2')
            output = avgpool(output)
            # 256 256 64
            output = conv2d(output, [3, 3, 64, 128], strides=1, name='conv3')
            output = conv2d(output, [3, 3, 128, 128], strides=1, name='conv4')
            output = avgpool(output)
            # 128 128 128
            output = conv2d(output, [3, 3, 128, 256], strides=1, name='conv5')
            output = conv2d(output, [3, 3, 256, 256], strides=1, name='conv6')
            output = conv2d(output, [3, 3, 256, 256], strides=1, name='conv7')
            output = avgpool(output)
            # 64 64 256
            output = conv2d(output, [3, 3, 256, 512], strides=1, name='conv8')
            output = conv2d(output, [3, 3, 512, 512], strides=1, name='conv9')
            output = conv2d(output, [3, 3, 512, 512], strides=1, name='conv10')
            output = avgpool(output)
            # 32 32 512
            output = conv2d(output, [3, 3, 512, 512], strides=1, name='conv11')
            output = conv2d(output, [3, 3, 512, 512], strides=1, name='conv12')
            output = conv2d(output, [1, 1, 512, 512], strides=1, name='conv13')
            output = conv2d(output, [1, 1, 512, (ngrid * ngrid * (nbox * 6 + nclass))], strides=2, name='conv14')
            return output

    def IoU(self, box1, box2):
        b1 = RotatedRect(box1[0], box1[1], box1[2], box1[3], box1[4])
        b2 = RotatedRect(box2[0], box2[1], box2[2], box2[3], box2[4])
        intersection = b1.intersection(b2).area
        return intersection / (b1.area() + b2.area() - intersection)

    def build_label(self, image, annotation):
        X = []
        Y = []
        W = []
        H = []
        C = []
        P = []
        obj_ = []
        objI_ = []
        noobj_ = []
        Image = []
        O = []

        for im, ann in zip(image, annotation):
            prelabel = get_training_data(ann)

            x = np.zeros([ngrid, ngrid, nbox])

            y = np.zeros([ngrid, ngrid, nbox])

            w = np.zeros([ngrid, ngrid, nbox])

            h = np.zeros([ngrid, ngrid, nbox])

            c = np.zeros([ngrid, ngrid, nbox])

            p = np.zeros([ngrid, ngrid, nclass])

            obj = np.zeros([ngrid, ngrid, nbox])

            objI = np.zeros([ngrid, ngrid])

            noobj = np.ones([ngrid, ngrid, nbox])

            orientation = np.zeros([ngrid, ngrid, nbox])

            for i, j in itertools.product(range(0, ngrid), range(0, ngrid)):

                if prelabel[i][j] is not None:
                    try:
                        index = 0

                        x[i][j][0] = (float(prelabel[i][j][index][0]) / len(im)) * ngrid - i

                        y[i][j][0] = (float(prelabel[i][j][index][1]) / len(im[0])) * ngrid - j

                        w[i][j][0] = np.sqrt(prelabel[i][j][index][2]) / len(im) * ngrid

                        h[i][j][0] = np.sqrt(prelabel[i][j][index][3]) / len(im[0])

                        c[i][j][0] = 1.0

                        p[i][j][int(prelabel[i][j][0][4])] = 1.0 / float(len(prelabel[i][j]))

                        obj[i][j][0] = 1.0

                        objI[i][j] = 1.0

                        noobj[i][j][0] = 0.0

                        orientation[i][j][0] = prelabel[i][j][0][-1] / 360
                    except ValueError:
                        print(self.classes.index(prelabel[i][j][0][4]))
                        exit()

            X.append(x)

            Y.append(y)

            W.append(w)

            H.append(h)

            C.append(c)

            P.append(p)

            obj_.append(obj)

            objI_.append(objI)

            noobj_.append(noobj)

            O.append(orientation)

            im = np.asarray(im)

            inputs = np.zeros((1, image_dim, image_dim, 4), dtype='float32')
            inputs[0] = (im / 255.0) * 2.0 - 1.0

            Image.append(inputs[0])

        X = np.array(X)

        Y = np.array(Y)

        W = np.array(W)

        H = np.array(H)

        C = np.array(C)

        P = np.array(P)

        obj_ = np.array(obj_)

        objI_ = np.array(objI_)

        noobj_ = np.array(noobj_)

        O = np.array(O)

        Image = np.array(Image)

        return {'x': Image,
                'x_': X,
                'y_': Y,
                'w_': W,
                'h_': H,
                'C_': C,
                'p_': P,
                'obj': obj_,
                'objI': objI_,
                'noobj': noobj_,
                'o_': O
                }

    def train(self):

        output = self.feed(self.x)
        class_prob = output[:, :, :, :nclass]
        confidence = output[:, :, :, nclass:nclass + 1]
        boxes = output[:, :, :, nclass + 1:]

        boxes0 = boxes[:, :, :, 0]  # x
        boxes1 = boxes[:, :, :, 1]  # y
        boxes2 = boxes[:, :, :, 2]  # w
        boxes3 = boxes[:, :, :, 3]  # h
        boxes4 = boxes[:, :, :, 4]  # o

        sub_x = tf.subtract(boxes0, self.x_)

        sub_y = tf.subtract(boxes1, self.y_)

        sub_w = tf.subtract(tf.sqrt(tf.abs(boxes2)), tf.sqrt(self.w_))

        sub_h = tf.subtract(tf.sqrt(tf.abs(boxes3)), tf.sqrt(self.h_))

        sub_o = tf.subtract(tf.abs(boxes4), tf.sqrt(self.o_))

        sub_c = tf.subtract(confidence, self.C_)

        sub_p = tf.subtract(class_prob, self.p_)

        loss_x = tf.multiply(self.lambdacoord,
                             tf.reduce_sum(tf.multiply(self.obj, tf.pow(sub_x, 2)), axis=[1, 2, 3]))

        tf.summary.scalar('loss_x', loss_x)

        loss_y = tf.multiply(self.lambdacoord,
                             tf.reduce_sum(tf.multiply(self.obj, tf.pow(sub_y, 2)), axis=[1, 2, 3]))

        tf.summary.scalar('loss_y', loss_y)

        loss_w = tf.multiply(self.lambdacoord,
                             tf.reduce_sum(tf.multiply(self.obj, tf.multiply(sub_w, 2)), axis=[1, 2, 3]))

        tf.summary.scalar('loss_w', loss_w)

        loss_h = tf.multiply(self.lambdacoord,
                             tf.reduce_sum(tf.multiply(self.obj, tf.pow(sub_h, 2)), axis=[1, 2, 3]))

        tf.summary.scalar('loss_h', loss_h)

        loss_o = tf.multiply(self.lambdacoord,
                             tf.reduce_sum(tf.multiply(self.obj, tf.pow(sub_o, 2)), axis=[1, 2, 3]))

        tf.summary.scalar('loss_o', loss_o)

        loss_c_obj = tf.reduce_sum(tf.multiply(self.obj, tf.pow(sub_c, 2)), axis=[1, 2, 3])

        tf.summary.scalar('loss_c_obj', loss_c_obj)

        loss_c_nobj = tf.multiply(self.lambdanoobj,
                                  tf.reduce_sum(tf.multiply(self.noobj, tf.pow(sub_c, 2)), axis=[1, 2, 3]))

        tf.summary.scalar('loss_c_noobj', loss_c_nobj)

        loss_p = tf.reduce_sum(tf.multiply(self.objI, tf.reduce_sum(tf.pow(sub_p, 2), axis=3)), axis=[1, 2])

        tf.summary.scalar('loss_p', loss_p)

        loss = tf.add_n(
            (loss_x, loss_y, loss_w, loss_h, loss_c_obj, loss_c_nobj, loss_p, loss_o))

        loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', loss)

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   10000, 0.96, staircase=True)
        learning_step = (tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step))

        init = tf.global_variables_initializer()

        merged = tf.summary.merge_all()
        saver = tf.train.Saver()
        tf.get_default_graph().finalize()

        with self.sess as sess:
            summary_writer = tf.summary.FileWriter(self.summaries_path, self.sess.graph)

            if first_run:
                sess.run(init)
                print('all variables are initialized.')

            else:
                saver.restore(sess, self.weights_file)
                print('model restored.')
            num_of_samples = len(os.listdir(image_dir))
            for i in range(self.training_step):
                random.shuffle(names)
                for i in range(int(num_of_samples / self.batch_size)):
                    images, annotations = read_one_batch(names, self.batch_size, i)
                    interpreted_data = self.build_label(images, annotations)
                    if tf.train.global_step(sess, global_step) % self.show_result == 0:

                        _, cost = sess.run([learning_step, loss],
                                           feed_dict={self.x: interpreted_data['x'],
                                                      self.x_: interpreted_data['x_'],
                                                      self.y_: interpreted_data['y_'],
                                                      self.w_: interpreted_data['w_'],
                                                      self.h_: interpreted_data['h_'],
                                                      self.o_: interpreted_data['o_'],
                                                      self.C_: interpreted_data['C_'],
                                                      self.p_: interpreted_data['p_'],
                                                      self.obj: interpreted_data['obj'],
                                                      self.noobj: interpreted_data['noobj'],
                                                      })
                        print('optimized')
                        print(colored('Cost:', 'blue'), colored(cost, 'green'))
                    else:
                        sess.run(learning_step,
                                 feed_dict={self.x: interpreted_data['x'],
                                            self.x_: interpreted_data['x_'],
                                            self.y_: interpreted_data['y_'],
                                            self.w_: interpreted_data['w_'],
                                            self.h_: interpreted_data['h_'],
                                            self.o_: interpreted_data['o_'],
                                            self.C_: interpreted_data['C_'],
                                            self.p_: interpreted_data['p_'],
                                            self.obj: interpreted_data['obj'],
                                            self.noobj: interpreted_data['noobj'],
                                            })

                    if tf.train.global_step(sess, global_step) % self.log_step == 0:
                        summary_writer.add_summary(merged, global_step=global_step)

                    if tf.train.global_step(sess, global_step) % self.save_step == 0:
                        saver.save(sess, self.weights_file)


# ____________________________ Hyperparameters _______________________________

ngrid = 16
nclass = 2
classes = ['pkw', 'truck']
image_dir = '../cropped_image/'
annotation_dir = 'csv/'
images_names = os.listdir(image_dir)
annotations_names = os.listdir(annotation_dir)
annotations_names.sort()
images_names.sort()
names = [(images_names[i], annotations_names[i]) for i in range(len(images_names))]
nbox = 1
image_dim = 512
convert_class = {'pkw': 0, 'truck': 1}
first_run = True

if len(os.listdir('csv/')) == 0:
    BASE_DIR = '../train/'
    CROP_DIR = '../cropped_image/'
    name_counter = 5203
    images_name = [name for name in os.listdir('../train/') if 'JPG' in name]
    for name in images_name:

        print('reading : ', colored(name + ' xml file...', 'green'))
        xml = get_related_xml(name)
        print('done.')
        print(colored('convert to DataFrame...', 'green'))
        df = read_xml(xml)
        print('done.')
        print(colored('reading image:' + name, 'green'))
        image = mpimage.imread(BASE_DIR + name)
        print('done.')
        if name == '2012-04-26-Muenchen-Tunnel_4K0G0070.JPG':
            df.set_value(386, 'center.x', 2270)
            df.set_value(386, 'center.y', 2947)
            df.set_value(386, 'size_height', 11)
            df.set_value(386, 'size_width', 22)
            df.set_value(386, 'angle', 87.)

        print(colored('cropping started...', 'green'))
        cropped = magic(image, df, kernel_size=512, r_stride=100, c_stride=100, epsilon=0.8)
        print('done.')

        for image, annotation in cropped:
            if len(annotation) > 0:
                _90 = rotate(image, 90)
                _180 = rotate(image, 180)
                _270 = rotate(image, 270)
                _90_annotation = rotate_annotation(dict_to_data_frame(annotation), 90, len(image[0]))
                _180_annotation = rotate_annotation(dict_to_data_frame(annotation), 180, len(image[0]))
                _270_annotation = rotate_annotation(dict_to_data_frame(annotation), 270, len(image[0]))

                _90_annotation.to_csv('csv/' + str(name_counter) + '_' + '90.csv', index=False)
                _180_annotation.to_csv('csv/' + str(name_counter) + '_' + '180.csv', index=False)
                _270_annotation.to_csv('csv/' + str(name_counter) + '_' + '270.csv', index=False)
                annotation = dict_to_data_frame(annotation)
                annotation.to_csv('csv/' + str(name_counter) + '_' + 'main.csv', index=False)

                mpimage.imsave(CROP_DIR + str(name_counter) + '_' + '90.jpeg', arr=_90)
                mpimage.imsave(CROP_DIR + str(name_counter) + '_' + '180.jpeg', arr=_180)
                mpimage.imsave(CROP_DIR + str(name_counter) + '_' + '270.jpeg', arr=_270)
                mpimage.imsave(CROP_DIR + str(name_counter) + '_' + 'main.jpeg', arr=image)

                name_counter += 1
                # plt.imshow(image)
                # annotation = dict_to_data_frame(annotation)
                # for i in range(len(annotation)):
                #     draw_bounding_box(annotation.iloc[i]['center.x'],
                #                       annotation.iloc[i]['center.y'],
                #                       annotation.iloc[i]['size_width'],
                #                       annotation.iloc[i]['size_height'],
                #                       annotation.iloc[i]['angle'], 'g')
                # plt.show()
        print(colored(name, 'green'), 'done.')
        print(colored('----------------------', 'blue'))
else:
    yolo = Yolo()
    yolo.train()
