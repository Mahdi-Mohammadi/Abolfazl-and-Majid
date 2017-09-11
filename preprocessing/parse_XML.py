import xmltodict
import numpy as np
import pandas as pd
from skimage.transform import rotate
import matplotlib.pyplot as plt
import os
from bs4 import BeautifulSoup
import re
import matplotlib.image as mpimage
from termcolor import colored
from shapely.geometry import box
from shapely.affinity import rotate as r
from matplotlib import pyplot
from descartes import PolygonPatch


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


def magic(image, images_cars, kernel_size, r_stride, c_stride, epsilon=1.):
    row = 0
    image_dim_0, image_dim_1 = np.shape(image)[0], np.shape(image)[1]
    cropped = []
    while row + kernel_size < image_dim_0:
        col = 0
        while col + kernel_size < image_dim_1:

            cars_in_this_sub_image = []

            for car in range(len(images_cars)):
                if in_image(car, images_cars, row, col, kernel_size, epsilon):
                    cars_in_this_sub_image.append({
                        'type': images_cars.iloc[car]['type'],
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
                cars_in_this.append({
                    'type': images_cars.iloc[car]['type'],
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
                cars_in.append({
                    'type': images_cars.iloc[car]['type'],
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


def get_related_xml(name):
    xmls = os.listdir('xml/')
    for n in xmls:
        xml = ''
        for line in open('xml/' + n):
            xml += line
        xml = BeautifulSoup(xml, 'lxml')
        if xml.find('image').text == name:
            return xml


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


def set_position(annotation, center):
    new_annotation = annotation
    coord_names = {"east": 1, "northeast": 2, "north": 3, "northwest": 4, "west": 5, "southwest": 6, "south": 7,
                   "southeast": 8}
    for i in range(len(annotation)):
        # x,y = 0,0
        # if annotation.iloc[i]['center.x'] > center:
        #     x = annotation.iloc[i]['center.x']  - center
        # else:
        #     x = center-annotation.iloc[i]['center.x']
        # if annotation.iloc[i]['center.y'] > center:
        #     y = annotation.iloc[i]['center.y'] - center
        # else:
        #     y = center - annotation.iloc[i]['center.y']
        import math
        teta = math.atan(
            (-annotation.iloc[i]['center.y'] + center) / (annotation.iloc[i]['center.x'] - center + 0.00001)) * 180 / np.pi
        # print(colored([0, 0], 'red'),
        #       colored([annotation.iloc[i]['center.x'], annotation.iloc[i]['center.y']], 'green'))
        # print(teta)
        # exit()
        if teta < 0:
            teta += 360
        pose = ''
        if 0 <= teta < 22.5 or 360 - 22.5 <= teta <= 360:
            pose = 'east'
        elif 22.5 <= teta < 45 + 22.5:
            pose = 'northeast'
        elif 45 + 22.5 <= teta < 90 + 22.5:
            pose = 'north'
        elif 90 + 22.5 <= teta < 135 + 22.5:
            pose = 'northwest'
        elif 135 + 22.5 <= teta < 180 + 22.5:
            pose = 'west'
        elif 180 + 22.5 <= teta < 180 + 45 + 22.5:
            pose = 'southwest'
        elif 180 + 45 + 22.5 <= teta < 270 + 22.5:
            pose = 'south'
        elif 270 + 22.5 <= teta < 270 + 45 + 22.5:
            pose = 'southeast'
        else:
            print(teta)
            print('-----------------------------')
        new_annotation.set_value(i, 'pose', pose)
        new_annotation.set_value(i, 'posenum', coord_names[pose])
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


images_name = [name for name in os.listdir('../train/') if 'JPG' in name]

BASE_DIR = '../train/'
CROP_DIR = '../cropped_image/'
name_counter = 0
for name in images_name:
    print('reading : ',colored(name+' xml file...','green'))
    xml = get_related_xml(name)
    print('done.')
    print(colored('convert to DataFrame...','green'))
    df = read_xml(xml)
    print('done.')
    print(colored('reading image:'+name,'green'))
    image = mpimage.imread(BASE_DIR + name)
    print('done.')
    print(colored('cropping started...','green'))
    cropped = magic(image, df, kernel_size=960, r_stride=100, c_stride=100, epsilon=0.73)
    print('done.')

    for image, annotation in cropped:
        if len(annotation) > 0:
            _90 = rotate(image, 90)
            _180 = rotate(image, 180)
            _270 = rotate(image, 270)
            _90_annotation = rotate_annotation(dict_to_data_frame(annotation), 90, len(image[0]))
            _180_annotation = rotate_annotation(dict_to_data_frame(annotation), 180, len(image[0]))
            _270_annotation = rotate_annotation(dict_to_data_frame(annotation), 270, len(image[0]))

            _90_annotation = set_position(_90_annotation, 960 / 2)
            _180_annotation = set_position(_180_annotation, 960 / 2)
            _270_annotation = set_position(_270_annotation, 960 / 2)

            write_to_xml(_90_annotation, 'XML', str(name_counter) + '_' + '90', 'new_xml/')
            write_to_xml(_180_annotation, 'XML', str(name_counter) + '_' + '180', 'new_xml/')
            write_to_xml(_270_annotation, 'XML', str(name_counter) + '_' + '270', 'new_xml/')
            annotation = dict_to_data_frame(annotation)
            write_to_xml(annotation, 'XML', str(name_counter) + '_' + 'main', 'new_xml/')

            mpimage.imsave(CROP_DIR + str(name_counter) + '_' + '90.jpeg', arr=_90)
            mpimage.imsave(CROP_DIR + str(name_counter) + '_' + '180.jpeg', arr=_180)
            mpimage.imsave(CROP_DIR + str(name_counter) + '_' + '270.jpeg', arr=_270)
            mpimage.imsave(CROP_DIR + str(name_counter) + '_' + 'main.jpeg', arr=image)

            name_counter += 1
    print(colored(name,'green'),'done.')
    print(colored('----------------------','blue'))
