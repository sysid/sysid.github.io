#! /usr/bin/python
# -*- coding: utf-8 -*-

"""module docstring
ipython --pdb -- CropFish2.py overlay --head 331 --tail 250
"""

# imports
import sys, os, argparse, logging  # NOQA
from pprint import pprint
from twBase import *  # NOQA
import glob
import h5py

import random
import json
from PIL import Image
from scipy import ndimage
import cv2

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

import matplotlib.ticker as plticker
import matplotlib.patches as patches


#Allow relative imports to directories above cwd/
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# constants
scriptPath = print(os.path.abspath(__file__))
# exception classes
# interface functions
# classes
# internal functions & classes

DATA_HOME_DIR = './data'
LABELS_DIR = 'labels/'
TRAIN_DIR = 'data/train/'
OUTPUT_DIR = DATA_HOME_DIR + '/output2/'
testImg = DATA_HOME_DIR + '/train/DOL/DOL_05635.jpg'
testImg = DATA_HOME_DIR + '/train/ALB/ALB_00010.jpg'
RAD_TO_DEGREE = 57.2958
LABELS_DIR = './labels/'
#target_size=(224, 224)


def unit_vector(vector):
    """ Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_classtype(name):
    return name.split('_')[0]

def get_img(img_path, target_size=None):
    '''returns float array with RGB ordering
        not suited for showing
    '''
    name = os.path.basename(img_path)

    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    log.info("Image", name=name, size=img.shape)
    return img, name

def load_annotation(label_path):
    with open(label_path) as data_file:
        data = json.load(data_file)
    return data

def get_img_annotation_data(name):
    label_path = LABELS_DIR + get_classtype(name) + '_labels.json'
    data = load_annotation(label_path)

    for img_data in data:
        #log.debug("xxx", name=name, x=img_data['filename'])
        if img_data['filename'] == name:
            return img_data
    return None


def plot(args):
    img_path = args.image
    img, name = get_img(img_path)

    fig, ax = plt.subplots(1)

    img_data = get_img_annotation_data(name)
    p_heads, p_tails = get_coords(name, img_data)

    #if hasattr(args, 'resize'):
    if args.resize is not None:
        if  p_heads is not None:  # resize_points must be before img resize
            p_heads = rescale_point(p_heads, img, args.resize)
            p_tails = rescale_point(p_tails, img, args.resize)
        img = cv2.resize(img,(args.resize,args.resize))

    if  p_heads is not None:
        plt.scatter(p_heads[0], p_heads[1], color="blue")
        plt.scatter(p_tails[0], p_tails[1], color="red")

    plt.imshow(image.array_to_img(img))
    plt.show()


def crop_image(args):
    img_path = args.image
    img, name = get_img(img_path)

    fig, ax = plt.subplots(1)

    img_data = get_img_annotation_data(name)
    if  img_data is not None:
        if len(img_data['annotations']) >= 2:
            dst, p_headsT, p_tailsT = process_image(img, img_data)
            ax.imshow(dst)
            plt.scatter(p_headsT[0], p_headsT[1], color="blue")
            plt.scatter(p_tailsT[0], p_tailsT[1], color="red")
    else:
        log.warn("Annotation data not found", name=name)
        ax.imshow(img)


    #ax.imshow(img)
    plt.show()


def get_coords(name, img_data):
    if  img_data is not None:
        if len(img_data['annotations']) >= 2:
            p_heads = np.array((img_data['annotations'][0]['x'], img_data['annotations'][0]['y']))
            p_tails = np.array((img_data['annotations'][1]['x'], img_data['annotations'][1]['y']))
            return p_heads, p_tails
    else:
        log.warn("Annotation data not found or complete", name=name)
    return None, None

def rescale_point(point, img, scale):
    img_height, img_width, _ = img.shape
    point[0] = scale/img_width * point[0]
    point[1] = scale/img_height * point[1]
    return point


def crop(args):
    """take all label files and process them"""
    label_files = glob.glob(LABELS_DIR + '*.json')
    for file in label_files:
        log.info("labels: ", labels=file)
        process_labels(file)

def process_labels(label_file):
    file_name = os.path.basename(label_file)
    class_name = file_name.split("_")[0]
    if not os.path.isdir(OUTPUT_DIR + class_name.upper()):
        os.mkdir(OUTPUT_DIR + class_name.upper())
    print("Processing " + class_name + " labels")

    data = load_annotation(label_file)

    for img_data in data:
        img_file = TRAIN_DIR + class_name.upper() + '/' + img_data['filename']
        if not os.path.exists(img_file):
            log.error("File not there: ", file=img_file)
        else:
            img = cv2.imread(img_file)
            # We will crop only images with both heads and tails present for cleaner dataset
            if len(img_data['annotations']) >= 2:
                dst, _, _ = process_image(img, img_data)
                cv2.imwrite(OUTPUT_DIR + class_name.upper() + '/' + img_data['filename'], dst)

def process_image(img, img_data):
    """
    rotates the annotated fish and crops it
    """
    img_height, img_width, _ = img.shape

    # make np vector for annotation points
    p_heads = np.array((img_data['annotations'][0]['x'], img_data['annotations'][0]['y']))
    p_tails = np.array((img_data['annotations'][1]['x'], img_data['annotations'][1]['y']))
    p_middle = (p_heads + p_tails)/2
    log.debug("Coord:", p_heads=p_heads, p_tails=p_tails, p_middle=p_middle)

    # for debugging
    plt.imshow(img)
    plt.scatter(p_heads[0], p_heads[1], color="blue")
    plt.scatter(p_tails[0], p_tails[1], color="red")

    # fish vector
    dist = np.linalg.norm(p_heads-p_tails)
    offset = 3.0 * dist / 4.0
    D = -1*p_tails + p_heads
    angle = np.rad2deg(angle_between((1, 0), D))
    #angle = angle_between((1, 0), D)
    log.debug("Rotating", angle=angle, D=D)

    # translate to center of img
    img_center = np.array([img_height/2, img_width/2])
    t = img_center - p_middle
    t = np.reshape(t, (2,1))
    T = np.concatenate((np.identity(2), t), axis=1)

    # get the Affine transformation matrix
    if p_heads[1] > p_tails[1]:  # head is above tail
        M = cv2.getRotationMatrix2D((p_middle[0], p_middle[1]), angle, 1)
    else:
        M = cv2.getRotationMatrix2D((p_middle[0], p_middle[1]), -angle, 1)


    # compinte affine transform: make them 3x3
    # http://stackoverflow.com/questions/13557066/built-in-function-to-combine-affine-transforms-in-opencv
    A1 = np.identity(3)
    A2 = np.identity(3)
    R = np.identity(3)
    A1[:2] = T
    A2[:2] = M
    R = A1@A2
    RR = R[:2]

    dst = cv2.warpAffine(img, RR, (img_height, img_width))

    # adjust the dim of the 2d vectors http://answers.opencv.org/question/19146/2x3-transformation-matrix/
    # (2x3) transformation matrxi (x0, y0) is translation ?
    p_headsT = RR @ np.append(p_heads, 1)
    p_tailsT = RR @ np.append(p_tails, 1)
    p_middleT = RR @ np.append(p_middle, 1)

    # for debugging
    #plt.imshow(dst)
    #plt.scatter(p_headsT[0], p_headsT[1], color="blue")
    #plt.scatter(p_tailsT[0], p_tailsT[1], color="red")


    x_left = max(0, p_middleT[0] - offset)
    x_right = min(img_width - 1, p_middleT[0] + offset)
    y_up = max(0, p_middleT[1] - offset)
    y_down = min(img_height - 1, p_middleT[1] + offset)
    x_left, x_right, y_up, y_down = int(x_left), int(x_right), int(y_up), int(y_down)
    dst = dst[y_up:y_down+1, x_left:x_right+1, :]

    # adjust the shortend coordinates
    p_headsT = (p_headsT[0]-x_left, p_headsT[1]-y_up)
    p_tailsT = (p_tailsT[0]-x_left, p_tailsT[1]-y_up)
    #plt.scatter(p_headsT[0]-x_left, p_headsT[1]-y_up, color="blue")
    #plt.scatter(p_tailsT[0], p_tailsT[1], color="red")

    return dst, p_headsT, p_tailsT


def create_data(args):
    """take all label files and process them
    read every image, resize image and heads/tails and add to X, Y
    """
    nwidth = args.grid
    nheight = args.grid
    X = []
    Y = []
    Z = []
    label_files = glob.glob(LABELS_DIR + '*.json')
    for file in label_files:
        log.info("labels: ", labels=file)

        file_name = os.path.basename(file)
        class_name = file_name.split("_")[0]

        print("Processing " + class_name + " labels")

        data = load_annotation(file)

        for img_data in data:
        #for img_data in data[:3]:
            img_file = TRAIN_DIR + class_name.upper() + '/' + img_data['filename']
            if not os.path.exists(img_file):
                log.error("File not there: ", file=img_file)
            else:
                img, name = get_img(img_file, target_size=(args.resize, args.resize))
                img_height, img_width, _ = img.shape
                dx = img_width/nwidth
                dy = img_height/nheight

                if len(img_data['annotations']) >= 2:
                    p_heads, p_tails = get_coords(name, img_data)

                    h_bucket, _ = find_bucket(p_heads, nwidth, nheight, dx, dy)
                    h_arr = np.zeros(nwidth*nheight)
                    h_arr[h_bucket] = 1

                    t_bucket, _ = find_bucket(p_tails, nwidth, nheight, dx, dy)
                    t_arr = np.zeros(nwidth*nheight)
                    t_arr[t_bucket] = 1

                    target = np.concatenate([h_arr, t_arr])

                    p_heads = rescale_point(p_heads, img, args.resize)
                    p_tails = rescale_point(p_tails, img, args.resize)

                    X.append(img)
                    Y.append(target)

                    # https://github.com/h5py/h5py/issues/796
                    # http://stackoverflow.com/questions/14472650/python-3-encode-decode-vs-bytes-str
                    Z.append(img_file.encode('utf-8'))

    X = np.array(X)
    X = preprocess_input(X)
    Y = np.array(Y)
    Z = np.array(Z)
    log.info("Created X array.", shape=X.shape)
    log.info("Created Y array.", shape=Y.shape)
    log.info("Created Z array.", shape=Y.shape)

    h5file = 'dataset.h5'
    if os.path.exists(h5file):
        os.remove(h5file)

    f = h5py.File(h5file)
    f['X'] = X
    f['Y'] = Y
    f['Z'] = Z
    f.close()


def overlay(args):
    img_path = args.image
    img, name = get_img(img_path)
    img_height, img_width, _ = img.shape
    nwidth = args.grid
    nheight = args.grid
    nbucket = nwidth * nheight

    label_path = LABELS_DIR + get_classtype(name) + '_labels.json'
    img_data = get_img_annotation_data(name)

    ax, (dx,dy) = plot_grid(img, nwidth, nheight)

    if args.head is not None:
        point = find_coord(args.head, nwidth, nheight, dx, dy)
        plot_box(ax, point, (dx, dy), "blue")

    if args.tail is not None:
        point = find_coord(args.tail, nwidth, nheight, dx, dy)
        plot_box(ax, point, (dx, dy), "red")

    if args.annotate:
        if len(img_data['annotations']) >= 2:
            p_heads = np.array((img_data['annotations'][0]['x'], img_data['annotations'][0]['y']))
            p_tails = np.array((img_data['annotations'][1]['x'], img_data['annotations'][1]['y']))

            plot_annotation(ax, p_heads, p_tails)

            bucket, point = find_bucket(p_heads, nwidth, nheight, dx, dy)
            log.info("Bucket head found:", bucket=bucket, point=point)
            plot_box(ax, point, (dx, dy), "blue")

            bucket, point = find_bucket(p_tails, nwidth, nheight, dx, dy)
            log.info("Bucket head found:", bucket=bucket, point=point)
            plot_box(ax, point, (dx, dy), "red")


    plt.show()


def plot_annotation(ax, p_heads=None, p_tails=None):
    if p_heads is not None:
        plt.scatter(p_heads[0], p_heads[1], color="blue", s=2)
    if p_tails is not None:
        plt.scatter(p_tails[0], p_tails[1], color="red", s=2)


def plot_box(ax, point, dim, color):
    ax.add_patch(
        patches.Rectangle(
            point,
            dim[0],
            dim[1],
            fill = False,
            edgecolor=color,
            linewidth=1
        )
    )


def plot_grid(img, nwidth, nheight, p_heads=None, p_tails=None):

    img_height, img_width, _ = img.shape
    my_dpi=300.
    dx = img_width/nwidth
    dy = img_height/nheight

    # Set up figure
    fig=plt.figure(figsize=(float(img.shape[0])/my_dpi,float(img.shape[1])/my_dpi),dpi=my_dpi)
    ax=fig.add_subplot(111)

    # Remove whitespace from around the img
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    # Set the gridding interval: here we use the major tick interval
    # http://stackoverflow.com/questions/20368413/draw-grid-lines-over-an-image-in-matplotlib
    xloc = plticker.FixedLocator([i * dx for i in range(nwidth)])
    yloc = plticker.FixedLocator([i * dy for i in range(nheight)])
    ax.xaxis.set_major_locator(xloc)
    ax.yaxis.set_major_locator(yloc)
    ax.xaxis.set_tick_params(labelbottom='off')
    ax.yaxis.set_tick_params(labelbottom='off')

    # Add the grid
    ax.grid(which='major', axis='both', linestyle='-', color='w', linewidth=.1)

    # Add the img
    ax.imshow(img)

    # Add some labels to the gridsquares
    for j in range(nheight):
        y=dy/2+j*dy
        for i in range(nwidth):
            x=dx/2.+float(i)*dx
            ax.text(x,y,'{:d}'.format(i+j*nwidth),color='w',ha='center',va='center', size=2)

    return ax, (dx, dy)

def find_bucket(point, nwidth, nheight, dx, dy):
    '''search the buckets for head/tail'''
    for j in range(nheight):
        ymin = j * dy
        ymax = j * dy + dy - 1
        if point[1] >= ymin and point[1] < ymax:
            for i in range(nwidth):
                xmin = i * dx
                xmax = i * dx + dx - 1
                if point[0] >= xmin and point[0] < xmax:
                    return i+j*nwidth, (xmin, ymin)
    return None, None

def find_coord(bucket, nwidth, nheigth, dx, dy):
    '''given the bucket return the x,y coord'''
    row = bucket//nwidth
    col = bucket%nheigth
    return (col*dx, row*dy)


def process_command_line(argv):
    # create the top-level parser
    parser = argparse.ArgumentParser(description="programpurpose")
    subparsers = parser.add_subparsers(title='subcommands', description='valid subcommands', help='additional help', dest='subcommand')
    subparsers.required = True  # makes sure, that a subcommand is given

    parser.add_argument("-t", "--test", help="test", action="store")
    parser.add_argument("-l", "--log", help="define log level", choices=['d', 'i', 'w', 'e', 'c'])

    # create the parser for the "crop_image" command
    parser_crop_image = subparsers.add_parser('crop_image', aliases=['ci'])
    parser_crop_image.add_argument('-x', type=int, default=1)
    parser_crop_image.add_argument('-i', '--image', type=str, default=testImg)
    parser_crop_image.set_defaults(func=crop_image)  # defines the function to call

    # create the parser for the "crop" command
    parser_crop = subparsers.add_parser('crop', aliases=['cr'])
    parser_crop.set_defaults(func=crop)  # defines the function to call

    # create the parser for the "overlay" command
    parser_overlay = subparsers.add_parser('overlay', aliases=['ovl'])
    parser_overlay.add_argument('-a', '--annotate', action='store_true')
    parser_overlay.add_argument('-i', '--image', type=str, default=testImg)
    parser_overlay.add_argument('--head', type=int)  # 331
    parser_overlay.add_argument('--tail', type=int)  # 250
    parser_overlay.add_argument('-g', '--grid', type=int, default=20)
    parser_overlay.set_defaults(func=overlay)  # defines the function to call

    # create the parser for the "create_data" command
    parser_create_data = subparsers.add_parser('create_data', aliases=['cd'])
    parser_create_data.add_argument('-g', '--grid', type=int, default=20)
    parser_create_data.add_argument('-r', '--resize', type=int, default=224)
    parser_create_data.set_defaults(func=create_data)  # defines the function to call

    # create the parser for the "plot" command
    parser_plot = subparsers.add_parser('plot', aliases=['p'])
    parser_plot.add_argument('-x', type=int, default=1)
    parser_plot.add_argument('-i', '--image', type=str, default=testImg)
    parser_plot.add_argument('-r', '--resize', type=int)
    parser_plot.set_defaults(func=plot)  # defines the function to call

    args = parser.parse_args(argv)

    return args


def main(argv=None):
    args = process_command_line(argv)

    # structured logging
    logging.basicConfig(format="", stream=sys.stderr, level=logging.DEBUG)
    logcfg(sys.stderr, logging.INFO, RenderEnum.console)
    log = structlog.get_logger(__name__)

    twStart()

    ### run the subcommand
    args.func(args)

    twEnd()
    return 0  # success


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
