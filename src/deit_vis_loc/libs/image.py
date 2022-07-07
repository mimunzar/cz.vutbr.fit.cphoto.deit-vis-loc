#!/usr/bin/env python3

import math
import os

import PIL.Image
import PIL.ImageOps
import Imath
import torch
import torchvision.transforms.functional as T

import src.deit_vis_loc.libs.util as util
import src.deit_vis_loc.libs.image_exr as image_exr


def im_read(fpath, exif_transpose):
    with PIL.Image.open(fpath, 'r') as im:
        if exif_transpose:
            im = PIL.ImageOps.exif_transpose(im)
        return T.to_tensor(im.convert('RGB'))


def im_write(fpath, tensor):
    T.to_pil_image(tensor).save(fpath)
    return (fpath, tensor)


FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

def exr_read(fpath, _):
    with image_exr.open(fpath, 'r') as im:
        dw = im.read_header()['dataWindow']
        ch = im.read_channel({'channel': 'R', 'type': FLOAT})
        return torch.frombuffer(ch, dtype = torch.float32) \
            .reshape(dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1) \
            .repeat(3, 1, 1)


def exr_write(fpath, tensor):
    data = tensor[0]
    with image_exr.open(fpath, 'w') as im:
        im.write_header({'shape': data.shape, 'channels': {'R': Imath.Channel(FLOAT)}})
        im.write_channel({'R': data.numpy().tobytes()})
    return (fpath, tensor)


EXRImage       = {'read': exr_read, 'write': exr_write }
Image          = {'read': im_read,  'write': im_write }
EXT_DISPATCHER = {
    '.PNG'  : Image,
    '.JPG'  : Image,
    '.JPEG' : Image,
    '.EXR'  : EXRImage,
}

def extension(fpath):
    return util.second(os.path.splitext(fpath)).upper()


def read_exif_transpose(fpath):
    return read(fpath, exif_transpose=True)


def read(fpath, exif_transpose=False):
    return EXT_DISPATCHER[extension(fpath)]['read'](fpath, exif_transpose)


def write(fpath, tensor):
    return EXT_DISPATCHER[extension(fpath)]['write'](fpath, tensor)


def scale(ratio, im):
    w, h  = im[0].shape
    dim_x = round(w*ratio)
    dim_y = round(h*ratio)
    return T.resize(im, [dim_x, dim_y], antialias=True)


def scale_to_fit(input_size, im):
    return scale(input_size/max(im[0].shape), im)


RENDER_FOV = math.radians(60)

def scale_by_fov(input_size, image_fov, im):
    im_re_fov = image_fov/RENDER_FOV
    re_im_px  = input_size/max(im[0].shape)
    return scale(re_im_px*im_re_fov, im)
    #^ Scale image first to be same size as renders and than scale it by its
    # FOV. Avoid upsampling by computing scale factor once.


def center_crop(input_size, im):
    w, h   = im[0].shape
    top_x = round((w - input_size)/2)
    top_y = round((h - input_size)/2)
    return T.crop(im, top_x, top_y, input_size, input_size)


def pad_to_square(input_size, im):
    w, h   = im[0].shape
    top_x  = round((input_size - w)/2)
    top_y  = round((input_size - h)/2)
    new_im = torch.zeros(3, input_size, input_size)
    new_im[:, top_x:(top_x + w), top_y:(top_y + h)] = im
    return new_im

