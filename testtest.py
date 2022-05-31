import random
import numpy as np
import urllib
import io
import streamlit as st
import torchvision.transforms as transforms
from convnext_v2 import *
from PIL import Image

def get_params(size):
    w, h = size
    new_h = h
    new_w = w

    x = random.randint(0, np.maximum(0, new_w - 512))
    y = random.randint(0, np.maximum(0, new_h - 512))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    osize = [512, 512]
    transform_list.append(transforms.Resize(osize, method))

    if params is None:
        transform_list.append(transforms.RandomCrop(512))
    else:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], 512)))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

def tensor2im(input_image, imtype=np.uint8):

    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


@st.cache(show_spinner=False)
def load_pth_from_url(url):

    with urllib.request.urlopen(url) as response:
        state_dict = torch.load(io.BytesIO(response.read()))

    return state_dict

@st.cache(show_spinner=False)
def load_local_image(url):
    with urllib.request.urlopen(url) as response:
        image = np.array(bytearray(response.read()), dtype='uint8')
    return image

uploaded_file = st.sidebar.file_uploader(" ")
image = load_local_image(uploaded_file)


def main(image):
    model = ConvNeXtGenerator_v2()

    load_path = r'http://cz.coder17.com/suuhou/convnext_v2_pth/latest_net_G.pth'
    state_dict = load_pth_from_url(load_path)

    net = model
    if isinstance(model, torch.nn.DataParallel):
        net = model.module
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    net.load_state_dict(state_dict)

    transform_params = get_params(image.size)
    transformation = get_transform(transform_params, grayscale=False)
    tensor_image = transformation(image).unsqueeze(0)

    with torch.no_grad():
        fake_b = model(tensor_image)

    np_image = tensor2im(fake_b, imtype=np.uint8)
    print(fake_b.shape)
    image_pil = Image.fromarray(np_image)

    return image_pil