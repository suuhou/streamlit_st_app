import random
import numpy as np
import urllib
import io
import streamlit as st
import torchvision.transforms as transforms
from convnext import *
from PIL import Image

def get_size(size, div = 16):

    def find_closest(s):
        q_ = int(s / div)
        s1 = div * q_
        s2 = div * (q_ + 1)
        fine_s = s1 if abs(s-s1) < abs(s-s2) else s2
        return fine_s

    w, h = size
    new_w = find_closest(w)
    new_h = find_closest(h)

    return [new_h, new_w]


def get_transform(osize, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    transform_list.append(transforms.Resize(osize, method))

    #transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=16, method=method)))

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
        print("The image size needs to be a multiple of 16. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 16" % (ow, oh, w, h))
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


@st.cache(show_spinner=False, allow_output_mutation=True)
def load_pth_from_url():

    load_path = r'http://cz.coder17.com/suuhou/concatcanvas2_layernorm_pth/latest_net_G.pth'
    with urllib.request.urlopen(load_path) as response:
        state_dict = torch.load(io.BytesIO(response.read()))

    return state_dict

@st.cache(show_spinner=False)
def load_local_canvas():

    height_paper_path = r'E:/python_work/streamlit_app_repo/streamlit_app/images/height_1000_2.png'
    texture_paer_path = r'E:/python_work/streamlit_app_repo/streamlit_app/images/paper_1000_2.png'

    height_paper = Image.open(height_paper_path).convert('RGB')
    texture_paer = Image.open(texture_paer_path).convert('RGB')

    return height_paper, texture_paer

def load_local_image():

    image_url = r'http://cz.coder17.com//suuhou/images_forfid_layernorm2/fake_{}.png'
    image_list = [image_url.format(str(random.randint(2000, 4000))) for _ in range(18)]

    return image_list


# @st.cache(show_spinner=False)
# def load_local_image(url):
#     with urllib.request.urlopen(url) as response:
#         image = np.array(bytearray(response.read()), dtype='uint8')
#     return image


def main(image):

    model = ConvNeXtGenerator()

    state_dict = load_pth_from_url()
    height_paper, texture_paer = load_local_canvas()

    net = model

    if isinstance(model, torch.nn.DataParallel):
        net = model.module
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    net.load_state_dict(state_dict)

    new_size = get_size(image.size)
    transformation = get_transform(new_size, grayscale=False)
    input = transformation(image).unsqueeze(0)
    canvas_height = transformation(height_paper).unsqueeze(0)
    canvas_paper = transformation(texture_paer).unsqueeze(0)
    tensor_images = [input, canvas_height, canvas_paper]

    with torch.no_grad():
        fake_b = model(torch.cat(tensor_images, dim=1))

    np_image = tensor2im(fake_b, imtype=np.uint8)
    image_pil = Image.fromarray(np_image)

    return image_pil

if __name__ == '__main__':
    image = Image.open(r'E:\python_work\streamlit_app_repo\streamlit_app\test2.png').convert('RGB')
    print('osize', image.size)

    new_image = main(image)
    print('new_size', image.size)

    # t = transforms.ToTensor()
    # t2 = transforms.ToPILImage()
    #
    # tensor_image = t(image)
    # print('tensor_image.shape', tensor_image.shape)
    #
    # back_image = t2(tensor_image)
    # print('back_image', back_image.size)

