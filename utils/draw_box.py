# encoding: utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

DRAW_FONT_PATH =os.path.dirname(os.path.abspath(__file__)) + "/songti_regular.ttf"

def draw_wide_rectangle(draw, xmin, ymin, xmax, ymax,
                        color=(255, 0, 0), width=5):
    for iw in range(width):
        draw.rectangle((xmin + iw,
                        ymin + iw,
                        xmax - iw,
                        ymax - iw),
                       outline=color)
    return draw


def draw_rects_and_texts(img_obj, results, class_name_list,
                         threshold, font_path=None):
    """
    :param img_obj:
    :param results: empty list or 2D array
    :param class_names:
    :param font:
    :return: draw: drawed_obj, if len(results) == 0,
    drawed_obj is the original image
    """
    num_classes = len(class_name_list)
    # use color tables of plt.
    colors = (plt.cm.brg(np.linspace(0, 1, num_classes))
              * 255).astype(np.uint8)
    # Copy one, since Draw will directly propose PIL images.
    copy_img_obj = img_obj.copy()
    drawed = ImageDraw.Draw(copy_img_obj)

    for i in range(0, len(results)):
        # pdb.set_trace()
        score = results[i, -2]
        if (score > 0) and (score < threshold):
            continue

        label = int(results[i, -1])
        name = class_name_list[label]
        #         color = colors[label % num_classes]
        color = colors[label % num_classes][:-1]
        color = tuple(color.tolist())
        # colorsys.hsv_to_rgb()
        xmin = int(round(results[i, 0]))
        ymin = int(round(results[i, 1]))
        xmax = int(round(results[i, 2]))
        ymax = int(round(results[i, 3]))

        # draw_inloop
        draw_size = 3
        draw_wide_rectangle(drawed, xmin, ymin, xmax, ymax,
                            color, width=draw_size)
        display_text = '%s: %.2f' % (name, score)
        if font_path is None:
            drawed.text((xmin, ymin), display_text, fill=(255, 0, 0))
        else:
            fontsize = 26
            font = ImageFont.truetype(font_path, fontsize)
            drawed.text((xmin, ymin), display_text, fill=(255, 0, 0),
                        font=font)

    return copy_img_obj

def draw_rects(img_obj, results, class_name_list, threshold):
    """
    :param img_obj:
    :param results: empty list or 2D array
    :param class_names:
    :return: draw: drawed_obj, if len(results) == 0,
    drawed_obj is the original image
    """
    num_classes = len(class_name_list)
    # use color tables of plt.
    colors = (plt.cm.brg(np.linspace(0, 1, num_classes))
              * 255).astype(np.uint8)
    # Copy one, since Draw will directly propose PIL images.
    copy_img_obj = img_obj.copy()
    drawed = ImageDraw.Draw(copy_img_obj)

    for i in range(0, len(results)):
        # pdb.set_trace()
        score = results[i, -2]
        if (score > 0) and (score < threshold):
            continue

        label = int(results[i, -1])
        name = class_name_list[label]
        #         color = colors[label % num_classes]
        color = colors[label % num_classes][:-1]
        color = tuple(color.tolist())
        # colorsys.hsv_to_rgb()
        xmin = int(round(results[i, 0]))
        ymin = int(round(results[i, 1]))
        xmax = int(round(results[i, 2]))
        ymax = int(round(results[i, 3]))

        # draw_inloop
        draw_size = 3
        draw_wide_rectangle(drawed, xmin, ymin, xmax, ymax,
                            color, width=draw_size)
        display_text = '%s: %.2f' % (name, score)
        # if font_path is None:
        #     drawed.text((xmin, ymin), display_text, fill=(255, 0, 0))
        # else:
        #     fontsize = 26
        #     font = ImageFont.truetype(font_path, fontsize)
        #     drawed.text((xmin, ymin), display_text, fill=(255, 0, 0),
        #                 font=font)
    return copy_img_obj


def draw_rects_and_texts_var_linewidth(img_obj, results, class_name_list,
                                       threshold, font_path=None):
    """
    :param img_obj:
    :param results: empty list or 2D array
    :param class_names:
    :param font:
    :return: draw: drawed_obj, if len(results) == 0,
    drawed_obj is the original image
    """
    num_classes = len(class_name_list)
    # use color tables of plt.
    colors = (plt.cm.brg(np.linspace(0, 1, num_classes))
              * 255).astype(np.uint8)
    # Copy one, since Draw will directly propose PIL images.
    copy_img_obj = img_obj.copy()
    drawed = ImageDraw.Draw(copy_img_obj)

    for i in range(0, len(results)):
        score = results[i, -2]
        if (score > 0) and (score < threshold):
            continue

        label = int(results[i, -1])
        name = class_name_list[label]
        #         color = colors[label % num_classes]
        color = colors[label % num_classes][:-1]
        color = tuple(color.tolist())
        # colorsys.hsv_to_rgb()
        xmin = int(round(results[i, 0]))
        ymin = int(round(results[i, 1]))
        xmax = int(round(results[i, 2]))
        ymax = int(round(results[i, 3]))

        # draw_inloop
        draw_size = max(3, int(round((ymax - ymin) / 10.0)))
        draw_wide_rectangle(drawed, xmin, ymin, xmax, ymax,
                            color, width=draw_size)
        display_text = '%s: %.2f' % (name, score)
        if font_path is None:
            drawed.text((xmin, ymax), display_text, fill=(255, 0, 0))

        else:
            fontsize = 10 * draw_size
            font = ImageFont.truetype(font_path, fontsize)
            drawed.text((xmin, ymax), display_text, fill=(255, 0, 0),
                        font=font)

    return copy_img_obj


def draw_results(img,
                 results,
                 class_names,
                 threshold=0.5,
                 save_img=False,
                 img_path=None):
    """
    Input:
      img: ndarray, RGB 0-255, uint8
      results: ndarray, float [minx, miny, maxx, maxy, score, id]
    Return:
      result img: RGB 0-255, uint8
    """
    img_obj = Image.fromarray(img)
    drawed_obj = draw_rects(img_obj, results, class_names,
                                      threshold)
    # save the image.
    if save_img:
        assert img_path is not None, \
            'If save image, img_path must be assigned.'
        drawed_obj.save(img_path)

    return np.asarray(drawed_obj)


def draw_results_with_font(img, results,
                           class_name_list,
                           font_path=DRAW_FONT_PATH,
                           threshold=0.6,
                           save_img=False,
                           img_path=None):
    """
    Input:
      img: ndarray, RGB 0-255, uint8
      results: ndarray, float [minx, miny, maxx, maxy, score, id]
    Return:
      result img: RGB 0-255, uint8
    """
    img_obj = Image.fromarray(img)
    drawed_obj = draw_rects_and_texts(img_obj, results, class_name_list,
                                      threshold,
                                      font_path=font_path)
    # save the image.
    if save_img:
        assert img_path is not None, \
            'If save image, img_path must be assigned.'
        drawed_obj.save(img_path)

    return np.asarray(drawed_obj)


if __name__ == '__main__':
    img_path = '15.jpg'
    img_arr = np.array(Image.open(img_path).convert('RGB'))
    # det_results是一个box的列表。box的每个元素是六维的列表或者数组，
    # 分别是[minx, miny, maxx, maxy, score, id]
    det_results = np.array([[5, 243, 996 , 995, 0.7, 1]])
    class_name_list = ['横向裂纹', '纵向裂纹']
    threshold = 0
    font_path = './songti_regular.ttf'
    drawed_img_arr = draw_results_with_font(img_arr, det_results,class_name_list)
    # drawed_img_arr = draw_results(img_arr, det_results, class_name_list)
    dst_path = 'test_result.jpg'
    Image.fromarray(drawed_img_arr).save(dst_path)
