import numpy as np
import cv2
from skimage.color import rgb2lab
from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
import math
def calFeatureHD(src):
    # 获取图像尺寸
    h, w, _ = src.shape
    print("Image Size:", h, w)

    # 调整图像大小
    if h <= w:
        W = 500
        H = math.ceil(500 * h / w)
    else:
        H = 500
        W = math.ceil(w / h * 500)
    print("Resized Image Size:", H, W)
    src_img = cv2.resize(src, (W, H), interpolation=cv2.INTER_LINEAR)
    # plt.imshow(src_img)
    # plt.title('Source Image')
    # plt.show()
    # 超像素分割参数
    region_size = 50
    regularizer = 10

    # 是否使用 Lab 颜色空间
    use_lab = True
    if use_lab:
        im = rgb2lab(src_img).astype(np.float32)
        r_im = im[:, :, 0]  # L
        g_im = im[:, :, 1]  # a
        b_im = im[:, :, 2]  # b
    else:
        im = img_as_float(src_img)
        r_im = im[:, :, 0]
        g_im = im[:, :, 1]
        b_im = im[:, :, 2]

    # 初始化输出图像
    des_im = np.zeros_like(im)
    r_des_im = des_im[:, :, 0]
    g_des_im = des_im[:, :, 1]
    b_des_im = des_im[:, :, 2]

    # 使用 SLIC 进行超像素分割
    # segments = slic(im, n_segments=H*W/2500, compactness=regularizer) + 1
    slic = cv2.ximgproc.createSuperpixelSLIC(im, algorithm=cv2.ximgproc.SLIC,region_size=region_size, ruler=regularizer)
    slic.iterate(30)
    segments = slic.getLabels() + 1

    # 计算每个超像素的平均值
    sp_label = np.unique(segments)
    for item in sp_label:
        idx = segments == item
        r_des_im[idx] = np.mean(r_im[idx])
        g_des_im[idx] = np.mean(g_im[idx])
        b_des_im[idx] = np.mean(b_im[idx])

    des_im = np.dstack((r_des_im, g_des_im, b_des_im))
    # plt.imshow(des_im)
    # plt.title('Desired Image')
    # plt.show()
    # 计算特征
    mean_node = []
    fea_loc = []
    loc = []

    for i in sp_label:
        idx = segments == i
        xx, yy = np.where(idx)
        xc = np.mean(xx)
        yc = np.mean(yy)
        loc.append([xc, yc])

        mean_node_r = np.mean(r_des_im[idx])
        mean_node_g = np.mean(g_des_im[idx])
        mean_node_b = np.mean(b_des_im[idx])
        mean_node.append([mean_node_r, mean_node_g, mean_node_b])

    # 归一化特征
    rgb_chnn = np.array(mean_node)
    fea_loc = 100 * (np.array(loc) - np.min(loc)) / (np.max(loc) - np.min(loc) + np.finfo(float).eps)
    fea_list = 100 * (rgb_chnn - np.min(rgb_chnn)) / (np.max(rgb_chnn) - np.min(rgb_chnn) + np.finfo(float).eps)

    print("Features done!")
    return fea_list, fea_loc, segments

# # # # 示例调用
# src = cv2.imread('D:\\Data\\BSDS500\\data\\images\\train\\train\\12003.jpg')
# src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

# fea_list, fea_loc, segments = calFeatureHD(src)
# print("Feature List:", fea_list)
# print("Feature Location:", fea_loc)
# print("Segments:", segments)
# plt.imshow(segments)
# plt.show()