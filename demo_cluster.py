import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import cv2
from skimage.segmentation import slic
from skimage.util import img_as_float
import time
from feature import calFeatureHD
from length_ratio_list import calLengthRatioListHD
from sec_ratio import calSecRatioHD
from conn_cluster import findConnClusterHD
from adj_metrix import findAdjMetrixHD
from threshold import findTreshold
from wight_num import wightNum
from visual_rebuild import visualRebuildHD
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from skimage import io, filters, morphology
import time
from PIL import Image

# 高维度聚类演示
# --------------------------

# 输入 maxLength: maxR
# dimension amount : Dim
# --------------------------------------

# 清除变量并关闭所有图形
plt.close('all')

# 添加路径
import sys
sys.path.append('..\\testData\\')

# 加载图像特征
srcimage = io.imread('test.png')
plt.figure()
plt.imshow(srcimage)
plt.show()

start_time = time.time()

# 计算高维特征
feaList, feaLoc, segments = calFeatureHD(srcimage)
srcData = feaList  # 或者 srcData = np.hstack((feaList, feaLoc))

# 绘制三维散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(srcData[:, 0], srcData[:, 1], srcData[:, 2])
plt.show()

# 参数结构
param = {'maxR': 10}

# Delaunay 三角剖分
TRI = Delaunay(srcData).simplices
param['Dm'] = TRI.shape[1]

# 计算邻接矩阵
adjList = findAdjMetrixHD(TRI, param)

# 计算长度比例列表
task = 1  # 1 表示一阶，2 表示二阶
dataList, ratioList = calLengthRatioListHD(adjList, task, srcData)
print(ratioList)
# 排序并绘制 Q
Q = np.sort(ratioList)
ymax = np.max(Q)
plt.figure()
plt.plot(Q, linewidth=2)
plt.title('Q')
plt.ylim([0, ymax])
plt.grid(True)
plt.show()

orderLabel = 1
NumQ1 = len(Q)
weightNum = wightNum(len(Q), orderLabel)
th = 0.12 * weightNum

T = findTreshold(Q, th, orderLabel)  # Q 的阈值
wl = T
print('aa')

# 排序数据列表
sortedDataList = np.array([dataList[i] for i in np.argsort(Q)])

# 一阶结果
resDataList = sortedDataList[Q <= T]

# 二阶计算
secDataList, secRatioList = calSecRatioHD(resDataList, feaList)

# 排序并绘制 Q2
Q2 = np.sort(secRatioList)
ymax = np.max(Q2)
plt.figure()
plt.plot(Q2, linewidth=2)
plt.title('Q2')
plt.ylim([0, ymax])
plt.grid(True)
plt.show()

orderLabel = 2
wn = 1
NumQ2 = len(Q2)
weightNum = wightNum(len(Q2), orderLabel)
th = 0.035 * weightNum * wn

T = findTreshold(Q2, th, orderLabel)  # Q2 的阈值

# 排序数据列表
sortedDataList = np.array([secDataList[i] for i in np.argsort(Q2)])

# 二阶结果
resList = sortedDataList[Q2 <= T]

# 可视化重建
isolatedList = np.setdiff1d(np.unique(segments), np.unique(resList))  # 孤立节点或异常值
viewImg = visualRebuildHD(srcimage, srcData, segments, resList, isolatedList)

# 边缘检测
BW = filters.canny(viewImg, sigma=0.001)
se = morphology.disk(2)
I2 = morphology.dilation(BW, se)
plt.figure()
plt.imshow(I2, cmap='gray')
plt.show()

# 调整图像大小
resizeImg = np.array(Image.fromarray(srcimage).resize(I2.shape))

# 叠加边缘检测结果
view_Res_Img = resizeImg.copy()
view_Res_Img[:, :, 0] = resizeImg[:, :, 0] + (I2 * 255).astype(np.uint8)
view_Res_Img[:, :, 1] = resizeImg[:, :, 1] + (I2 * 255).astype(np.uint8)
view_Res_Img[:, :, 2] = resizeImg[:, :, 2] + (I2 * 255).astype(np.uint8)

plt.figure()
plt.imshow(view_Res_Img)
plt.show()
io.imsave('view_Res_Img.png', view_Res_Img)

print(f"Elapsed time: {time.time() - start_time} seconds")

# # 清除变量，关闭所有图形窗口
# plt.close('all')

# # 添加数据路径
# # 这里 Python 没有直接对应的 addpath 函数，需要确保文件路径正确
# # 假设数据文件和代码在同一目录下

# # 加载图像特征
# srcimage = cv2.imread('test.png')
# srcimage = cv2.cvtColor(srcimage, cv2.COLOR_BGR2RGB)
# plt.figure()
# plt.imshow(srcimage)
# plt.show()

# start_time = time.time()

# # 计算特征
# feaList, feaLoc, segments = calFeatureHD(srcimage)
# srcData = feaList

# # 绘制 3D 散点图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(srcData[:, 0], srcData[:, 1], srcData[:, 2])
# plt.show()

# # 参数设置
# param = {'maxR': 10}

# # 进行 Delaunay 三角剖分
# TRI = Delaunay(srcData).simplices
# param['Dm'] = TRI.shape[1]

# # 查找邻接矩阵
# adjList = findAdjMetrixHD(TRI, param)

# # 计算长度比例列表
# task = 1  # 1 for 1st order, 2 for 2nd
# dataList, ratioList = calLengthRatioListHD(adjList, task, srcData)

# # 对比例列表排序
# Q = np.sort(ratioList)
# ymax = np.max(Q)

# # 绘制 Q 图
# plt.figure()
# plt.plot(Q, linewidth=2)
# plt.title('Q')
# plt.ylim([0, ymax])
# plt.grid(True)
# plt.show()

# orderLabel = 1
# NumQ1 = len(Q)
# # 假设 wightNum 函数的实现
# def wightNum(length_Q, orderLabel):
#     return 1  # 简单返回 1 作为示例
# weightNum = wightNum(NumQ1, orderLabel)
# th = 0.12 * weightNum

# # 查找阈值
# T = findTreshold(Q, th, orderLabel)
# wl = T
# print('aa')

# # 排序数据列表
# oList = np.argsort(ratioList)
# sortedDataList = dataList[oList]

# # 一阶结果
# resDataList = sortedDataList[Q <= T]

# # 异常值移除
# secDataList, secRatioList = calSecRatioHD(resDataList, feaList)

# # 对二阶比例列表排序
# Q2 = np.sort(secRatioList)
# ymax = np.max(Q2)

# # 绘制 Q2 图
# plt.figure()
# plt.plot(Q2, linewidth=2)
# plt.title('Q2')
# plt.ylim([0, ymax])
# plt.grid(True)
# plt.show()

# orderLabel = 2
# wn = 1
# NumQ2 = len(Q2)
# weightNum = wightNum(NumQ2, orderLabel)
# th = 0.035 * weightNum * wn

# # 查找二阶阈值
# T = findTreshold(Q2, th, orderLabel)

# # 排序二阶数据列表
# oList = np.argsort(secRatioList)
# sortedDataList = secDataList[oList]

# # 二阶结果
# resList = sortedDataList[Q2 <= T]

# # 可视化重建
# isolatedList = np.setdiff1d(np.unique(segments), np.unique(resList))
# viewImg = visualRebuildHD(srcimage, srcData, segments, resList, isolatedList)

# # 边缘检测
# BW = cv2.Canny(viewImg.astype(np.uint8), 0.001, 0.001)
# se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
# I2 = cv2.dilate(BW, se)

# # 显示边缘图
# plt.figure()
# plt.imshow(I2, cmap='gray')
# plt.show()

# # 调整原图像大小
# resizeImg = cv2.resize(srcimage, (I2.shape[1], I2.shape[0]))

# # 合并边缘图和原图像
# view_Res_Img = resizeImg.copy()
# view_Res_Img[:, :, 0] = np.clip(resizeImg[:, :, 0] + I2 * 255, 0, 255).astype(np.uint8)
# view_Res_Img[:, :, 1] = np.clip(resizeImg[:, :, 1] + I2 * 255, 0, 255).astype(np.uint8)
# view_Res_Img[:, :, 2] = np.clip(resizeImg[:, :, 2] + I2 * 255, 0, 255).astype(np.uint8)

# # 显示合并后的图像
# plt.figure()
# plt.imshow(view_Res_Img)
# plt.show()

# # 保存图像
# cv2.imwrite('view_Res_Img.png', cv2.cvtColor(view_Res_Img, cv2.COLOR_RGB2BGR))

# end_time = time.time()
# print(f"运行时间: {end_time - start_time} 秒")