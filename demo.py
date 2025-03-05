import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Delaunay
import os
from feature import calFeatureHD
from length_ratio_list import calLengthRatioListHD
from sec_ratio import calSecRatioHD
from conn_cluster import findConnClusterHD
from adj_metrix import findAdjMetrixHD
from threshold import findTreshold
from wight_num import wightNum
from visual_rebuild import visualRebuildHD

# 清空变量，关闭所有图形窗口
# 这里 Python 没有直接对应的 clear 和 close all 命令

# 添加路径，Python 中可以使用 sys.path.append
import sys
sys.path.append('..\\testData\\')
sys.path.append('.\\CGVSsalient\\')
sys.path.append('E:\\PandaSpaceSyn\\WorkingSpace\\Tolerance\\ToleranceModel\\Evaluation\\BSDSdata\\BSR_bsds500\\BSDS500\\data\\images\\train\\')

# 加载参数
wn1 = 1  # 10
wn = 1
wf = 0.1

# 定义图像文件名
Img_name_string = 'D:\\Data\\BSDS500\\data\\images\\train\\train\\12003.jpg'
res_name = os.path.splitext(Img_name_string)[0]
res_wn_string = f'.\\hardresults\\{res_name}_{wf}'

# 加载图像特征
srcimage = cv2.imread(Img_name_string)
if srcimage is None:
    print(f"无法读取图像: {Img_name_string}")
    sys.exit(1)

w, h, d = srcimage.shape

if d < 3:
    srcimage = np.dstack([srcimage] * 3)

# 显示图像
plt.figure()
plt.imshow(cv2.cvtColor(srcimage, cv2.COLOR_BGR2RGB))
plt.show()

# 计算特征
import time
start_time = time.time()
# [colorList, feaLoc, feaCues, segments] = calFeatureHD(srcimage);
# srcData = [colorList feaCues feaLoc];
colorList, feaLoc, segments = calFeatureHD(srcimage)
srcData = np.hstack([colorList, feaLoc])
# print(segments.shape)
# srcData = [colorList];

# 可视化特征
# plt.figure()
# plt.scatter(srcData[:, 0], srcData[:, 1], srcData[:, 2])
# plt.show()

# 使用 KNN 计算邻接列表
k = 4
D = squareform(pdist(srcData, 'euclidean'))
print(D.shape)
sortval = np.sort(D, axis=1)
print(sortval.shape)
# print(sortval)
sortpos = np.argsort(D, axis=1)
print(sortpos.shape)
# print(sortpos)
neighborIds = sortpos[:, 1:k + 1]
print(neighborIds.shape)
# print(neighborIds)
adjList = []
for i in range(len(srcData)):
    for j in range(k):
        adjList.append([i, neighborIds[i, j]])
adjList = np.array(adjList)
print(adjList.shape)
# 计算长度比例列表
task = 1  # 1 for 1st order, 2 for 2nd
dataList, ratioList = calLengthRatioListHD(adjList, task, srcData, wf)
# print(dataList.shape)
# print(ratioList)
# 排序并绘制 Q 曲线
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
th1 = 0.12 * wn1  # weightNum; # 0.12

T = findTreshold(Q, th1, orderLabel)  # 阈值 for Q;
wl = T
# wl = np.max(Q)
print('aa')

# 排序数据列表
oList = np.argsort(ratioList)
sortedDataList = dataList[oList]
resDataList = sortedDataList[Q <= wl]

# print(len(ratioList))
# print(sortedDataList.shape)
# print(resDataList.shape)

# 计算二阶比例列表
secDataList, secRatioList = calSecRatioHD(resDataList, srcData, wf)
print(len(secRatioList))
print(secDataList)
# 排序并绘制 Q2 曲线
Q2 = np.sort(secRatioList)
ymax = np.max(Q2)
plt.figure()
plt.plot(Q2, linewidth=2)
plt.title('Q2')
plt.ylim([0, ymax])
plt.grid(True)
plt.show()

orderLabel = 2
th = 0.035 * wn
T2 = findTreshold(Q2, th, orderLabel)  # 阈值 for Q; wl = T.
maxnum = np.max(secDataList)+1
oList = np.argsort(secRatioList)
sortedDataList = secDataList[oList]
resList = sortedDataList[Q2 <= T2]

# 可视化重建
isolatedList1 = np.setdiff1d(np.unique(segments), np.unique(resDataList))
isolatedList = np.setdiff1d(np.unique(segments), np.unique(resList))

viewImg = findConnClusterHD(srcData, segments, resList, isolatedList, maxnum)
end_time = time.time()
print(f"耗时: {end_time - start_time} 秒")

plt.figure()
plt.imshow(viewImg)
plt.colorbar()
plt.axis('off')
plt.show()

# viewImg = visualRebuildHD(srcimage, srcData, segments, resList, isolatedList); # two step

# 边缘检测
BW = cv2.Canny(viewImg.astype(np.uint8), 0.001, 0.001)
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
I2 = cv2.dilate(BW, se)

plt.figure()
plt.imshow(I2, cmap='gray')
plt.show()

resizeImg = cv2.resize(srcimage, (I2.shape[1], I2.shape[0]))
view_Res_Img = resizeImg.copy()
view_Res_Img[:, :, 0] = cv2.add(resizeImg[:, :, 0], I2 * 255)
view_Res_Img[:, :, 1] = cv2.add(resizeImg[:, :, 1], I2 * 255)
view_Res_Img[:, :, 2] = cv2.add(resizeImg[:, :, 2], I2 * 255)

plt.figure()
plt.imshow(cv2.cvtColor(view_Res_Img, cv2.COLOR_BGR2RGB))
plt.show()

resized_mask = cv2.resize(I2, (h, w))
cv2.imwrite(f'{res_wn_string}_mask.png', resized_mask)
resized_view_Res_Img = cv2.resize(view_Res_Img, (h, w))
cv2.imwrite(f'{res_wn_string}.png', resized_view_Res_Img)

# FlattenedData = viewImg.flatten()
# MappedFlattened = mapminmax(FlattenedData, 0, 1)
# MappedData = MappedFlattened.reshape(viewImg.shape)
# resmap = cv2.normalize(MappedData, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# cv2.imwrite('resmap.png', resmap)