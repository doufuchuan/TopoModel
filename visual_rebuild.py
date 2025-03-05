import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def visualRebuildHD(srcImg, avgOppChnn, segments, resultList, isolatedList):
    # 合并并去重邻接列表
    reList = np.vstack([resultList, np.fliplr(resultList)])
    reList = np.unique(reList, axis=0)

    # 获取分割标签
    segLabel = np.unique(segments)

    # 对特征进行去重
    C = np.unique(avgOppChnn, axis=0)

    newList = []
    for i in range(len(C)):
        segTemp = []
        for j in range(len(avgOppChnn)):
            if np.array_equal(avgOppChnn[j], C[i]):
                segTemp.append(segLabel[j])
        if len(segTemp) > 1:
            from itertools import combinations
            nk = np.array(list(combinations(segTemp, 2)))
            newList.extend(nk)

    newList = np.array(newList)
    reList = np.vstack([reList, newList])
    reList = np.unique(reList, axis=0)

    # 计算节点数量
    nodeNum = len(reList[:, 0]) + len(isolatedList)

    # 创建有向图
    G = nx.DiGraph()
    for i in range(1, nodeNum + 1):
        G.add_node(i)
    for edge in reList:
        G.add_edge(edge[0], edge[1])

    # 绘制图形
    plt.figure()
    pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue', font_size=10, font_color='black', edge_color='gray')
    plt.show()

    # 查找连通分量
    bins = list(nx.connected_components(G.to_undirected()))
    bin_dict = {}
    for i, bin in enumerate(bins, start=1):
        for node in bin:
            bin_dict[node] = i

    # 初始化视图图像
    viewImg = np.zeros_like(segments)

    # 为每个分割区域分配连通分量标签
    for idx in range(len(segLabel)):
        if idx < len(bins):
            viewImg[segments == segLabel[idx]] = bin_dict.get(segLabel[idx], 0)
        else:
            viewImg[segments == segLabel[idx]] = 0

    print('11')
    return viewImg