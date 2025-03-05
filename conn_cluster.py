# import numpy as np
# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import connected_components

# def findConnClusterHD(avgOppChnn, segments, resultList, isolatedList, maxnum):
#     # 合并并去重邻接列表
#     reList = np.unique(np.vstack([resultList, np.fliplr(resultList)]), axis=0)
#     segLabel = np.unique(segments)

#     # 构建图的稀疏矩阵
#     s = reList[:, 0]
#     t = reList[:, 1]
#     G = csr_matrix((np.ones(len(s)), (s, t)), shape=(maxnum, maxnum))

#     # 查找连通分量
#     conn_num, cluster_idx_List = connected_components(G, directed=True, return_labels=True)

#     # 对结果列表中的点进行升序排序
#     ascend_sort_dot_idx_list = np.sort(np.unique(resultList))
#     Asd = ascend_sort_dot_idx_list

#     # 初始化视图图像
#     viewImg = -np.ones(segments.shape)

#     # 为每个连通分量分配标签
#     for k in Asd:
#         viewImg[segments == segLabel[k]] = cluster_idx_List[k]

#     # 处理离群点（注释掉的部分）
#     # for out_idx in isolatedList:
#     #     viewImg[segments == out_idx] = np.max(viewImg) + 1

#     return viewImg

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def findConnClusterHD(avg_opp_chnn, segments, result_list, isolated_list, maxnum):
    # 生成双向邻接表
    re_list = np.unique(np.vstack([result_list, np.fliplr(result_list)]), axis=0)
    seg_label = np.unique(segments)
    
    # 构建稀疏邻接矩阵
    s = re_list[:, 0].astype(int)
    t = re_list[:, 1].astype(int)
    print(len(s), len(t))
    print(maxnum)
    G = csr_matrix((np.ones_like(s), (s, t)), shape=(maxnum, maxnum))
    
    # 计算连通分量
    conn_num, cluster_idx_list = connected_components(G, directed=True, connection='weak')
    
    # 生成排序后的节点列表
    ascend_sort_dot_idx_list = np.sort(np.unique(result_list.astype(int)))
    
    # 初始化结果矩阵
    view_img = np.full_like(segments, -1, dtype=int)
    
    # 映射聚类结果到图像矩阵
    for k in ascend_sort_dot_idx_list:
        mask = segments == seg_label[k]
        view_img[mask] = cluster_idx_list[k]
    
    print('oo')  # 保留原始调试输出
    return view_img