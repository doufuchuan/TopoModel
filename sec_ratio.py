# import numpy as np

# def calSecRatioHD(resDataList, feaList, wf=0.2):
#     # 合并正反序的 resDataList
#     adjList = np.vstack((resDataList, np.fliplr(resDataList)))
    
#     # 获取唯一的第一个元素
#     uniList = np.unique(adjList[:, 0])
#     secList = []
    
#     for uItem in uniList:
#         tempList = adjList[:, 0]
#         uRows = np.where(tempList == uItem)[0]
#         secListTemp = []
        
#         for un in uRows:
#             secListTempLocation = tempList == adjList[un, 1]
#             secListTemp.append(adjList[secListTempLocation, :])
        
#         secList.append(np.unique(np.vstack(secListTemp), axis=0))
    
#     adjSecList = secList
#     uniSecLoc = uniList
    
#     # 计算 secRatioList 和 secDataList
#     len_adjSecList = len(adjSecList)
#     dataListTemp = []
#     ratioList = []
    
#     for cNum in range(len_adjSecList):
#         cList = adjSecList[cNum]
#         uniNum = np.unique(cList[:, 0])
#         cellDataList = []
#         cellLenLine = []
        
#         for item in uniNum:
#             locTemp = cList[:, 0] == item
#             a = cList[locTemp, 0]
#             b = cList[locTemp, 1]
#             lenItemLine = []
            
#             for i in range(len(a)):
#                 lenItemLineTemp_color = np.sum((feaList[a[i], 0:3] - feaList[b[i], 0:3]) ** 2)
#                 lenItemLineTemp_position = np.sum((feaList[a[i], 3:] - feaList[b[i], 3:]) ** 2)
#                 lenItemLineTemp = np.sqrt(wf * wf * lenItemLineTemp_position + lenItemLineTemp_color)
#                 lenItemLine.append(lenItemLineTemp)
            
#             cellDataList.append(cList[locTemp, :])
#             cellLenLine.extend(lenItemLine)
        
#         dataListTemp.append(cellDataList)
#         # print(cellLenLine)
#         ratioList.extend(np.array(cellLenLine) / (1 if len(cellLenLine)==0 else np.min(cellLenLine) + np.finfo(float).eps))
    
#     secRatioList = ratioList
#     secDataList = dataListTemp
    
#     # 输出结果
#     print('oo')
#     return secDataList, secRatioList

import numpy as np

def calSecRatioHD(res_data_list, fea_list, wf):
    # 构建双向邻接列表
    adj_list = np.vstack([res_data_list, np.fliplr(res_data_list)])
    
    uni_list = np.unique(adj_list[:, 0])
    sec_list = []
    
    # 构建二阶邻接关系
    for u_item in uni_list:
        temp_list = adj_list[:, 0]
        u_rows = np.where(temp_list == u_item)[0]
        sec_list_temp = []
        
        for un in u_rows:
            sec_list_temp_location = temp_list == adj_list[un, 1]
            sec_list_temp.extend(adj_list[sec_list_temp_location].tolist())
        
        # 去重处理
        sec_list.append(np.unique(sec_list_temp, axis=0))
    
    adj_sec_list = sec_list
    data_list = []
    ratio_list = []

    # 主计算循环
    for c_num in range(len(adj_sec_list)):
        c_list = adj_sec_list[c_num]
        uni_num = np.unique(c_list[:, 0])
        cell_data_list = []
        cell_len_line = []
        
        for item in uni_num:
            loc_temp = c_list[:, 0] == item
            a = c_list[loc_temp, 0]
            b = c_list[loc_temp, 1]
            len_item_line = []
            
            # 高维特征距离计算
            for i in range(len(a)):
                # 调整MATLAB索引到Python索引（重要！）
                color_diff = fea_list[int(a[i])-1, 0:3] - fea_list[int(b[i])-1, 0:3]
                position_diff = fea_list[int(a[i])-1, 3:] - fea_list[int(b[i])-1, 3:]
                
                # 加权距离公式
                distance = np.sqrt(wf**2 * np.sum(position_diff**2) + np.sum(color_diff**2))
                len_item_line.append(distance)
            
            # 收集计算结果
            cell_data_list.extend(c_list[loc_temp].tolist())
            cell_len_line.extend(len_item_line)
        
        # 计算比例系数
        min_len = np.min(cell_len_line) + np.finfo(float).eps
        ratio_list.extend(np.array(cell_len_line) / min_len)
        data_list.extend(cell_data_list)
    
    print('oo')  # 保持原始调试输出
    return np.array(data_list), np.array(ratio_list)