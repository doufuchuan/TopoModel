import numpy as np

def calLengthRatioListHD(adj_list, task, fea_list, wf=0.2):
    if task == 1:
        uni_num = np.unique(adj_list[:, 0])
        len_line = []
        line_ratio = []
        data_list = []
        
        for item in uni_num:
            loc_temp = adj_list[:, 0] == item
            a = adj_list[loc_temp, 0]
            b = adj_list[loc_temp, 1]
            len_item_line = []
            
            for i in range(len(a)):
                # 计算颜色和位置的差异
                len_item_line_temp_color = np.sum((fea_list[a[i], :3] - fea_list[b[i], :3]) ** 2)
                len_item_line_temp_position = np.sum((fea_list[a[i], 3:] - fea_list[b[i], 3:]) ** 2)
                
                # 综合计算长度
                len_item_line_temp = np.sqrt(wf * wf * len_item_line_temp_position + len_item_line_temp_color)
                len_item_line.append(len_item_line_temp)
            
            data_list.append(adj_list[loc_temp, :])
            len_line.extend(len_item_line)
            line_ratio.extend(np.array(len_item_line) / (np.min(len_item_line) + np.finfo(float).eps))
        
        data_list = np.vstack(data_list)
    
    elif task == 2:
        length = len(adj_list)
        len_line = []
        line_ratio = []
        
        for c_num in range(length):
            c_list = adj_list[c_num]
            uni_num = np.unique(c_list[:, 0])
            cell_len_line = []
            cell_line_ratio = []
            
            for item in uni_num:
                loc_temp = c_list[:, 0] == item
                a = c_list[loc_temp, 0]
                b = c_list[loc_temp, 1]
                len_item_line = []
                
                for i in range(len(a)):
                    len_item_line_temp = np.sqrt(np.sum((fea_list[a[i], :] - fea_list[b[i], :]) ** 2))
                    len_item_line.append(len_item_line_temp)
                
                cell_len_line.extend(len_item_line)
                cell_line_ratio.extend(np.array(len_item_line) / np.min(len_item_line))
            
            len_line.append(cell_len_line)
            line_ratio.append(cell_line_ratio)
        
        data_list = []
    
    else:
        print('Wrong task label')
        return None, None
    
    return data_list, line_ratio

# # 示例调用
# adj_list = np.array([[1, 2], [2, 3], [3, 1]])
# fea_list = np.array([[0, 0, 0, 1, 1], [1, 1, 1, 2, 2], [2, 2, 2, 3, 3]])
# task = 1

# data_list, line_ratio = cal_length_ratio_list_hd(adj_list, task, fea_list)
# print("Data List:", data_list)
# print("Line Ratio:", line_ratio)