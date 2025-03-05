import numpy as np

def findAdjMetrixHD(TriList, param):
    dim = param['Dm']
    twoColumList = []

    # 生成二维列表
    for i in range(dim - 1):
        for j in range(i + 1, dim):
            twoColumList.extend(TriList[:, [i, j]].tolist())
    twoColumList = np.array(twoColumList)

    # 获取唯一的行
    uniList = np.unique(twoColumList, axis=0)
    len_uniList = len(uniList)

    # 移除重复的反向对
    for i in range(len_uniList):
        listValue = uniList[i]
        if listValue[0] != 0:
            flipped_value = np.flip(listValue)
            iList = np.where((uniList == flipped_value).all(axis=1))[0]
            if len(iList) > 0:
                uniList[iList] = [0, 0]

    # 移除值为 0 的行
    adjListTemp = uniList[~(uniList == 0).all(axis=1)]
    adjList = np.vstack([adjListTemp[:len(adjListTemp) // 2], adjListTemp[len(adjListTemp) // 2:]]).T

    print('hh')
    return adjList