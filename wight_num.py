import numpy as np

def wightNum(Num, order):
    if order == 1:
        a = 0.0005
    elif order == 2:
        a = 0.00005
    else:
        print('wrong order')
        return None

    c = 0
    y = 2 / (1 + np.exp(-a * (Num - c))) - 1
    wNum = 1 - y
    return wNum