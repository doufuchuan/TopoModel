import numpy as np
import matplotlib.pyplot as plt

def findTreshold(sortedList, th, orderLabel):
    Q = np.sort(sortedList)
    N = len(Q)
    t_exp = th
    G = []
    temp_D2_Q = 0
    temp_D1_Q = 0
    D1_Q = []
    D2_Q = []
    D = []

    for i in range(N):
        # 一阶导数
        if 1 < i < N - 1:
            temp_D1_Q = (Q[i + 1] - Q[i - 1]) / 2
        D1_Q.append(temp_D1_Q)

        # 二阶导数
        if 2 < i < N - 2:
            temp_D2_Q = abs((Q[i + 1] + Q[i - 2] - Q[i - 1] - Q[i])) / 4
        D2_Q.append(temp_D2_Q)

    if orderLabel == 1:
        D = Q * np.array(D1_Q)
    elif orderLabel == 2:
        D = Q * Q * np.array(D2_Q)

    Qm = []
    for i in range(N):
        m = np.mean(Q[:i + 1])
        Qm.append(D[i] / m)

    meanQm = np.mean(Qm)

    # 绘制图形
    plt.figure()
    plt.plot(Q, linewidth=1)
    plt.plot(Qm, '-r', linewidth=1)
    plt.title('Q & Qm')
    plt.legend(['Q', 'Qm'])
    plt.grid(True)

    # 寻找 T
    indx = np.where(np.array(Qm) >= t_exp)[0]
    if len(indx) == 0:
        indx = N - 1
    else:
        indx = indx[0]

    if indx == N - 1:
        T = Q[indx]
    else:
        T = 0.5 * (Q[indx] + Q[indx + 1])

    if T is None:
        T = Q[-1]

    # 在 Q 图中绘制 T
    xt = [np.where(Q >= T)[0][0], np.where(Q >= T)[0][0]]
    yt = [0, T]
    plt.plot(xt, yt, '--m', linewidth=2)
    plt.plot(xt, yt, '.k', markersize=20)

    xt = [0, np.where(Q >= T)[0][0]]
    yt = [T, T]
    plt.plot(xt, yt, '--m', linewidth=1.5)
    plt.plot(xt, yt, '.k', markersize=20)

    xt = [0, np.where(Q >= T)[0][0]]
    yt = [t_exp, t_exp]
    plt.plot(xt, yt, '--g', linewidth=1.5)
    plt.plot(xt, yt, '.k', markersize=20)

    plt.show()

    return T