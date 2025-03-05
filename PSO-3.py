import random
from collections import namedtuple
import xlrd
import numpy as np
import matplotlib.pyplot as plt

# 定义航班和停机位结构体
Flight = namedtuple('Flight', ['hangban', 'daoda', 'likai', 'yizhan', 'gateType'])
Gate = namedtuple('Gate', ['type', 'number', 'width'])

# 随机生成一个符合约束条件的个体
def create_one():
    X = np.zeros([dim, gateNum])  # 初始化分配矩阵
    for i in range(dim):  # 遍历每个航班
        while 1 not in X[i]:  # 确保航班分配到一个停机位
            rand = random.randint(0, gateNum - 1)  # 随机选择一个停机位
            # 检查约束条件2：飞机翼展 <= 停机位宽度
            if Flight_list[i].yizhan <= Gate_list[rand].width:
                conflict = False
                for j in range(i - 1, -1, -1):
                    if X[j][rand] == 1:  # 检查同一停机位的航班
                                # 检查时间间隔是否满足 15 分钟
                        if Flight_list[j].likai + gama > Flight_list[i].daoda:
                            conflict = True
                            break
                if not conflict:
                        # 检查约束条件4：相邻停机位的航班进出错时 5 分钟
                    if rand != 0:  # 检查前一个停机位
                        if Gate_list[rand].number - Gate_list[rand - 1].number <= 2:  # 检查停机位是否相邻
                            for j in range(i - 1, -1, -1):
                                if X[j][rand - 1] == 1:
                                        # 检查时间间隔是否满足 5 分钟
                                    if Flight_list[j].likai + beta > Flight_list[i].daoda:
                                        conflict = True
                                        break
                    if rand != gateNum - 1:  # 检查后一个停机位
                        if Gate_list[rand + 1].number - Gate_list[rand].number <= 10:  # 检查停机位是否相邻
                            for j in range(i - 1, -1, -1):
                                if X[j][rand + 1] == 1:
                                        # 检查时间间隔是否满足 5 分钟
                                    if Flight_list[j].likai + beta > Flight_list[i].daoda:
                                        conflict = True
                                        break
                    if not conflict:
                        X[i][rand] = 1  # 分配停机位
    return X

# 检查解是否满足约束条件
def check_constraints(X):
    for i in range(dim):
        for j in range(gateNum):
            if X[i][j] == 1:
                # 检查约束条件2：飞机翼展 <= 停机位宽度
                if Flight_list[i].yizhan > Gate_list[j].width:
                    return False
                # 检查时间冲突（约束条件5-8）
                for k in range(i - 1, -1, -1):
                    if X[k][j] == 1:
                        # 检查时间间隔是否满足 15 分钟
                        if Flight_list[k].likai + gama > Flight_list[i].daoda:
                            return False
                # 检查约束条件4：相邻停机位的航班进出错时 5 分钟
                if j != 0:  # 检查前一个停机位
                    if Gate_list[j].number - Gate_list[j - 1].number <= 10:  # 检查停机位是否相邻
                        for k in range(i - 1, -1, -1):
                            if X[k][j - 1] == 1:
                                if Flight_list[k].likai + beta > Flight_list[i].daoda:
                                    return False
                if j != gateNum - 1:  # 检查后一个停机位
                    if Gate_list[j + 1].number - Gate_list[j].number <= 10:  # 检查停机位是否相邻
                        for k in range(i - 1, -1, -1):
                            if X[k][j + 1] == 1:
                                if Flight_list[k].likai + beta > Flight_list[i].daoda:
                                    return False
    return True

# 修复不满足约束条件的解
def repair_solution(X):
    for i in range(dim):
        for j in range(gateNum):
            if X[i][j] == 1:
                # 修复约束条件2
                if Flight_list[i].yizhan > Gate_list[j].width:
                    X[i][j] = 0  # 取消分配
                # 修复时间冲突
                for k in range(i - 1, -1, -1):
                    if X[k][j] == 1:
                        # 检查时间间隔是否满足 15 分钟
                        if Flight_list[k].likai + gama > Flight_list[i].daoda:
                            X[i][j] = 0  # 取消分配
                            # 修复约束条件4：相邻停机位的航班进出错时 5 分钟
                if j != 0:  # 检查前一个停机位
                    if Gate_list[j].number - Gate_list[j - 1].number <= 10:  # 检查停机位是否相邻
                        for k in range(i - 1, -1, -1):
                            if X[k][j - 1] == 1:
                                if Flight_list[k].likai + beta > Flight_list[i].daoda:
                                    X[i][j] = 0  # 取消分配
                if j != gateNum - 1:  # 检查后一个停机位
                    if Gate_list[j + 1].number - Gate_list[j].number <= 10:  # 检查停机位是否相邻
                        for k in range(i - 1, -1, -1):
                            if X[k][j + 1] == 1:
                                if Flight_list[k].likai + beta > Flight_list[i].daoda:
                                    X[i][j] = 0  # 取消分配
    return X

# 适应度函数
def fitness(X):
    fitness_1 = 0  # 目标函数1，统计远机位数量
    fitness_2 = 0  # 目标函数2，统计近机位的空闲时间
    for i in range(dim):
        for j in range(gateNum):
            if X[i][j] == 1:
                if Gate_list[j].type == 1:  # 远机位
                    fitness_1 += 1
                else:  # 近机位
                    # 计算空闲时间
                    for k in range(i - 1, -1, -1):
                        if X[k][j] == 1:
                            fitness_2 += max(0, Flight_list[i].daoda - Flight_list[k].likai)
    return omiga1 * fitness_1 + omiga2 * fitness_2
# 粒子群位置更新
def PSO(X, pbest, gbest):
    omiga = omiga_max - (omiga_max - omiga_min) * (t / max_iter)  # 惯性权重
    newX = np.zeros([dim, gateNum], dtype="int64")  # 初始化新解
    for i in range(dim):
        r1 = np.random.uniform(omiga_min, omiga_max)
        if r1 < omiga:
            r2 = np.random.random()
            if r2 < c1:
                newX[i] = pbest[i]  # 学习个体最优
            elif r2 < c2:
                newX[i] = gbest[i]  # 学习全局最优
            else:
                newX[i] = X[i]
        else:
            newX[i] = X[i]

    # 检查新解是否满足约束条件
    if not check_constraints(newX):
        newX = repair_solution(newX)  # 修复解
    return newX

# 主程序
if __name__ == '__main__':
    # 设置参数
    max_iter = 10  # 最大迭代次数
    pop_size = 50  # 种群大小
    alpha = 0.0312499975  # 约束条件3，航班占用停机位的最小时间，45min
    beta = 0.0069444416666667  # 相邻停机位的航班进出时间间隔，5min
    gama = 0.010416665  # 同一停机位相邻航班最小安全时间间隔，15min
    omiga1 = 0.6  # 适应度函数1权重
    omiga2 = 0.4  # 适应度函数2权重
    omiga_max = 0.9  # 最大惯性权重
    omiga_min = 0.4  # 最小惯性权重
    c1 = 0.5  # 个体学习因子
    c2 = 0.5  # 群体学习因子
    dim = 337  # 个体维度，即航班个数
    gateNum = 228  # 停机位个数

    # 读取数据
    Flight_list = []  # 航班集合
    Gate_list = []  # 停机位集合
    workbook1 = xlrd.open_workbook('./data/flight_data.xls')  # 读取航班数据
    worksheet1 = workbook1.sheet_by_index(0)
    workbook2 = xlrd.open_workbook('./data/gate_data.xls')  # 读取停机位数据
    worksheet2 = workbook2.sheet_by_index(0)

    # 读取航班信息
    for i in range(dim):
        hangban = worksheet1.cell_value(i + 1, 0)
        daoda = worksheet1.cell_value(i + 1, 1)
        likai = worksheet1.cell_value(i + 1, 2)
        yizhan = worksheet1.cell_value(i + 1, 3)
        gateType = 0 if likai - daoda < 0.166667 else 1  # 约束条件9
        flight = Flight(hangban, daoda, likai, yizhan, gateType)
        Flight_list.append(flight)

    # 读取停机位信息
    for i in range(gateNum):
        type = worksheet2.cell_value(i + 1, 0)
        number = worksheet2.cell_value(i + 1, 1)
        width = worksheet2.cell_value(i + 1, 2)
        gate = Gate(type, number, width)
        Gate_list.append(gate)

    # 初始化
    curve = np.zeros(max_iter)  # 迭代曲线
    remainTimeCurve = np.zeros(max_iter)  # 声明空闲时间的迭代曲线空间,声明一个数组 remainTimeCurve，用于存储在每次迭代中记录的空闲时间，数组大小同样为 max_iter
    popX = np.zeros([pop_size, dim, gateNum], dtype="int64")  # 种群空间
    Pbest = np.zeros([pop_size, dim, gateNum], dtype="int64")  # 个体最优
    Gbest = np.zeros([dim, gateNum], dtype="int64")  # 全局最优
    popF = np.zeros(pop_size)  # 适应度空间

    for i in range(pop_size):
        popX[i] = create_one()  # 生成初始解
        Pbest[i] = popX[i]  # 初始化个体最优
        popF[i] = fitness(popX[i])  # 计算适应度

    best_index = np.argmin(popF)  # 找到全局最优
    Gbest = popX[best_index]

    # 主循环
    for t in range(max_iter):
        for i in range(pop_size):
            popX[i] = PSO(popX[i], Pbest[i], Gbest)  # 更新位置
            popF[i] = fitness(popX[i])  # 计算适应度
            if popF[i] < fitness(Pbest[i]):  # 更新个体最优
                Pbest[i] = popX[i]
        best_index = np.argmin(popF)  # 更新全局最优
        Gbest = popX[best_index]
        curve[t] = np.min(popF)  # 记录迭代结果
        print(f'第 {t} 次迭代，最优适应度值为：{np.min(popF)}')

        # 保存第t次迭代的最小空闲时间
        gateNumber = []  # 初始化一个空列表 gateNumber，用于存储在每个近机位停靠的航班数量
        flightNumber = []  # 初始化另一个空列表 flightNumber，用于记录每个近机位的航班数
        time_sum = []  # 初始化空列表 time_sum，用于存储每个近机位的空闲时间
        for lie in range(gateNum):  # 遍历所有停机位
            if Gate_list[lie].type == 0:  # 挑选近机位
                index = []  # 统计停在该机位的航班,初始化一个空列表 index，用于记录当前近机位上停靠的航班的索引
                for col in range(dim - 1, -1, -1):  # 从后向前遍历所有航班（dim 是航班的总数），这样做的目的是为了方便计算航班之间的时间间隔
                    if Gbest[col][lie] == 1:  # 检查在全局最优解 Gbest 中，航班 col 是否停在当前的近机位 lie。
                        index.append(col)  # 如果是，则将该航班的索引添加到 index 列表中
                if len(index) != 0:  # 判断 index 列表是否为空，只有在有航班停靠的情况下才进行记录
                    gateNumber.append(int(Gate_list[lie].number))  # 将当前近机位的编号（转换为整数）添加到 gateNumber 列表中
                    flightNumber.append(
                        len(index))  # 记录本次迭代的停在各机位的航班数量,记录在当前近机位上停靠的航班数量，即 index 列表的长度，并将其添加到 flightNumber 列表中
                time = 0  # 初始化一个变量 time，用于计算当前近机位的空闲时间
                for i in range(1, len(index), 1):  # 从 index 列表的第二个元素开始遍历，以计算各航班之间的空闲时间
                    time = time + (Flight_list[index[i - 1]].daoda - Flight_list[
                        index[i]].likai)  # 对于相邻的航班，计算其到达时间和离开时间之间的时间差，并将其累加到 time 变量中。这表示航班之间的空闲时间
                time_sum.append(time)  # 将计算得到的空闲时间添加到 time_sum 列表中,
        remainTimeCurve[t] = sum(time_sum)  # 记录本次迭代近机位的空闲时间,计算 time_sum 列表中所有空闲时间的总和，并将结果记录在 remainTimeCurve 中，t 是当前迭代的索引
        print('第', t, '次迭代，近机位的空闲时间为', remainTimeCurve[t], '最优适应度值为', curve[t])
        # 以上统计近机位数据

        gateNumber1 = []  # 初始化一个空列表 gateNumber1，用于存储当前迭代中近机位的编号
        flightNumber1 = []  # 初始化另一个空列表 flightNumber1，用于记录在近机位上停靠的航班数量
        for lie in range(gateNum):  # 遍历所有停机位
            if Gate_list[lie].type == 1:  # 挑选远机位
                index = []  # 统计停在该机位的航班,初始化一个空列表 index，用于记录当前近机位上停靠的航班的索引
                for col in range(dim - 1, -1, -1):  # 从后向前遍历所有航班（dim 是航班的总数），目的是便于计算航班停靠信息
                    if Gbest[col][lie] == 1:
                        index.append(col)  # 检查在全局最优解 Gbest 中，航班 col 是否停在当前的远机位 lie。如果是，则将该航班的索引添加到 index 列表中
                if len(index) != 0:  # 判断 index 列表是否为空，只有在有航班停靠的情况下才进行记录
                    gateNumber1.append(int(Gate_list[lie].number))  # 将当前近机位的编号（转换为整数）添加到 gateNumber1 列表中
                    flightNumber1.append(
                        len(index))  # 记录本次迭代的停在各机位的航班数量,记录在当前远机位上停靠的航班数量，即 index 列表的长度，并将其添加到 flightNumber1 列表中
    # endregion

    # 输出结果
    index_list = np.where(Gbest == 1)[1]
    print('-----------------最终得到的最优航班调度方案-----------------')
    for i in range(dim):
        print(f'航班 {int(Flight_list[i].hangban)} 的停机位：{Gate_list[index_list[i]].number}')

    # 绘制适应度迭代图
    plt.figure()
    plt.plot(curve, color='green', label='PSO')
    plt.title('总适应度值迭代图')
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值')
    plt.legend()
    plt.show()
