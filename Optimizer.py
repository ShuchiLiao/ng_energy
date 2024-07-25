import numpy as np
from deap import base, creator, tools, algorithms
from constants import ROOM_SET_TEMP_RANGE, SFWDSDZ_RANGE, COMFORTABLE_ROOM_TEMP, OB3_SCALE_FACTOR
from utils import setup_logger


class GenericOptimizer:
    def __init__(self, residual, room_numbers, model,
                 population_size=50, cxpb=0.7, mupb=0.2, generations=50):
        self.residual = residual
        self.room_numbers = room_numbers  # office numbers + 1 setting value for SFWDSDZ
        self.model = model
        self.population_size = population_size
        self.cxpb = cxpb
        self.mupb = mupb
        self.generations = generations

        self.toolbox = base.Toolbox()
        self.rt_range = ROOM_SET_TEMP_RANGE
        self.wt_range = SFWDSDZ_RANGE
        self.log = setup_logger('optimization log', 'logs/optimize.log')
        self.setup()

    def setup(self):
        #定义一个适应度类FitnessMulti，其继承自base.Fitness，并指定权重weights.-1:最小化； 1：最大化
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0))  # weights for 3 objective functions
        #定义个体类Individual，继承自Python的内置list类，并包含了前面定义的适应度函数FitnessMulti。
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        #注册一个函数attr_room，用于生成一个在self.rt_range范围内的随机浮点数。
        self.toolbox.register("attr_room", np.random.uniform, self.rt_range[0], self.rt_range[1])
        #注册一个函数attr_wind，用于生成一个在self.wt_range范围内的随机浮点数。
        self.toolbox.register("attr_wind", np.random.uniform, self.wt_range[0], self.wt_range[1])
        #注册一个生成个体的函数individual，它调用tools.initCycle来初始化一个creator.Individual实例。
        # 这个实例包含room_num个attr_room和一个attr_wind生成的随机数
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                              (self.toolbox.attr_room,) * self.room_numbers + (self.toolbox.attr_wind,), n=1)
        #注册一个生成种群的函数population，它调用tools.initRepeat来生成一个包含多个个体的列表
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        #注册交叉操作mate，使用的是tools.cxBlend交叉操作，交叉率为0.5。
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        #注册变异操作mutate，使用的是tools.mutGaussian变异操作，均值为0，标准差为1，每个基因位的变异概率为0.2。
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        #注册选择操作select，使用的是NSGA-II选择算法tools.selNSGA2。
        self.toolbox.register("select", tools.selNSGA2)
        #注册评估函数evaluate，它调用self.objective_fun来评估个体的适应度。
        self.toolbox.register("evaluate", self.objective_fun)

    def objective_fun(self, individual):
        # individual 个体，TEMPSET的温度范围限定，SFWDSDZ的温度限定
        individual[:-1] = np.clip(individual[:-1], self.rt_range[0], self.rt_range[1])
        individual[-1] = np.clip(individual[-1], self.wt_range[0], self.wt_range[1])

        #将residual和个体组合成输入，进行模型预测， residual和individual都是1d list
        # list1+ list2 进行合并，然后转化为2d list-》2的array [sequence_len, feature_num],输入格式符合，
        # 可以直接调用model的预测函数
        X = np.array([self.residual + individual])

        # prediction格式[predictsteps, out_feature_num],即[1, room_num+1]
        pred = self.model.ga_predict(X)

        pt_rt_last_moment = X.reshape(-1)[:int(self.room_numbers + 1)]  # 展开为1d array，只取Pt和Roomtemp值
        pt_rt_this_moment_pred = pred.reshape(-1)  # 展开为1d array

        objective1 = (np.mean(np.abs(pt_rt_this_moment_pred[1:] - COMFORTABLE_ROOM_TEMP)) /  # 上一时刻和舒适温度的差距
                      np.mean(np.abs(pt_rt_last_moment[1:] - COMFORTABLE_ROOM_TEMP)))  # 未来时刻和舒适温度的差距
        # 当未来温度与舒适温度的差距越小， objective 1越小。即相对于上一时刻，这一时刻越靠近舒适度。

        objective2 = pt_rt_this_moment_pred[0] / pt_rt_last_moment[0]  # 总功率，越小耗电月底
        # 这一时刻功率越小越好（相较于上一时刻用电量），

        # log 1p: return log (1+x)
        temp_change = np.mean(np.abs(pt_rt_this_moment_pred[1:] - pt_rt_last_moment[1:]))
        efficiency = temp_change / pt_rt_this_moment_pred[0] if pt_rt_this_moment_pred[0] != 0 else 0
        objective3 = np.log1p(efficiency) * OB3_SCALE_FACTOR  # 单位能耗温度变化，越大则暗示能效越高

        return objective1, objective2, objective3

    def optimize_and_advise(self, data_columns):
        self.clear_session()  # 开始优化前清理会话
        # 生成种群
        population = self.toolbox.population(n=self.population_size)
        #创建一个Statistics对象，用于收集统计信息。lambda ind: ind.fitness.values是一个函数，用于提取个体的适应度值
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        #一个统计函数min，用于计算适应度值的最小值，并在每一代记录。axis=0表示对每个目标函数分别计算最小值。
        stats.register("min", np.min, axis=0)

        # 记录初始参数
        self.log.info(f"Initial parameters: Population size={self.population_size}, Crossover probability={self.cxpb}, "
                      f"Mutation probability={self.mupb}, Generations={self.generations}")
        self.log.info(f"Optimizing using {self.model.model_name} in process...")

        # 调用algorithms.eaMuPlusLambda函数运行遗传算法。参数说明如下：
        # population：初始种群。
        # self.toolbox：包含所有注册的操作和评估函数的工具箱。
        # mu：种群规模。
        # lambda_：每代生成的新个体数，为种群规模的两倍。
        # cxpb：交叉概率。
        # mupb：变异概率。
        # ngen：进化的代数。
        # stats：统计信息的收集对象。
        # halloffame：名人堂对象，用于保存最佳个体（这里设置为None）。
        # verbose：控制输出信息的详细程度（设置为0表示不输出）。
        population, logbook = algorithms.eaMuPlusLambda(population, self.toolbox, mu=self.population_size,
                                                        lambda_=2 * self.population_size,
                                                        cxpb=self.cxpb, mutpb=self.mupb, ngen=self.generations,
                                                        stats=stats, halloffame=None, verbose=1)
        #调用tools.sortNondominated函数对种群进行非支配排序，获取Pareto前沿解。len(population)表示所有个体的数量，
        # first_front_only=True表示只返回第一前沿的个体（最优解）
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # Find the solution with the minimum value for objective 1 (temperature stability)
        self.log.info(f"Done. ")
        optimal_solution_temp_stability = min(pareto_front, key=lambda ind: ind.fitness.values[0])
        result1 = self.recommendation(optimal_solution_temp_stability, data_columns)

        # Find the solution with the minimum value for objective 2 (energy consumption)
        optimal_solution_energy_consumption = min(pareto_front, key=lambda ind: ind.fitness.values[1])
        result2 = self.recommendation(optimal_solution_energy_consumption, data_columns)

        # Find the solution with the maximum value for objective 3 (energy efficiency)
        optimal_solution_energy_efficiency = max(pareto_front, key=lambda ind: ind.fitness.values[2])
        result3 = self.recommendation(optimal_solution_energy_efficiency, data_columns)

        results = {
            '舒适模式': result1,
            '节能模式': result2,
            '降耗模式': result3
        }

        return results

    def recommendation(self, solution, data_columns):
        X = np.array([self.residual + solution])
        # prediction格式[predictsteps, out_feature_num],即[1, room_num+1]
        pred = self.model.ga_predict(X)
        prediction = pred.reshape(-1).tolist()  # 展开成1dlist

        # 列名
        setup_columns = data_columns[-int(self.room_numbers + 1):]  # 'TEMPSET',"SFWDSDZ" 列名
        pt_room_columns = data_columns[:int(self.room_numbers + 1)]  # 'Pt' 和'TEMPROOM’ 列名

        setup = {col: sol for col, sol in zip(setup_columns, solution)}
        predict = {col: p for col, p in zip(setup_columns, prediction)}

        result = {'空调设定温度': {k: setup[k] for i, k in enumerate(setup) if i < len(setup) - 1},
                  '送风设定温度': {k: setup[k] for i, k in enumerate(setup) if i == len(setup) - 1},
                  '房间预测温度': {k: setup[k] for i, k in enumerate(predict) if i > 0},
                  '总功率预测': {k: setup[k] for i, k in enumerate(predict) if i == 0}
                  }
        return result

    def clear_session(self):
        # 清理会话，重新初始化种群和统计信息
        self.toolbox.unregister("population")
        self.toolbox.unregister("individual")
        self.toolbox.unregister("attr_room")
        self.toolbox.unregister("attr_wind")
        self.toolbox.unregister("mate")
        self.toolbox.unregister("mutate")
        self.toolbox.unregister("select")
        self.toolbox.unregister("evaluate")
        self.setup()
        self.log.info("Session cleared and re-initialized.")
