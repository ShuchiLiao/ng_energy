import numpy as np
from deap import base, creator, tools, algorithms

class GenericOptimizer:
    def __init__(self, request, room_numbers, model,
                 roomsettemp_range=(20, 26), windtemp_range=(10, 14),
                 population_size=50, cxpb=0.7, mutpb=0.2, generations=50):
        request_len = int(len(request))
        length = request_len - room_numbers - 1
        self.request = request[:length]
        self.room_numbers = room_numbers  # office numbers + 1 setting value for SFWDSDZ
        self.model = model
        self.population_size = population_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.generations = generations

        self.toolbox = base.Toolbox()
        self.rt_range = roomsettemp_range
        self.wt_range = windtemp_range
        self.setup()

    def setup(self):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0))  # weights for 3 objective functions
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox.register("attr_float", np.random.uniform, self.rt_range[0], self.rt_range[1])
        self.toolbox.register("attr_wind", np.random.uniform, self.wt_range[0], self.wt_range[1])
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                              (self.toolbox.attr_float,) * self.room_numbers + (self.toolbox.attr_wind,), n=1)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("evaluate", self.objective_fun)

    def objective_fun(self, individual):
        individual[:-1] = np.clip(individual[:-1], self.rt_range[0], self.rt_range[1])
        individual[-1] = np.clip(individual[-1], self.wt_range[0], self.wt_range[1])

        X = np.array([self.request + individual])
        X_scaled = self.model.scaler_X.transform(X)
        predictions = self.model.predict(X_scaled)

        # not used, cause predictions already inverse transformed in model.predict()
        # pred_scaled_data = self.scaler_y.inverse_transform(predictions.reshape(1, -1))

        pred_scaled = predictions.reshape(-1)

        objective1 = np.sum(np.abs(pred_scaled[:-1] - 21))  # assume 23 degree is a more comfortable temp
        objective2 = pred_scaled[-1]
        objective3 = (np.sum(np.abs(pred_scaled[:-1] - np.array(individual[:-1]))) / pred_scaled[-1] * 1000) if \
            pred_scaled[-1] != 0 else 0

        return objective1, objective2, objective3

    def optimize_and_advise(self):
        population = self.toolbox.population(n=self.population_size)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min, axis=0)

        population, logbook = algorithms.eaMuPlusLambda(population, self.toolbox, mu=self.population_size,
                                                        lambda_=2 * self.population_size,
                                                        cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.generations,
                                                        stats=stats, halloffame=None, verbose=0)

        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # Find the solution with the minimum value for objective 1 (temperature stability)

        optimal_solution_temp_stability = min(pareto_front, key=lambda ind: ind.fitness.values[0])
        result1 = self.recommendation(optimal_solution_temp_stability)

        # Find the solution with the minimum value for objective 2 (energy consumption)
        optimal_solution_energy_consumption = min(pareto_front, key=lambda ind: ind.fitness.values[1])
        result2 = self.recommendation(optimal_solution_energy_consumption)

        # Find the solution with the maximum value for objective 3 (energy efficiency)
        optimal_solution_energy_efficiency = max(pareto_front, key=lambda ind: ind.fitness.values[2])
        result3 = self.recommendation(optimal_solution_energy_efficiency)

        results = {
            '舒适模式': result1,
            '节能模式': result2,
            '降耗模式': result3
        }
        print(results)
        return results

    def recommendation(self, solution):
        X = np.array([self.request + solution])
        X_scaled = self.model.scaler_X.transform(X)
        predictions = self.model.predict(X_scaled)[0]

        # pred_scaled_data = self.scaler_y.inverse_transform(predictions.reshape(1, -1))

        pred_scaled = predictions.reshape(-1).tolist()

        result = {'空调设定温度': solution[:-1],
                  '送风设定温度': solution[-1],
                  '房间预测温度': pred_scaled[:-1],
                  '总功率预测': pred_scaled[-1]
                  }
        return result