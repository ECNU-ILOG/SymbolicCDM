import logging
import ray
import numpy as np
from deap import base, creator, tools, algorithms
from tqdm import tqdm

from .eval import accuracy, root_mean_squared_error, area_under_curve, f1_score, degree_of_agreement
from .utility import StudentDataSet, parallelCompute, cxSimulatedBinary, mutGaussian, initInteractionFunction, exam
from .operators import sigmoid, tanh


class Argument:
    def __init__(self, r_matrix: np.ndarray,
                 q_matrix: np.ndarray,
                 knowledge_number: int,
                 question_number: int,
                 student_id: int):
        self.r_matrix = r_matrix
        self.q_matrix = q_matrix
        self.knowledge_number = knowledge_number
        self.question_number = question_number
        self.interaction_function = initInteractionFunction
        self.interaction_function_string = "default interaction function"
        self.student_id = student_id

        # ga toolbox configuration
        self.toolbox = base.Toolbox()
        self.toolboxInit()

        # other settings
        self.population = None
        self.hof = None

    def toolboxInit(self):
        self.toolbox.register("arguments", self.generateIndividual)
        self.toolbox.register("individual", tools.initIterate, creator.individual_argument, self.toolbox.arguments)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", cxSimulatedBinary, eta=10.0)
        self.toolbox.register("mutate", mutGaussian, mu=0.0, sigma=1.0, )

    def generateIndividual(self):
        proficiency_level = sigmoid(np.random.normal(0.0, 1.0, self.knowledge_number), offset=0.0)
        discrimination = sigmoid(np.random.normal(0.0, 1.0, (self.question_number, 1)), offset=0.0)

        return [proficiency_level, discrimination]

    def evaluate(self, individual, threshold=0.5):
        fitness = 0.0
        proficiency_level, discrimination = individual
        for idx in range(self.question_number):
            fitness += 2 * self.r_matrix[idx] * (tanh(self.interaction_function(self.q_matrix[idx],
                                                                                proficiency_level,
                                                                                discrimination[idx])) - threshold)

        return fitness,

    def train(self, init=False):
        population_size = 100
        ngen = 50
        cxpb = 0.5
        mutpb = 0.5
        threshold = 0.5
        scale = 0.1
        if init:
            self.population = self.toolbox.population(n=population_size)
            self.hof = tools.HallOfFame(maxsize=1,
                                        similar=lambda x, y:
                                        (x[0] == y[0]).all() and
                                        (x[1] == y[1]).all())
        else:
            # to re-weight the population after switch
            for individual in self.population:
                individual.fitness.delValues()
        # monotonicity assumption
        for idx in range(self.question_number):
            if self.r_matrix[idx] == 0:
                continue
            for individual in self.population:
                proficiency_level, discrimination = individual
                prediction = tanh(self.interaction_function(self.q_matrix[idx],
                                                            proficiency_level,
                                                            discrimination[idx]))
                if prediction <= threshold and self.r_matrix[idx] == 1:
                    increment = np.random.uniform(0, scale, (1, self.knowledge_number)) * self.q_matrix[
                        idx]
                    individual[0] = proficiency_level + increment
                elif prediction > threshold and self.r_matrix[idx] == -1:
                    decrement = np.random.uniform(0, scale, (1, self.knowledge_number)) * self.q_matrix[
                        idx]
                    individual[0] = proficiency_level - decrement
        self.population, _ = algorithms.eaSimple(self.population, self.toolbox,
                                                 cxpb, mutpb, ngen,
                                                 halloffame=self.hof, verbose=False)

    def unpack(self):
        proficiency_level = self.hof.items[0][0].copy()
        discrimination = self.hof.items[0][1].copy()
        mask = self.r_matrix.astype(bool)
        prediction = []
        for idx in range(self.question_number):
            if not mask[idx]:
                continue
            prediction.append(tanh(self.interaction_function(self.q_matrix[idx],
                                                             proficiency_level,
                                                             discrimination[idx])))
        prediction = np.array(prediction)
        prediction = np.where(prediction >= 0.5, 1, -1)
        acc = np.mean(prediction == self.r_matrix[mask])
        return proficiency_level, discrimination, acc

    def update(self, interaction_func, interaction_func_str):
        self.interaction_function = interaction_func
        self.interaction_function_string = interaction_func_str


class GeneticArgumentSearch:
    def __init__(self, q_matrix: np.ndarray,
                 student_number: int,
                 question_number: int,
                 knowledge_number: int,
                 train_set: StudentDataSet, ):
        self.q_matrix = q_matrix
        self.student_number = student_number
        self.question_number = question_number
        self.knowledge_number = knowledge_number
        self.train_set = train_set
        self.interaction_function = initInteractionFunction
        self.interaction_function_string = "default interaction function"

        # initialize lists containing arguments from each student
        self.arguments = []
        self.weights = np.full(self.student_number, 1 / self.student_number)
        self.initArguments()

    def initArguments(self):
        r_matrix = np.zeros((self.student_number, self.question_number))
        for line in self.train_set:
            r_matrix[line[0] - 1][line[1] - 1] = line[2]
            if line[2] == 0:
                r_matrix[line[0] - 1][line[1] - 1] = -1

        for i in range(self.student_number):
            self.arguments.append(Argument(r_matrix[i], self.q_matrix,
                                           self.knowledge_number, self.question_number, i))

    def train(self, init=False):
        print("Genetic algorithm-based search")
        print("Start searching...")
        # parallel computing
        # ray.init(logging_level=logging.WARNING)
        # results = []
        # remaining_ids = []
        # with tqdm(total=self.student_number) as pbar:
        #     for argument in self.arguments:
        #         result_id = parallelCompute.remote(argument, (init,))
        #         remaining_ids.append(result_id)
        #
        #     while remaining_ids:
        #         done_ids, remaining_ids = ray.wait(remaining_ids)
        #         for result_id in done_ids:
        #             results.append(ray.get(result_id))
        #             pbar.update()
        #
        # self.arguments = sorted(results, key=lambda x: x.student_id)
        # ray.shutdown()
        with tqdm(total=self.student_number) as pbar:
            for argument in self.arguments:
                argument.train(init)
                pbar.update()


    def evaluate(self,
                 student_data: StudentDataSet,
                 q_matrix: np.ndarray,
                 proficiency_level: np.ndarray,
                 discrimination: np.ndarray):
        prediction, truth = exam(student_data, q_matrix,
                                 proficiency_level,
                                 discrimination,
                                 self.interaction_function)

        acc = accuracy(prediction, truth)
        rmse = root_mean_squared_error(prediction, truth)
        auc = area_under_curve(prediction, truth)
        f1 = f1_score(prediction, truth)
        doa = degree_of_agreement(self.q_matrix, proficiency_level, student_data)

        return acc, rmse, auc, f1, doa

    def update(self, interaction_func, interaction_func_str):
        self.interaction_function = interaction_func
        self.interaction_function_string = interaction_func_str
        for argument in self.arguments:
            argument.update(interaction_func, interaction_func_str)

    def unpack(self):
        pl, dis, acc = self.arguments[0].unpack()
        proficiency_levels = [pl]
        discriminations = [dis]
        accs = [acc]
        for argument in self.arguments[1:]:
            pl, dis, acc = argument.unpack()
            proficiency_levels.append(pl)
            discriminations.append(dis)
            accs.append(acc)

        proficiency_level = np.vstack(proficiency_levels)
        acc = np.array(accs)
        discriminations = np.array(discriminations)
        # normalize
        acc = acc / np.sum(acc)
        discrimination = discriminations[0] * acc[0]
        # weight
        for idx in range(1, len(acc)):
            discrimination += discriminations[idx] * acc[idx]

        return self.q_matrix, proficiency_level, discrimination
