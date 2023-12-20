import operator
import numpy as np
from deap import base, creator, tools, gp, algorithms

from .eval import accuracy, root_mean_squared_error, area_under_curve, f1_score, degree_of_agreement, loss
from .operators import dot, tanh, sigArctan, sigmoid
from .utility import StudentDataSet, exam, mutUniformWithPruning


class InteractionFunc:
    def __init__(self, q_matrix: np.ndarray,
                 train_set: StudentDataSet):
        self.q_matrix = q_matrix
        self.train_set = train_set
        self.proficiency_level = None
        self.discrimination = None

        # our interation function is:
        # prob = f(qMatrix, proficiencyLevel, discrimination)
        self.input_type = [np.ndarray, np.ndarray, np.float64]
        self.output_type = np.float64

        # construction set
        self.primitiveSet = gp.PrimitiveSetTyped("main", self.input_type, self.output_type)
        self.primitiveSetInit()

        # gp toolbox configuration
        self.toolbox = base.Toolbox()
        self.toolboxInit()

        # gp multi statistics configuration
        self.multiStatistics = tools.MultiStatistics(accuracy=tools.Statistics(lambda ind: ind.fitness.values[1]),
                                                     RMSE=tools.Statistics(lambda ind: ind.fitness.values[0]), )

        self.multiStatistics.register("min", np.min)
        self.multiStatistics.register("max", np.max)

        # weights for boosting
        self.weights = np.full(len(self.train_set), 1 / len(self.train_set))

        # other settings
        self.population = None
        self.hof = None

    def primitiveSetInit(self):
        # considering monotonicity, we choose these operators
        self.primitiveSet.addPrimitive(operator.add, [np.ndarray, np.ndarray], np.ndarray)
        self.primitiveSet.addPrimitive(operator.add, [np.ndarray, np.float64], np.ndarray)
        self.primitiveSet.addPrimitive(operator.add, [np.float64, np.float64], np.float64)
        self.primitiveSet.addPrimitive(operator.mul, [np.ndarray, np.ndarray], np.ndarray)
        self.primitiveSet.addPrimitive(operator.mul, [np.float64, np.ndarray], np.ndarray)
        self.primitiveSet.addPrimitive(operator.mul, [np.float64, np.float64], np.float64)
        self.primitiveSet.addPrimitive(dot, [np.ndarray, np.ndarray], np.float64)
        self.primitiveSet.addPrimitive(tanh, [np.ndarray], np.ndarray)
        self.primitiveSet.addPrimitive(tanh, [np.float64], np.float64)
        self.primitiveSet.addPrimitive(sigmoid, [np.ndarray], np.ndarray)
        self.primitiveSet.addPrimitive(sigmoid, [np.float64], np.float64)
        self.primitiveSet.addPrimitive(np.square, [np.ndarray], np.ndarray)
        self.primitiveSet.addPrimitive(np.square, [np.float64], np.float64)
        self.primitiveSet.addPrimitive(sigArctan, [np.ndarray], np.ndarray)
        self.primitiveSet.addPrimitive(sigArctan, [np.float64], np.float64)
        # rename argument
        self.primitiveSet.renameArguments(ARG0='q_matrix', ARG1='proficiency_level', ARG2='discrimination')

    def toolboxInit(self):
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.primitiveSet, min_=4, max_=5)
        self.toolbox.register("individual", tools.initIterate, creator.individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.primitiveSet)

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", mutUniformWithPruning, pset=self.primitiveSet)

        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    def evaluate(self, individual, is_weight=True):
        currentInteractionFunc = self.toolbox.compile(expr=individual)
        prediction, truth = exam(self.train_set,
                                 self.q_matrix,
                                 self.proficiency_level,
                                 self.discrimination,
                                 currentInteractionFunc)
        # cpx = complexity(individual, len(self.input_type))
        if is_weight:
            acc = accuracy(prediction, truth, weights=self.weights)
            rmse = root_mean_squared_error(prediction, truth, weights=self.weights)
        else:
            acc = accuracy(prediction, truth)
            rmse = root_mean_squared_error(prediction, truth)

        return rmse, acc,

    def train(self, init=True):
        population_size = 50
        ngen = 10
        cxpb = 0.5
        mutpb = 0.1
        if init:
            self.population = self.toolbox.population(n=population_size)
            self.hof = tools.HallOfFame(maxsize=1)
        else:
            for individual in self.population:
                individual.fitness.delValues()
        self.population, _ = algorithms.eaSimple(self.population, self.toolbox,
                                                 cxpb, mutpb, ngen,
                                                 stats=self.multiStatistics,
                                                 halloffame=self.hof, verbose=False)

    def unpack(self, is_compiled=False):
        if self.hof:
            if is_compiled:
                return self.toolbox.compile(expr=self.hof.items[0])
            else:
                return self.hof.items[0]
        else:
            return None

    def update(self, q_matrix: np.ndarray,
               proficiency_level: np.ndarray,
               discrimination: np.ndarray, ):
        self.q_matrix = q_matrix
        self.proficiency_level = proficiency_level.copy()
        self.discrimination = discrimination.copy()

    def re_weight(self, estimator_weight: np.ndarray):
        self.weights = np.power(self.weights, estimator_weight)
        self.weights /= np.sum(self.weights)


class GeneticInteractionFunc:
    def __init__(self, q_matrix: np.ndarray,
                 boosting: int,
                 train_set: StudentDataSet, ):

        self.boosting = boosting
        self.beta = np.full(boosting, 1 / boosting)

        self.q_matrix = q_matrix
        self.proficiency_level = None
        self.discrimination = None

        self.train_set = train_set
        self.interaction = InteractionFunc(q_matrix, train_set)

        self.interaction_funcs = []
        self.interaction_funcs_string = []

    def __str__(self):
        if len(self.interaction_funcs) != 0:
            string = ""
            for i in range(len(self.interaction_funcs)):
                string += (str(self.beta[i]) + "*[" + self.interaction_funcs_string[i] + "]")
                if i != len(self.interaction_funcs) - 1:
                    string += '+'
            return string
        else:
            return "default"

    def evaluation(self, student_data: StudentDataSet) -> tuple:
        current_interaction_func = self.function()
        prediction, truth = exam(student_data,
                                 self.q_matrix,
                                 self.proficiency_level,
                                 self.discrimination,
                                 current_interaction_func)

        acc = accuracy(prediction, truth)
        rmse = root_mean_squared_error(prediction, truth)
        auc = area_under_curve(prediction, truth)
        f1 = f1_score(prediction, truth)
        doa = degree_of_agreement(self.q_matrix, self.proficiency_level, student_data)

        return acc, rmse, auc, f1, doa

    def train(self):
        print("Genetic programming search (boosting)")
        beta = np.zeros(self.boosting)
        interaction_funcs = []
        interaction_funcs_string = []
        # init weight in each epoch
        self.interaction.weights = np.full(len(self.train_set), 1 / len(self.train_set))
        for i in range(self.boosting):
            print("Boosting {}".format(i + 1))
            self.interaction.train()
            # measure the latest one
            prediction, truth = exam(self.train_set,
                                     self.q_matrix,
                                     self.proficiency_level,
                                     self.discrimination,
                                     self.interaction.unpack(is_compiled=True))
            losses = loss(prediction, truth)
            mean_loss = np.sum(losses * self.interaction.weights)
            estimator_weight = mean_loss / (1 - mean_loss)
            self.interaction.re_weight(estimator_weight)
            # add tree and weight
            interaction_funcs.append(self.interaction.unpack(is_compiled=True))
            interaction_funcs_string.append(str(self.interaction.unpack()))
            beta[i] = estimator_weight
            _, acc = self.interaction.evaluate(self.interaction.unpack(), is_weight=False)
            print("Accuracy in this boosting: {}".format(acc))
            print("Boosting part:", interaction_funcs_string[-1])
        # update
        self.beta = beta
        self.interaction_funcs = interaction_funcs
        self.interaction_funcs_string = interaction_funcs_string
        # normalize
        self.beta /= np.sum(self.beta)
        print("Combined function:", str(self))

    def function(self):
        if len(self.interaction_funcs) != 0:
            def final_function(q_matrix: np.ndarray, proficiency_level: np.ndarray, discrimination: np.ndarray):
                prediction = 0
                for i in range(len(self.interaction_funcs)):
                    prediction += self.beta[i] * self.interaction_funcs[i](q_matrix, proficiency_level, discrimination)
                return prediction
            return final_function
        else:
            return None

    def update(self, q_matrix: np.ndarray,
               proficiency_level: np.ndarray,
               discrimination: np.ndarray, ):
        self.q_matrix = q_matrix
        self.proficiency_level = proficiency_level.copy()
        self.discrimination = discrimination.copy()
        self.interaction.update(q_matrix, proficiency_level, discrimination)
