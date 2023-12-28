import operator
import numpy as np

from deap import base, creator, tools, gp, algorithms

import SCDM.operators as operators  # not build-in type!
from .eval import accuracy, area_under_curve, f1_score
from .utility import mut_uniform_with_pruning, sel_tournament, exam, init_interaction_function


class InteractionFunc:
    def __init__(self, train_set):
        self.train_set = train_set
        self.proficiency = None
        self.difficulty = None
        self.discrimination = None

        # f(discrimination, proficiency - difficulty, q_matrix_line)
        self.input_type = [np.float64, np.ndarray, np.ndarray]
        self.output_type = np.float64

        # construction set
        self.primitive_set = gp.PrimitiveSetTyped("main", self.input_type, self.output_type)
        self.primitive_set_init()

        # gp toolbox configuration
        self.toolbox = base.Toolbox()
        self.toolbox_init()

        # gp multi statistics configuration
        self.multi_statistics = tools.MultiStatistics(
            AUC=tools.Statistics(lambda ind: ind.fitness.values[0]),
            accuracy=tools.Statistics(lambda ind: ind.fitness.values[1]),
        )

        self.multi_statistics.register("min", np.min)
        self.multi_statistics.register("max", np.max)

        # other settings
        self.population = None
        self.hof = None

    def primitive_set_init(self):
        # including all necessary base functions (meet monotonicity assumption)
        self.primitive_set.addPrimitive(operators.add, [np.ndarray, np.ndarray], np.ndarray)
        self.primitive_set.addPrimitive(operators.add, [np.ndarray, np.float64], np.ndarray)
        self.primitive_set.addPrimitive(operators.add, [np.float64, np.float64], np.float64)
        self.primitive_set.addPrimitive(operators.mul, [np.ndarray, np.ndarray], np.ndarray)
        self.primitive_set.addPrimitive(operators.mul, [np.ndarray, np.float64], np.ndarray)
        self.primitive_set.addPrimitive(operators.mul, [np.float64, np.float64], np.float64)
        self.primitive_set.addPrimitive(operators.dot, [np.ndarray, np.ndarray], np.float64)
        self.primitive_set.addPrimitive(operators.tanh, [np.ndarray], np.ndarray)
        # rename arguments
        # De: discrimination
        # PDk: proficiency_level - difficulty
        # Q: Q-matrix
        self.primitive_set.renameArguments(ARG0="De", ARG1="PDk", ARG2="Q")

    def toolbox_init(self):
        # register all genetic operations
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.primitive_set, min_=5, max_=5)
        self.toolbox.register("individual", tools.initIterate, creator.individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.primitive_set)

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("select", sel_tournament, tournament_size=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", mut_uniform_with_pruning, pset=self.primitive_set)

        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    def evaluate(self, individual):
        currentInteractionFunc = self.toolbox.compile(expr=individual)
        y_pred, y_true = exam(self.train_set,
                              self.proficiency,
                              self.difficulty,
                              self.discrimination,
                              currentInteractionFunc,)

        acc = accuracy(y_pred, y_true)
        auc = area_under_curve(y_pred, y_true)
        return auc, acc,

    def train(self):
        population_size = 200
        ngen = 10
        cxpb = 0.5
        mutpb = 0.1
        self.population = self.toolbox.population(n=population_size)
        # create hall of fame to record best individual
        if self.hof is None:
            self.hof = tools.HallOfFame(maxsize=1)
        self.population, _ = algorithms.eaSimple(self.population, self.toolbox,
                                                 cxpb, mutpb, ngen,
                                                 stats=self.multi_statistics,
                                                 halloffame=self.hof, verbose=False)

    def unpack(self, is_compiled=False):
        if self.hof:
            if is_compiled:
                return self.toolbox.compile(expr=self.hof.items[0])
            else:
                return self.hof.items[0]
        else:
            return init_interaction_function

    def update(self, proficiency, difficulty, discrimination):
        self.proficiency = proficiency.copy()
        self.difficulty = difficulty.copy()
        self.discrimination = discrimination.copy()


class GeneticInteractionFunc:
    def __init__(self, train_set, train_size):
        # genetic programming and algorithm init
        creator.create("fitness_if", base.Fitness, weights=(1.0, 1.0))
        creator.create("individual", gp.PrimitiveTree, fitness=creator.fitness_if)

        self.train_size = train_size

        self.proficiency = None
        self.difficulty = None
        self.discrimination = None

        self.train_set = train_set
        self.interaction = InteractionFunc(train_set)

        self.interaction_funcs = []
        self.interaction_funcs_string = []

    def __str__(self):
        if len(self.interaction_funcs) != 0:
            return self.interaction_funcs_string[0]
        else:
            return "default"

    def evaluation(self, test_data) -> tuple:
        current_interaction_func = self.function()
        prediction, truth = exam(test_data,
                                 self.proficiency,
                                 self.difficulty,
                                 self.discrimination,
                                 current_interaction_func,)

        acc = accuracy(prediction, truth)
        auc = area_under_curve(prediction, truth)
        f1 = f1_score(prediction, truth)

        return acc, auc, f1,

    def train(self):
        print("Genetic programming search")
        interaction_funcs = []
        interaction_funcs_string = []
        self.interaction.train()
        interaction_funcs.append(self.interaction.unpack(is_compiled=True))
        interaction_funcs_string.append(str(self.interaction.unpack()))
        self.interaction_funcs = interaction_funcs
        self.interaction_funcs_string = interaction_funcs_string
        print("Final Function:", str(self))

    def function(self):
        if len(self.interaction_funcs) != 0:
            def final_function(discrimination, proficiency_level, q_matrix):
                return self.interaction_funcs[0](discrimination, proficiency_level, q_matrix)
            return final_function
        else:
            return init_interaction_function

    def update(self, proficiency, difficulty, discrimination):
        self.proficiency = proficiency.copy()
        self.difficulty = difficulty.copy()
        self.discrimination = discrimination.copy()
        self.interaction.update(proficiency, difficulty, discrimination)
