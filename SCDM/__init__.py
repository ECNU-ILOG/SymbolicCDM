__author__ = "Junhao Shen"
__version__ = "2.0"
__revision__ = "1.2.0"

from deap import base, gp, creator
from . import model

# genetic programming and algorithm init
creator.create("fitness_if", base.Fitness, weights=(-1.0, 1.0))
creator.create("individual", gp.PrimitiveTree, fitness=creator.fitness_if)
creator.create("fitness_arg", base.Fitness, weights=(1.0,))
creator.create("individual_argument", list, fitness=creator.fitness_arg)
