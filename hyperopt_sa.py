
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import math
import numpy as np
from nn_model import *


toolbox = base.Toolbox()


def rnd_lr(min, max):
    a = int(math.log10(min))
    b = int(math.log10(max))
    r = np.random.randint(a, b) * np.random.rand()
    return math.pow(10, r)


def init_model(_class, lr):
    model = _class(lr())
    return model


def eval_model(ind):
    score = ind.evaluate()
    return (score, )


def mutate_model(model):
    model.lr = toolbox.attr_lr()
    return model,


def crossover_models(model1, model2):
    return model1, model2


def clone_model(model):
    new_model = creator.Individual(model.lr)
    new_model.model.set_weights(model.model.get_weights())
    return new_model


class MyHallOfFame(tools.HallOfFame):
    def __init__(self, k):
        tools.HallOfFame.__init__(self, k)

    def insert(self, item):
        from bisect import bisect_right
        i = bisect_right(self.keys, item.fitness)
        self.items.insert(len(self) - i, item.lr)
        self.keys.insert(i, item.fitness)

    def update(self, population):
        if len(self) == 0 and self.maxsize !=0:
            # Working on an empty hall of fame is problematic for the
            # "for else"
            self.insert(population[0])

        for ind in population:
            if ind.fitness > self.keys[-1] or len(self) < self.maxsize:
                for hofer in self:
                    # Loop through the hall of fame to check for any
                    # similar individual
                    if self.similar(ind, hofer):
                        break
                else:
                    # The individual is unique and strictly better than
                    # the worst
                    if len(self) >= self.maxsize:
                        self.remove(-1)
                    self.insert(ind)


def main():
    CXPB, MUTPB = 0.5, 1
    NGEN = 1

    LR_MIN = 0.0001
    LR_MAX = 1

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", NnModel, fitness=creator.FitnessMax)

    toolbox.register("attr_lr", rnd_lr, LR_MIN, LR_MAX)
    toolbox.register("model", init_model, creator.Individual, lr=toolbox.attr_lr)
    toolbox.register("population", tools.initRepeat, list, toolbox.model)

    toolbox.register("clone", clone_model)

    toolbox.register("evaluate", eval_model)
    toolbox.register("mate", crossover_models)
    toolbox.register("mutate", mutate_model)
    toolbox.register("select", tools.selBest)

    hof = MyHallOfFame(1)
    pop = toolbox.population(n=3)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=stats, halloffame=hof, verbose=__debug__)

    print("Best LearningRate:", hof.items[0])
    print("Best Accuracy:", hof.keys[0].values[0])

    return hof.items[0], hof.keys[0].values[0]


if __name__ == "__main__":
    main()

