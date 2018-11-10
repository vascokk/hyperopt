
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from pyspark import SparkContext, SparkConf
import math
import numpy as np
from nn_model import *
from individual_float import IndividualFloat

toolbox = base.Toolbox()
np.random.seed(42)


def rnd_lr(min, max):
    a = int(math.log10(min))
    b = int(math.log10(max))
    r = np.random.randint(a, b) * np.random.rand()
    return math.pow(10, r)


def init_individual(_class, lr):
    ind = _class(lr())
    return ind


def eval_individual(ind):
    model = NnModel(ind.value)
    score = model.evaluate()
    return (score, )


def mutate_individual(_ind):
    new_lr = toolbox.attr_lr()
    return creator.Individual(new_lr),


def crossover_individuals(ind1, ind2):
    return creator.Individual(abs(ind1.value-ind2.value)), creator.Individual(ind1.value+ind2.value)


def main():
    CXPB, MUTPB = 0.5, 1
    NGEN = 1

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", IndividualFloat, fitness=creator.FitnessMax)

    LR_MIN = 0.0001
    LR_MAX = 1
    toolbox.register("attr_lr", rnd_lr, LR_MIN, LR_MAX)
    toolbox.register("init_individual", init_individual, creator.Individual, lr=toolbox.attr_lr)
    toolbox.register("population", tools.initRepeat, list, toolbox.init_individual)


    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=1)  # needs to be k<= len(population)

    appName = 'Genetic Evolution'
    conf = SparkConf().setAppName(appName).setMaster("local")

    SparkContext.setSystemProperty('spark.rpc.message.maxSize', '2000')

    sc = SparkContext(conf=conf).getOrCreate()
    sc.setLogLevel('DEBUG')

    def sparkMap(algorithm, population):
        return sc.parallelize(population).map(algorithm).collect()


    toolbox.register("map", sparkMap)

    hof = tools.HallOfFame(1)
    pop = toolbox.population(n=8)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=stats, halloffame=hof, verbose=__debug__)


    print(pop)
    print(stats)

    print("HoF ======================= :")
    print("Best LR:", hof.items[0].value)
    print("Best score:", hof.keys[0].values[0])

    return hof.items[0].value, hof.keys[0].values[0]


if __name__ == "__main__":
    main()


