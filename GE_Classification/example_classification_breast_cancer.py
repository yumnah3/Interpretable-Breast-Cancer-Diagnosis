# -*- coding: utf-8 -*-
import grape
import algorithms

from os import path
import pandas as pd
import numpy as np
from deap import creator, base, tools
import random
import csv

from functions import sigmoid
from functions import add, sub, mul, pdiv

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve
import sys

# Suppressing Warnings:
import warnings
warnings.filterwarnings("ignore")

run = 1
problem = 'breast_cancer'
scenario = 0#int(sys.argv[1])

N_RUNS = 1#30

def setDataSet(problem, RANDOM_SEED):
    np.random.seed(RANDOM_SEED)
    if problem == 'breast_cancer':
        if scenario == 0:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/cc_segments_adasyn_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/holdout_10.csv", sep=",")    
        if scenario == 1:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/cc_segments_borderline_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/holdout_10.csv", sep=",")    
        if scenario == 2:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/cc_segments_smote-enn_training (1).csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/holdout_10.csv", sep=",")    
        if scenario == 3:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/cc_segments_smote_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/holdout_10.csv", sep=",")    
        if scenario == 4:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/cc_segments_smotenc_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/holdout_10.csv", sep=",")    
        if scenario == 5:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/cc_segments_smotetomek_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/holdout_10.csv", sep=",")    
        if scenario == 6:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/cc_segments_svmsmote_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/holdout_10.csv", sep=",")    
        if scenario == 7:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/cc_segments_mixup_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/holdout_10.csv", sep=",")    
        if scenario == 8:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/cc_segments_STEM_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC_segments/holdout_10.csv", sep=",")
        if scenario == 9:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_adasyn_trainning.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_holdout.csv", sep=",")
        if scenario == 10:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_borderlinesmote_trainning.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_holdout.csv", sep=",")
        if scenario == 11:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_smote-enn_trainning.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_holdout.csv", sep=",")
        if scenario == 12:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_smote_trainning.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_holdout.csv", sep=",")
        if scenario == 13:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_smotenc_trainning.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_holdout.csv", sep=",")
        if scenario == 14:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_smotetomek_trainning.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_holdout.csv", sep=",")
        if scenario == 15:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_svmsmote_trainning.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_holdout.csv", sep=",")
        if scenario == 16:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_mixup_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_holdout.csv", sep=",")
        if scenario == 17:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_STEM_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/MLO_segments/mlo_segments_holdout.csv", sep=",")
        if scenario == 18:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_segments_adasyn_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_segments_holdout.csv", sep=",")
            data_test = data_test.iloc[:, 1:] #first column of test doesn't have a name (and it's just an index)
        if scenario == 19:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_segments_borderline_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_segments_holdout.csv", sep=",")
            data_test = data_test.iloc[:, 1:] #first column of test doesn't have a name (and it's just an index)
        if scenario == 20:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_segments_smote-enn_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_segments_holdout.csv", sep=",")
            data_test = data_test.iloc[:, 1:] #first column of test doesn't have a name (and it's just an index)
        if scenario == 21:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_segments_smote_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_segments_holdout.csv", sep=",")
            data_test = data_test.iloc[:, 1:] #first column of test doesn't have a name (and it's just an index)
        if scenario == 22:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_segments_smote-nc_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_segments_holdout.csv", sep=",")
            data_test = data_test.iloc[:, 1:] #first column of test doesn't have a name (and it's just an index)
        if scenario == 23:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_segments_smotetomek_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_segments_holdout.csv", sep=",")
            data_test = data_test.iloc[:, 1:] #first column of test doesn't have a name (and it's just an index)
        if scenario == 24:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_segments_svmsmote_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_segments_holdout.csv", sep=",")
            data_test = data_test.iloc[:, 1:] #first column of test doesn't have a name (and it's just an index)
        if scenario == 25:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_segments_mixup_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_segments_holdout.csv", sep=",")
            data_test = data_test.iloc[:, 1:] #first column of test doesn't have a name (and it's just an index)
        if scenario == 26:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_STEM_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/CC&MLO_segments/cc&mlo_segments_holdout.csv", sep=",")
            data_test = data_test.iloc[:, 1:] #first column of test doesn't have a name (and it's just an index)
        if scenario == 27:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_adasyn_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_holdout.csv", sep=",")
        if scenario == 28:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_borderline_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_holdout.csv", sep=",")
        if scenario == 29:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_smoten_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_holdout.csv", sep=",")
        if scenario == 30:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_smote_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_holdout.csv", sep=",")
        if scenario == 31:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_smotenc_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_holdout.csv", sep=",")
        if scenario == 32:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_smotetomek_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_holdout.csv", sep=",")
        if scenario == 33:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_svmsmote_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_holdout.csv", sep=",")
        if scenario == 34:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_Mixup_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_holdout.csv", sep=",")
        if scenario == 35:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_STEM_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/Full_images_CC&MLO/F_cc&MLO_holdout.csv", sep=",")
        if scenario == 36:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_adasyn_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_holdout.csv", sep=",")
        if scenario == 37:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_borderlinesmote_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_holdout.csv", sep=",")
        if scenario == 38:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_smote-enn_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_holdout.csv", sep=",")
        if scenario == 39:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_smote_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_holdout.csv", sep=",")
        if scenario == 40:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_smotenc_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_holdout.csv", sep=",")
        if scenario == 41:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_smotetomek_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_holdout.csv", sep=",")
        if scenario == 42:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_svmsmote_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_holdout.csv", sep=",")
        if scenario == 43:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_Mixup_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_holdout.csv", sep=",")
        if scenario == 44:
            data_train = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_STEM_training.csv", sep=",")
            data_test = pd.read_csv(r"datasets/STEM_DATASET/WBC/WBC_holdout.csv", sep=",")
            
        if scenario >= 36 and scenario <= 44:
            GRAMMAR_FILE = 'breast_cancer_30features.bnf'
        else:
            GRAMMAR_FILE = 'breast_cancer_52features.bnf'
            
        l = data_train.shape[0]
        Y_train = np.zeros([l,], dtype=int)
        for i in range(l):
            Y_train[i] = data_train['diagnosis'].iloc[i]
        data_train.pop('diagnosis')
        l = data_test.shape[0]
        Y_test = np.zeros([l,], dtype=int)
        for i in range(l):
            Y_test[i] = data_test['diagnosis'].iloc[i]
        data_test.pop('diagnosis')
        
        X_train = data_train.to_numpy().transpose()
        X_test = data_test.to_numpy().transpose()
        
    BNF_GRAMMAR = grape.Grammar(r"grammars/" + GRAMMAR_FILE)
    
    return X_train, Y_train, X_test, Y_test, BNF_GRAMMAR

def eval_test(individual, points):
    x = points[0]
    Y = points[1]
    
    pred = eval(individual.phenotype)
    pred_prob = sigmoid(pred) #probability of being of class 1
    fpr, tpr, thresholds = roc_curve(Y, pred_prob) # calculate roc curves
    gmeans = np.sqrt(tpr * (1-fpr)) # calculate the g-mean for each threshold
    ix = np.argmax(gmeans) # locate the index of the largest g-mean
    best_threshold=thresholds[ix]
    pred = np.where(pred_prob > best_threshold, 1, np.where(pred_prob < best_threshold, 0, random.randint(0, 1)))
    
    AUC = roc_auc_score(Y, pred_prob)
    
    acc = accuracy_score(Y, pred)
    
    f1 = f1_score(Y, pred)
    
    precision = precision_score(Y, pred)
    
    recall = recall_score(Y, pred)
    
    return AUC, acc, f1, precision, recall, best_threshold, fpr[ix], tpr[ix]
    
def fitness_eval1(individual, points):
    x = points[0]
    Y = points[1]
    
    if individual.invalid == True:
        return np.NaN,
    else:
        try:
            pred = eval(individual.phenotype)
            pred_prob = sigmoid(pred) #probability of being of class 1            
            fitness = 1 - roc_auc_score(Y, pred_prob)
        except (FloatingPointError, ZeroDivisionError, OverflowError,
                MemoryError, ValueError, TypeError):
            return np.NaN,
        
        return fitness,

toolbox = base.Toolbox()

# define a single objective, minimising fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create('Individual', grape.Individual, fitness=creator.FitnessMin)

toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual) 

toolbox.register("evaluate", fitness_eval1)

# Tournament selection:
toolbox.register("select", tools.selTournament, tournsize=6)

# Single-point crossover:
toolbox.register("mate", grape.crossover_onepoint)

# Flip-int mutation:
toolbox.register("mutate", grape.mutation_int_flip_per_codon)

POPULATION_SIZE = 200
MAX_INIT_TREE_DEPTH = 10
MIN_INIT_TREE_DEPTH = 4

MAX_GENERATIONS = 100
P_CROSSOVER = 0.8
P_MUTATION = 0.01
ELITE_SIZE = 1
HALLOFFAME_SIZE = 1

CODON_CONSUMPTION = 'lazy'
GENOME_REPRESENTATION = 'list'
MAX_GENOME_LENGTH = None#'auto'

MAX_TREE_DEPTH = 35 #equivalent to 17 in GP with this grammar
MAX_WRAPS = 0
CODON_SIZE = 255

REPORT_ITEMS = ['gen', 'invalid', 'avg', 'std', 'min', 'max', 
                'fitness_test',
                'test_AUC',
                'test_acc',
                'test_f1',
                'test_precision',
                'test_recall',
                'best_threshold', 
                'fpr', 
                'tpr',
                'best_ind_length', 'avg_length', 
                'best_ind_nodes', 'avg_nodes', 
                'best_ind_depth', 'avg_depth', 
                'avg_used_codons', 'best_ind_used_codons', 
                'behavioural_diversity',
                'fitness_diversity',
                'structural_diversity', 
                'evaluated_inds',
                'selection_time', 'generation_time',
                'frequency',
                'best_ind_phenotype']

def count_substrings(input_string, n):
    counts = [0] * n

    for i in range(n):
        substring = f'x[{i}]'
        start_index = 0
        while True:
            index = input_string.find(substring, start_index)
            if index == -1:
                break
            counts[i] += 1
            start_index = index + 1

    return counts

for i in range(N_RUNS):
    print()
    print()
    print("Run:", i + run)
    print()
    
    RANDOM_SEED = i + run
    
    X_train, Y_train, X_test, Y_test, BNF_GRAMMAR = setDataSet(problem, RANDOM_SEED) #We set up this inside the loop for the case in which the data is defined randomly

    random.seed(RANDOM_SEED) 

    # create initial population (generation 0):
    population = toolbox.populationCreator(pop_size=POPULATION_SIZE, 
                                       bnf_grammar=BNF_GRAMMAR, 
                                       min_init_depth=MIN_INIT_TREE_DEPTH,
                                       max_init_depth=MAX_INIT_TREE_DEPTH,
                                       codon_size=CODON_SIZE,
                                       codon_consumption=CODON_CONSUMPTION,
                                       genome_representation=GENOME_REPRESENTATION
                                        )
    
    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALLOFFAME_SIZE)
    
    # prepare the statistics object:
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)
    
    # perform the Grammatical Evolution flow:
    population, logbook = algorithms.ge_eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, elite_size=ELITE_SIZE,
                                              bnf_grammar=BNF_GRAMMAR, 
                                              codon_size=CODON_SIZE, 
                                              max_tree_depth=MAX_TREE_DEPTH,
                                              max_genome_length=MAX_GENOME_LENGTH,
                                              points_train=[X_train, Y_train], 
                                              points_test=[X_test, Y_test], 
                                              codon_consumption=CODON_CONSUMPTION,
                                              report_items=REPORT_ITEMS,
                                              genome_representation=GENOME_REPRESENTATION,      
                                              invalidate_max_depth=False,
                                              problem=problem,
                                              stats=stats, halloffame=hof, verbose=False)
    
    import textwrap
    best = hof.items[0].phenotype
    print("Best individual: \n","\n".join(textwrap.wrap(best,80)))
    print("\nTraining Fitness: ", hof.items[0].fitness.values[0])
    
    print("Depth: ", hof.items[0].depth)
    print("Length of the genome: ", len(hof.items[0].genome))
    print(f'Used portion of the genome: {hof.items[0].used_codons/len(hof.items[0].genome):.2f}')
    
    n_features = len(X_train)
    frequency = []
    for j in range(MAX_GENERATIONS):
        frequency.append(np.NaN)
    frequency_final = count_substrings(hof.items[0].phenotype, n_features)
    print(frequency_final)
    frequency.append(frequency_final)
    
    max_fitness_values, mean_fitness_values = logbook.select("max", "avg")
    min_fitness_values, std_fitness_values = logbook.select("min", "std")
    fitness_test = logbook.select("fitness_test")
    
    best_ind_length = logbook.select("best_ind_length")
    avg_length = logbook.select("avg_length")

    selection_time = logbook.select("selection_time")
    generation_time = logbook.select("generation_time")
    gen, invalid = logbook.select("gen", "invalid")
    avg_used_codons = logbook.select("avg_used_codons")
    best_ind_used_codons = logbook.select("best_ind_used_codons")
    
    best_ind_nodes = logbook.select("best_ind_nodes")
    avg_nodes = logbook.select("avg_nodes")

    best_ind_depth = logbook.select("best_ind_depth")
    avg_depth = logbook.select("avg_depth")

    behavioural_diversity = logbook.select("behavioural_diversity") 
    structural_diversity = logbook.select("structural_diversity") 
    fitness_diversity = logbook.select("fitness_diversity")     
    evaluated_inds = logbook.select("evaluated_inds") 
    
    best_phenotype = [float('nan')] * MAX_GENERATIONS
    best_phenotype.append(best)
    
    AUC, acc, f1, precision, recall, best_threshold, fpr, tpr = eval_test(hof.items[0], [X_test, Y_test])
    
    test_AUC = [float('nan')] * MAX_GENERATIONS
    test_AUC.append(AUC)
    
    test_acc = [float('nan')] * MAX_GENERATIONS
    test_acc.append(acc)
    
    test_f1 = [float('nan')] * MAX_GENERATIONS
    test_f1.append(f1)
    
    test_precision = [float('nan')] * MAX_GENERATIONS
    test_precision.append(precision)
    
    test_recall = [float('nan')] * MAX_GENERATIONS
    test_recall.append(recall)
    
    threshold = [float('nan')] * MAX_GENERATIONS
    threshold.append(best_threshold)
    
    test_fpr = [float('nan')] * MAX_GENERATIONS
    test_fpr.append(fpr)
    
    test_tpr = [float('nan')] * MAX_GENERATIONS
    test_tpr.append(tpr)
        
    r = RANDOM_SEED
    
    header = REPORT_ITEMS
    
    address = r"./results/" + problem + "/" + str(scenario) + "/"
        
    with open(address + str(r) + ".csv", "w", encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(header)
        for value in range(len(max_fitness_values)):
            writer.writerow([gen[value], invalid[value], mean_fitness_values[value],
                             std_fitness_values[value], min_fitness_values[value],
                             max_fitness_values[value], 
                             fitness_test[value],
                             test_AUC[value],
                             test_acc[value],
                             test_f1[value],
                             test_precision[value],
                             test_recall[value],
                             threshold[value],
                             test_fpr[value],
                             test_tpr[value],
                             best_ind_length[value], 
                             avg_length[value], 
                             best_ind_nodes[value],
                             avg_nodes[value],
                             best_ind_depth[value],
                             avg_depth[value],
                             avg_used_codons[value],
                             best_ind_used_codons[value], 
                             behavioural_diversity[value],
                             fitness_diversity[value],
                             structural_diversity[value],
                             evaluated_inds[value],
                             selection_time[value], 
                             generation_time[value],
                             frequency[value],
                             best_phenotype[value]])