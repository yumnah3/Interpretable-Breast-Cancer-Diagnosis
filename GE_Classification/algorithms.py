#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
import math
import numpy as np
import time
import warnings

from deap import tools

def pattern(seq):
    storage = {}
    max_freqs = []
#    previous_max = 0
    for length in range(5,int(len(seq)/2)+1):
        valid_strings = {}
        for start in range(0,len(seq)-length+1):
            valid_strings[start] = tuple(seq[start:start+length])
        candidates = set(valid_strings.values())
        if len(candidates) != len(valid_strings):
#                    print("Pattern found for " + str(length))
            storage = valid_strings
            freq = []
            for v in storage.values():
                if list(storage.values()).count(v) > 1:
                    freq.append(list(storage.values()).count(v))
            current_max = max(freq)
 #           if current_max == previous_max:
  #              pass
   #         else:
            max_freqs.append([length, current_max])
    #            previous_max = current_max
        else:
#                    print("No pattern found for " + str(length))
            break
    return max_freqs#set(v for v in storage.values() if list(storage.values()).count(v) > 1)

def varAnd(population, toolbox, cxpb, mutpb,
           bnf_grammar, codon_size, max_tree_depth, codon_consumption,
           invalidate_max_depth,
           genome_representation, max_genome_length):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    """
    offspring = [toolbox.clone(ind) for ind in population]
#    invalid = [ind for ind in population if ind.invalid]
#    print("number of invalids going to cross/mut", len(invalid))

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i],
                                                          bnf_grammar, 
                                                          max_tree_depth, 
                                                          codon_consumption,
                                                          invalidate_max_depth,
                                                          genome_representation,
                                                          max_genome_length)
            #del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        offspring[i], = toolbox.mutate(offspring[i], mutpb,
                                       codon_size, bnf_grammar, 
                                       max_tree_depth, codon_consumption,
                                       invalidate_max_depth,
                                       max_genome_length)
        #del offspring[i].fitness.values

    return offspring

class hofWarning(UserWarning):
    pass

def ge_eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, elite_size, 
                bnf_grammar, codon_size, max_tree_depth, 
                max_genome_length=None,
                points_train=None, points_test=None, codon_consumption='eager', 
                report_items=None,
                genome_representation='list',
                invalidate_max_depth=False,
                problem=None,
                stats=None, halloffame=None, 
                verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_, with some adaptations to run GE
    on GRAPE.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param elite_size: The number of best individuals to be copied to the 
                    next generation.
    :params bnf_grammar, codon_size, max_tree_depth: Parameters 
                    used to mapper the individuals after crossover and
                    mutation in order to check if they are valid.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    """
    
    if problem == 'lawnmower64':
        l = 64
        c = 1
    elif problem == 'lawnmower144':
        l = 144
        c = 1
    elif problem == 'lawnmower196':
        l = 196
        c = 1
    elif problem == 'multiplier2':
        l = 16
        c = 4
    elif problem == 'multiplier3' or problem == 'multiplier3_v2':
        l = 64
        c = 6
    else:
        try:
            l, c = np.shape(points_train[1])
        except(ValueError): #points_train[1] has a single column
            l = len(points_train[1])
            c = 1
    n_possible_fitnesses = l*c + 1 #mostly, c=1, but for example, for multiplier problems, c is the length of each output
    
    logbook = tools.Logbook()
    if report_items:
        logbook.header = report_items
    else:
        logbook.header = ['gen', 'invalid'] + (stats.fields if stats else [])
    
    if halloffame is None:
        if elite_size != 0:
            raise ValueError("You should add a hof object to use elitism.") 
        else:
            warnings.warn('You will not register results of the best individual while not using a hof object.', hofWarning)
            #logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['avg_length', 'avg_nodes', 'avg_depth', 'avg_used_codons', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']
    else:
        if halloffame.maxsize < 1:
            raise ValueError("HALLOFFAME_SIZE should be greater or equal to 1")
        if elite_size > halloffame.maxsize:
            raise ValueError("HALLOFFAME_SIZE should be greater or equal to ELITE_SIZE")         
#        if points_test:
#            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['fitness_test', 'best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes', 'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']
#        else:
#            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes', 'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']

    start_gen = time.time()
    evaluated_inds = 0
    # Evaluate the individuals with an invalid fitness
    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind, points_train)
            evaluated_inds += 1
        
    valid0 = [ind for ind in population if not ind.invalid]
    valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]
    if codon_consumption == 'cosmo_eager' or codon_consumption == 'cosmo_eager_depth' or codon_consumption == 'cosmo_total':
        cosmo_inds = len([ind for ind in valid if ind.cosmo])
    else:
        cosmo_inds = 0
    
    if len(valid0) != len(valid):
        warnings.warn("Warning: There are valid individuals with fitness = NaN in the population. We will avoid them in the statistics and selection process.")
    invalid = len(population) - len(valid0) #We use the original number of invalids in this case, because we just want to count the completely mapped individuals    
    
    list_structures = []
    if 'fitness_diversity' in report_items:
        list_fitnesses = []
    if 'behavioural_diversity' in report_items:
        behaviours = np.zeros([len(valid), len(valid[0].fitness_each_sample)], dtype=float)
    
    #for ind in offspring:
    for idx, ind in enumerate(valid):
        list_structures.append(str(ind.structure))
        if 'fitness_diversity' in report_items:
            list_fitnesses.append(str(ind.fitness.values[0]))
        if 'behavioural_diversity' in report_items:
            behaviours[idx, :] = ind.fitness_each_sample
        
    unique_structures = np.unique(list_structures, return_counts=False)  
    if 'fitness_diversity' in report_items:
        unique_fitnesses = np.unique(list_fitnesses, return_counts=False) 
    if 'behavioural_diversity' in report_items:
        unique_behaviours = np.unique(behaviours, axis=0)
    
    structural_diversity = len(unique_structures)/len(population)
    fitness_diversity = len(unique_fitnesses)/n_possible_fitnesses if 'fitness_diversity' in report_items else 0 #TODO generalise for other problems, because it only works if the fitness is proportional to the number of testcases correctly predicted
    behavioural_diversity = len(unique_behaviours)/len(population) if 'behavioural_diversity' in report_items else 0
 
    length = [len(ind.genome) for ind in valid]
    avg_length = sum(length)/len(length)
    
    nodes = [ind.nodes for ind in valid]
    avg_nodes = sum(nodes)/len(nodes)
    
    depth = [ind.depth for ind in valid]
    avg_depth = sum(depth)/len(depth)
    
    used_codons = [ind.used_codons for ind in valid]
    avg_used_codons = sum(used_codons)/len(used_codons)
    
    end_gen = time.time()
    generation_time = end_gen-start_gen
    
    # Update the hall of fame with the generated individuals
    if halloffame is not None:
        halloffame.update(valid)
        best_ind_length = len(halloffame.items[0].genome) 
        best_ind_nodes = halloffame.items[0].nodes
        best_ind_depth = halloffame.items[0].depth
        best_ind_used_codons = halloffame.items[0].used_codons
        if 'best_ind_pattern' in report_items:
            best_ind_pattern = pattern(halloffame.items[0].structure)
        else:
            best_ind_pattern = 0
        if not verbose:
            print("gen =", 0, ", Fitness =", halloffame.items[0].fitness.values[0], ", Invalids =", invalid, ", generation_time =", round(generation_time, 2))
        
    selection_time = 0
    
    if points_test:
        fitness_test = np.NaN
    
    record = stats.compile(population) if stats else {}
    if points_test: 
        logbook.record(gen=0, invalid=invalid, **record, 
                       fitness_test=fitness_test,
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       cosmo_inds=cosmo_inds,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       evaluated_inds=evaluated_inds,
                       best_ind_pattern=best_ind_pattern,
                       avg_n_cases=0,
                       selection_time=selection_time, 
                       generation_time=generation_time)
    else:
        logbook.record(gen=0, invalid=invalid, **record, 
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       cosmo_inds=cosmo_inds,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       evaluated_inds=evaluated_inds,
                       best_ind_pattern=best_ind_pattern,
                       avg_n_cases=0,
                       selection_time=selection_time, 
                       generation_time=generation_time)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(logbook.select("gen")[-1]+1, ngen + 1):
        start_gen = time.time()    
    
        # Select the next generation individuals
        start = time.time()    
        offspring = toolbox.select(valid, len(population)-elite_size)
        end = time.time()
        selection_time = end-start
        lexicase_cases = [ind.n_cases for ind in offspring]
        avg_n_cases = sum(lexicase_cases)/len(lexicase_cases)

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb,
                           bnf_grammar, codon_size, max_tree_depth, 
                           codon_consumption, invalidate_max_depth, 
                           genome_representation, max_genome_length)

#        if max_genome_length == 'auto': #We need to calculate the average used codons
#            valid_ = [ind for ind in offspring if not ind.invalid]
#            used_codons = [ind.used_codons for ind in valid_]
#            avg_used_codons = sum(used_codons)/len(used_codons)
        
        # Evaluate the individuals with an invalid fitness
        evaluated_inds = 0
        for ind in offspring:
            if max_genome_length: #if it is not None
                #If auto, we invalidate the individuals with tail greater than 
                #the average used codons in the previous generation
                if max_genome_length == 'auto': 
                    pass
  #                  tail = len(ind.genome) - ind.used_codons 
   #                 for i in range(tail - int(0.5*avg_used_codons)):
    #                    ind.genome.pop()
                #If it is a number, we invalidate the individuals with length
                #greater than this number
                else:
                    if len(ind.genome) > max_genome_length:
                        ind.invalid = True
            #Now, we evaluate the individual
            #Note that the individual can also be invalid for other reasons
            #so we always need to consider this in the fitness function
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind, points_train)
                evaluated_inds += 1
                
        #Update population for next generation
        population[:] = offspring
        #Include in the population the elitist individuals
        for i in range(elite_size):
            population.append(halloffame.items[i])
            
        valid0 = [ind for ind in population if not ind.invalid]
        valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]
        if codon_consumption == 'cosmo_eager' or codon_consumption == 'cosmo_eager_depth' or codon_consumption == 'cosmo_total':
            cosmo_inds = len([ind for ind in valid if ind.cosmo])
        else:
            cosmo_inds = 0
        
        if len(valid0) != len(valid):
            warnings.warn("Warning: There are valid individuals with fitness = NaN in the population. We will avoid in the statistics and selection process.")
        invalid = len(population) - len(valid0) #We use the original number of invalids in this case, because we just want to count the completely mapped individuals
        
        list_structures = []
        if 'fitness_diversity' in report_items:
            list_fitnesses = []
        if 'behavioural_diversity' in report_items:
            behaviours = np.zeros([len(valid), len(valid[0].fitness_each_sample)], dtype=float)
        
        #for ind in offspring:
        for idx, ind in enumerate(valid):
            #if ind.invalid == True:
            #    invalid += 1
            #else:
            list_structures.append(str(ind.structure))
            if 'fitness_diversity' in report_items:
                list_fitnesses.append(str(ind.fitness.values[0]))
            if 'behavioural_diversity' in report_items:
                behaviours[idx, :] = ind.fitness_each_sample
                
        unique_structures = np.unique(list_structures, return_counts=False)  
        if 'fitness_diversity' in report_items:
            unique_fitnesses = np.unique(list_fitnesses, return_counts=False) 
        if 'behavioural_diversity' in report_items:
            unique_behaviours = np.unique(behaviours, axis=0)
        
        structural_diversity = len(unique_structures)/len(population)
        fitness_diversity = len(unique_fitnesses)/n_possible_fitnesses if 'fitness_diversity' in report_items else 0 #TODO generalise for other problems, because it only works if the fitness is proportional to the number of testcases correctly predicted
        behavioural_diversity = len(unique_behaviours)/len(population) if 'behavioural_diversity' in report_items else 0
        
        length = [len(ind.genome) for ind in valid]
        avg_length = sum(length)/len(length)
        
        nodes = [ind.nodes for ind in valid]
        avg_nodes = sum(nodes)/len(nodes)
        
        depth = [ind.depth for ind in valid]
        avg_depth = sum(depth)/len(depth)
            
        used_codons = [ind.used_codons for ind in valid]
        avg_used_codons = sum(used_codons)/len(used_codons)
        
        end_gen = time.time()
        generation_time = end_gen-start_gen
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(valid)
            best_ind_length = len(halloffame.items[0].genome)
            best_ind_nodes = halloffame.items[0].nodes
            best_ind_depth = halloffame.items[0].depth
            best_ind_used_codons = halloffame.items[0].used_codons
            if 'best_ind_pattern' in report_items:
                best_ind_pattern = pattern(halloffame.items[0].structure)
            else:
                best_ind_pattern = 0
            if not verbose:
                print("gen =", gen, " Fitness =", halloffame.items[0].fitness.values[0], "best_ind depth = ", best_ind_depth, ", Invalids =", invalid, ", generation_time =", round(generation_time, 2), "diversity (beh/fit) = ", behavioural_diversity, fitness_diversity, "evaluated inds = ", evaluated_inds, "avg used codons =", avg_used_codons, "avg_n_cases =", avg_n_cases)
         #       print(best_ind_pattern)
         #       depth_hof = [ind.depth for ind in halloffame.items]
         #       min_depth_= min(depth_hof)
         #       for ind in halloffame.items:
         #           if ind.depth == min_depth_:
         #               print(ind.phenotype)
         #               print(ind.depth)
         #               print(ind.fitness.values[0])
         #               print()
         #               break
            if points_test:
                if gen < ngen:
                    fitness_test = np.NaN
                else:
                    fitness_test = toolbox.evaluate(halloffame.items[0], points_test)[0]
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        if points_test: 
            logbook.record(gen=gen, invalid=invalid, **record, 
                       fitness_test=fitness_test,
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       cosmo_inds=cosmo_inds,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       evaluated_inds=evaluated_inds,
                       best_ind_pattern=best_ind_pattern,
                       avg_n_cases=avg_n_cases,
                       selection_time=selection_time, 
                       generation_time=generation_time)
        else:
            logbook.record(gen=gen, invalid=invalid, **record, 
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       cosmo_inds=cosmo_inds,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       evaluated_inds=evaluated_inds,
                       best_ind_pattern=best_ind_pattern,
                       avg_n_cases=avg_n_cases,
                       selection_time=selection_time, 
                       generation_time=generation_time)
                
        if verbose:
            print(logbook.stream)

    return population, logbook

def ge_eaSimplePandemic(population, toolbox, cxpb, mutpb, ngen, elite_size, 
                bnf_grammar, codon_size, max_tree_depth, 
                max_genome_length=None,
                points_train=None, points_test=None, codon_consumption='eager', 
                report_items=None,
                genome_representation='list',
                invalidate_max_depth=False,
                problem=None,
                pandemic_approach='periodic', #periodic, periodic2, automatic
                period=None, #used for pandemic_approach='periodic' (should be a value) and for pandemic_approach='periodic2'(should be a list)
                stats=None, halloffame=None, 
                verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_, with some adaptations to run GE
    on GRAPE.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param elite_size: The number of best individuals to be copied to the 
                    next generation.
    :params bnf_grammar, codon_size, max_tree_depth: Parameters 
                    used to mapper the individuals after crossover and
                    mutation in order to check if they are valid.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    """
    
    if problem == 'lawnmower64':
        l = 64
        c = 1
    elif problem == 'lawnmower144':
        l = 144
        c = 1
    elif problem == 'lawnmower196':
        l = 196
        c = 1
    elif problem == 'multiplier2':
        l = 16
        c = 4
    elif problem == 'multiplier3' or problem == 'multiplier3_v2':
        l = 64
        c = 6
    else:
        try:
            l, c = np.shape(points_train[1])
        except(ValueError): #points_train[1] has a single column
            l = len(points_train[1])
            c = 1
    n_possible_fitnesses = l*c + 1 #mostly, c=1, but for example, for multiplier problems, c is the length of each output
    
    logbook = tools.Logbook()
    if report_items:
        logbook.header = report_items
    else:
        logbook.header = ['gen', 'invalid'] + (stats.fields if stats else [])
    
    if halloffame is None:
        if elite_size != 0:
            raise ValueError("You should add a hof object to use elitism.") 
        else:
            warnings.warn('You will not register results of the best individual while not using a hof object.', hofWarning)
            #logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['avg_length', 'avg_nodes', 'avg_depth', 'avg_used_codons', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']
    else:
        if halloffame.maxsize < 1:
            raise ValueError("HALLOFFAME_SIZE should be greater or equal to 1")
        if elite_size > halloffame.maxsize:
            raise ValueError("HALLOFFAME_SIZE should be greater or equal to ELITE_SIZE")         
#        if points_test:
#            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['fitness_test', 'best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes', 'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']
#        else:
#            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes', 'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']

    if pandemic_approach == 'periodic' and period == None:
        raise ValueError("For using the periodic approach, you should set a value for period.") 
    if pandemic_approach == 'periodic2':
        if type(period) is not list:
            raise ValueError("For periodic2, period should be a list")
        if period[0] == 0 or period[1] == 0:
            raise ValueError("Each period should be different of zero.")
        
    start_gen = time.time()
    evaluated_inds = 0
    # Evaluate the individuals with an invalid fitness
    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind, points_train)
            evaluated_inds += 1
        
    valid0 = [ind for ind in population if not ind.invalid]
    valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]
    if codon_consumption == 'cosmo_eager' or codon_consumption == 'cosmo_eager_depth' or codon_consumption == 'cosmo_total':
        cosmo_inds = len([ind for ind in valid if ind.cosmo])
    else:
        cosmo_inds = 0
    
    if len(valid0) != len(valid):
        warnings.warn("Warning: There are valid individuals with fitness = NaN in the population. We will avoid them in the statistics and selection process.")
    invalid = len(population) - len(valid0) #We use the original number of invalids in this case, because we just want to count the completely mapped individuals    
    
    list_structures = []
    if 'fitness_diversity' in report_items:
        list_fitnesses = []
    if 'behavioural_diversity' in report_items:
        behaviours = np.zeros([len(valid), len(valid[0].fitness_each_sample)], dtype=float)
    
    #for ind in offspring:
    for idx, ind in enumerate(valid):
        list_structures.append(str(ind.structure))
        if 'fitness_diversity' in report_items:
            list_fitnesses.append(str(ind.fitness.values[0]))
        if 'behavioural_diversity' in report_items:
            behaviours[idx, :] = ind.fitness_each_sample
        
    unique_structures = np.unique(list_structures, return_counts=False)  
    if 'fitness_diversity' in report_items:
        unique_fitnesses = np.unique(list_fitnesses, return_counts=False) 
    if 'behavioural_diversity' in report_items:
        unique_behaviours = np.unique(behaviours, axis=0)
    
    structural_diversity = len(unique_structures)/len(population)
    fitness_diversity = len(unique_fitnesses)/n_possible_fitnesses if 'fitness_diversity' in report_items else 0 #TODO generalise for other problems, because it only works if the fitness is proportional to the number of testcases correctly predicted
    behavioural_diversity = len(unique_behaviours)/len(population) if 'behavioural_diversity' in report_items else 0
 
    length = [len(ind.genome) for ind in valid]
    avg_length = sum(length)/len(length)
    
    nodes = [ind.nodes for ind in valid]
    avg_nodes = sum(nodes)/len(nodes)
    
    depth = [ind.depth for ind in valid]
    avg_depth = sum(depth)/len(depth)
    
    used_codons = [ind.used_codons for ind in valid]
    avg_used_codons = sum(used_codons)/len(used_codons)
    
    end_gen = time.time()
    generation_time = end_gen-start_gen
    
    # Update the hall of fame with the generated individuals
    if halloffame is not None:
        halloffame.update(valid)
        best_ind_length = len(halloffame.items[0].genome) 
        best_ind_nodes = halloffame.items[0].nodes
        best_ind_depth = halloffame.items[0].depth
        best_ind_used_codons = halloffame.items[0].used_codons
        if 'best_ind_pattern' in report_items:
            best_ind_pattern = pattern(halloffame.items[0].structure)
        else:
            best_ind_pattern = 0
        if not verbose:
            print("gen =", 0, ", Fitness =", halloffame.items[0].fitness.values, ", Invalids =", invalid, ", generation_time =", round(generation_time, 2))
        
    selection_time = 0
    
    if points_test:
        fitness_test = np.NaN
    
    record = stats.compile(population) if stats else {}
    if points_test: 
        logbook.record(gen=0, invalid=invalid, **record, 
                       fitness_test=fitness_test,
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       cosmo_inds=cosmo_inds,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       evaluated_inds=evaluated_inds,
                       best_ind_pattern=best_ind_pattern,
                       avg_n_cases=0,
                       selection_time=selection_time, 
                       generation_time=generation_time)
    else:
        logbook.record(gen=0, invalid=invalid, **record, 
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       cosmo_inds=cosmo_inds,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       evaluated_inds=evaluated_inds,
                       best_ind_pattern=best_ind_pattern,
                       avg_n_cases=0,
                       selection_time=selection_time, 
                       generation_time=generation_time)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    if pandemic_approach == 'periodic':
        period_ = 0
    elif pandemic_approach == 'periodic2':
        period0_ = 0
        period1_ = 0
    elif pandemic_approach == 'automatic':
        current_method = '1'
    
    for gen in range(logbook.select("gen")[-1]+1, ngen + 1):
        start_gen = time.time()    
    
        # Select the next generation individuals
        start = time.time()    
        if pandemic_approach == 'periodic':
            period_ += 1
            if period_ <= period:
                offspring = toolbox.select1(valid, len(population)-elite_size)
            elif period_ <= 2*period:
                offspring = toolbox.select2(valid, len(population)-elite_size)
            else:
                period_ = 1
                offspring = toolbox.select1(valid, len(population)-elite_size)
        elif pandemic_approach == 'periodic2':
            if period0_ == period[0] and period1_ == period[1]:
                period0_ = 0
                period1_ = 0
            if period0_ < period[0]:
                period0_ += 1
                offspring = toolbox.select1(valid, len(population)-elite_size)
            elif period1_ < period[1]:
                period1_ += 1
                offspring = toolbox.select2(valid, len(population)-elite_size)
        elif pandemic_approach == 'automatic':
            if current_method == '1':
                offspring = toolbox.select1(valid, len(population)-elite_size)
            elif current_method == '2':
                offspring = toolbox.select2(valid, len(population)-elite_size)
        end = time.time()
        selection_time = end-start
        lexicase_cases = [ind.n_cases for ind in offspring]
        avg_n_cases = sum(lexicase_cases)/len(lexicase_cases)
        
        for ind in offspring:
            ind.n_cases = 0
        
#        invalid_selected = [ind for ind in offspring if ind.invalid]
#        print("invalid selected =", len(invalid_selected))
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb,
                           bnf_grammar, codon_size, max_tree_depth, 
                           codon_consumption, invalidate_max_depth, 
                           genome_representation, max_genome_length)

#        if max_genome_length == 'auto': #We need to calculate the average used codons
#            valid_ = [ind for ind in offspring if not ind.invalid]
#            used_codons = [ind.used_codons for ind in valid_]
#            avg_used_codons = sum(used_codons)/len(used_codons)
        
        # Evaluate the individuals with an invalid fitness
        evaluated_inds = 0
        if pandemic_approach == 'automatic':
            change = True
        for ind in offspring:
            if max_genome_length: #if it is not None
                #If auto, we invalidate the individuals with tail greater than 
                #the average used codons in the previous generation
                if max_genome_length == 'auto': 
                    pass
                #greater than this number
                else:
                    if len(ind.genome) > max_genome_length:
                        ind.invalid = True
            #Now, we evaluate the individual
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind, points_train)
                evaluated_inds += 1
                if pandemic_approach == 'automatic':
                    if change:
                        if ind.fitness.values[0] < halloffame.items[0].fitness.values[0]:
                            change = False # if at least one new individuals improved the previous best fitness, we do not change the method
        if pandemic_approach == 'automatic' and change:
            if current_method == '1':
                current_method = '2'
            elif current_method == '2':
                current_method = '1'
        #Update population for next generation
        population[:] = offspring
        #Include in the population the elitist individuals
        for i in range(elite_size):
            population.append(halloffame.items[i])
            
        valid0 = [ind for ind in population if not ind.invalid]
        valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]
        if codon_consumption == 'cosmo_eager' or codon_consumption == 'cosmo_eager_depth' or codon_consumption == 'cosmo_total':
            cosmo_inds = len([ind for ind in valid if ind.cosmo])
        else:
            cosmo_inds = 0
        
        if len(valid0) != len(valid):
            warnings.warn("Warning: There are valid individuals with fitness = NaN in the population. We will avoid in the statistics and selection process.")
        invalid = len(population) - len(valid0) #We use the original number of invalids in this case, because we just want to count the completely mapped individuals
        
        list_structures = []
        if 'fitness_diversity' in report_items:
            list_fitnesses = []
        if 'behavioural_diversity' in report_items:
            behaviours = np.zeros([len(valid), len(valid[0].fitness_each_sample)], dtype=float)
        
        #for ind in offspring:
        for idx, ind in enumerate(valid):
            #if ind.invalid == True:
            #    invalid += 1
            #else:
            list_structures.append(str(ind.structure))
            if 'fitness_diversity' in report_items:
                list_fitnesses.append(str(ind.fitness.values[0]))
            if 'behavioural_diversity' in report_items:
                behaviours[idx, :] = ind.fitness_each_sample
                
        unique_structures = np.unique(list_structures, return_counts=False)  
        if 'fitness_diversity' in report_items:
            unique_fitnesses = np.unique(list_fitnesses, return_counts=False) 
        if 'behavioural_diversity' in report_items:
            unique_behaviours = np.unique(behaviours, axis=0)
        
        structural_diversity = len(unique_structures)/len(population)
        fitness_diversity = len(unique_fitnesses)/n_possible_fitnesses if 'fitness_diversity' in report_items else 0 #TODO generalise for other problems, because it only works if the fitness is proportional to the number of testcases correctly predicted
        behavioural_diversity = len(unique_behaviours)/len(population) if 'behavioural_diversity' in report_items else 0
        
        length = [len(ind.genome) for ind in valid]
        avg_length = sum(length)/len(length)
        
        nodes = [ind.nodes for ind in valid]
        avg_nodes = sum(nodes)/len(nodes)
        
        depth = [ind.depth for ind in valid]
        avg_depth = sum(depth)/len(depth)
            
        used_codons = [ind.used_codons for ind in valid]
        avg_used_codons = sum(used_codons)/len(used_codons)
        
        end_gen = time.time()
        generation_time = end_gen-start_gen
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(valid)
            best_ind_length = len(halloffame.items[0].genome)
            best_ind_nodes = halloffame.items[0].nodes
            best_ind_depth = halloffame.items[0].depth
            best_ind_used_codons = halloffame.items[0].used_codons
            if 'best_ind_pattern' in report_items:
                best_ind_pattern = pattern(halloffame.items[0].structure)
            else:
                best_ind_pattern = 0
            if not verbose:
                print("gen =", gen, " Fitness =", halloffame.items[0].fitness.values[0], "best_ind depth = ", best_ind_depth, ", Invalids =", invalid, ", generation_time =", round(generation_time, 2), "diversity (beh/fit) = ", behavioural_diversity, fitness_diversity, "evaluated inds = ", evaluated_inds, "avg used codons =", avg_used_codons, "avg_n_cases =", avg_n_cases)
            if points_test:
                if gen < ngen:
                    fitness_test = np.NaN
                else:
                    fitness_test = toolbox.evaluate(halloffame.items[0], points_test)[0]
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        if points_test: 
            logbook.record(gen=gen, invalid=invalid, **record, 
                       fitness_test=fitness_test,
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       cosmo_inds=cosmo_inds,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       evaluated_inds=evaluated_inds,
                       best_ind_pattern=best_ind_pattern,
                       avg_n_cases=avg_n_cases,
                       selection_time=selection_time, 
                       generation_time=generation_time)
        else:
            logbook.record(gen=gen, invalid=invalid, **record, 
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       cosmo_inds=cosmo_inds,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       evaluated_inds=evaluated_inds,
                       best_ind_pattern=best_ind_pattern,
                       avg_n_cases=avg_n_cases,
                       selection_time=selection_time, 
                       generation_time=generation_time)
                
        if verbose:
            print(logbook.stream)

    return population, logbook

def ge_eaSimpleMultiGE(population, toolbox, cxpb, mutpb, ngen, elite_size, 
                bnf_grammar, codon_size, max_tree_depth, 
                max_genome_length=None,
                points_train=None, points_test=None, codon_consumption='multiGE', 
                report_items=None,
                genome_representation='list',
                stats=None, halloffame=None, 
                verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_, with some adaptations to run GE
    on GRAPE.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param elite_size: The number of best individuals to be copied to the 
                    next generation.
    :params bnf_grammar, codon_size, max_tree_depth: Parameters 
                    used to mapper the individuals after crossover and
                    mutation in order to check if they are valid.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    """
    
    n_genomes = len(population[0].genome)
    if codon_consumption != 'multiGE':
        raise ValueError("This algorithm should be used only for multiGE")
    elif genome_representation != 'list':
        raise ValueError("multiGE is implemented only for genome_representation='list'")
    
    logbook = tools.Logbook()
    
    if halloffame is None:
        if elite_size != 0:
            raise ValueError("You should add a hof object to use elitism.") 
        else:
            warnings.warn('You will not register results of the best individual while not using a hof object.', hofWarning)
            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['avg_length', 'avg_nodes', 'avg_depth', 'avg_used_codons', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']
    else:
        if halloffame.maxsize < 1:
            raise ValueError("HALLOFFAME_SIZE should be greater or equal to 1")
        if elite_size > halloffame.maxsize:
            raise ValueError("HALLOFFAME_SIZE should be greater or equal to ELITE_SIZE")         
        if points_test:
            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['fitness_test', 'best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes', 'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']
        else:
            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes', 'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']

    start_gen = time.time()        
    # Evaluate the individuals with an invalid fitness
    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind, points_train)
        
    valid0 = [ind for ind in population if not ind.invalid]
    valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]
    if len(valid0) != len(valid):
        warnings.warn("Warning: There are valid individuals with fitness = NaN in the population. We will avoid them.")
    invalid = len(population) - len(valid0) #We use the original number of invalids in this case, because we just want to count the completely mapped individuals    
    
    list_structures = []
    if 'fitness_diversity' in report_items:
        list_fitnesses = []
    if 'behavioural_diversity' in report_items:
        behaviours = np.zeros([len(valid), len(valid[0].fitness_each_sample)], dtype=float)
    
    #for ind in offspring:
    for idx, ind in enumerate(valid):
        #if ind.invalid == True:
        #    invalid += 1
        #else:
        list_structures.append(str(ind.structure))
        if 'fitness_diversity' in report_items:
            list_fitnesses.append(str(ind.fitness.values[0]))
        if 'behavioural_diversity' in report_items:
            behaviours[idx, :] = ind.fitness_each_sample
            
    unique_structures = np.unique(list_structures, return_counts=False)  
    if 'fitness_diversity' in report_items:
        unique_fitnesses = np.unique(list_fitnesses, return_counts=False) 
    if 'behavioural_diversity' in report_items:
        unique_behaviours = np.unique(behaviours, axis=0)
    
    structural_diversity = len(unique_structures)/len(population)
    fitness_diversity = len(unique_fitnesses)/(len(points_train[1])+1) if 'fitness_diversity' in report_items else 0 #TODO generalise for other problems, because it only works if the fitness is proportional to the number of testcases correctly predicted
    behavioural_diversity = len(unique_behaviours)/len(population) if 'behavioural_diversity' in report_items else 0
 
    length_list = [[len(ind.genome[i]) for ind in valid] for i in range(n_genomes)]
    avg_length = [sum(length)/len(length) for length in length_list]
    
    nodes_list = [[ind.nodes[i] for ind in valid] for i in range(n_genomes)]
    avg_nodes = [sum(nodes)/len(nodes) for nodes in nodes_list]
    
    depth_list = [[ind.depth[i] for ind in valid] for i in range(n_genomes)]
    avg_depth = [sum(depth)/len(depth) for depth in depth_list]
    
    used_codons_list = [[ind.used_codons[i] for ind in valid] for i in range(n_genomes)]
    avg_used_codons = [sum(used_codons)/len(used_codons) for used_codons in used_codons_list]
    
    end_gen = time.time()
    generation_time = end_gen-start_gen
    
    # Update the hall of fame with the generated individuals
    if halloffame is not None:
        halloffame.update(valid)
        best_ind_length = len(halloffame.items[0].genome) 
        best_ind_nodes = halloffame.items[0].nodes
        best_ind_depth = halloffame.items[0].depth
        best_ind_used_codons = halloffame.items[0].used_codons
        if not verbose:
            print("gen =", 0, ", Fitness =", halloffame.items[0].fitness.values, ", generation_time =", generation_time)
        
    selection_time = 0
    
    if points_test:
        fitness_test = np.NaN
    
    record = stats.compile(population) if stats else {}
    if points_test: 
        logbook.record(gen=0, invalid=invalid, **record, 
                       fitness_test=fitness_test,
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time, 
                       generation_time=generation_time)
    else:
        logbook.record(gen=0, invalid=invalid, **record, 
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time, 
                       generation_time=generation_time)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(logbook.select("gen")[-1]+1, ngen + 1):
        start_gen = time.time()    
    
        # Select the next generation individuals
        start = time.time()    
        offspring = toolbox.select(population, len(population)-elite_size)
        end = time.time()
        selection_time = end-start
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb,
                           bnf_grammar, codon_size, max_tree_depth, 
                           codon_consumption, genome_representation,
                           max_genome_length)

#        if max_genome_length == 'auto': #We need to calculate the average used codons
#            valid_ = [ind for ind in offspring if not ind.invalid]
#            used_codons = [ind.used_codons for ind in valid_]
#            avg_used_codons = sum(used_codons)/len(used_codons)
        
        # Evaluate the individuals with an invalid fitness
        for ind in offspring:
            if max_genome_length: #if it is not None
                #If auto, we invalidate the individuals with tail greater than 
                #the average used codons in the previous generation
                if max_genome_length == 'auto': 
                    pass
  #                  tail = len(ind.genome) - ind.used_codons 
   #                 for i in range(tail - int(0.5*avg_used_codons)):
    #                    ind.genome.pop()
                #If it is a number, we invalidate the individuals with length
                #greater than this number
                else:
                    if len(ind.genome) > max_genome_length:
                        ind.invalid = True
            #Now, we evaluate the individual
            #Note that the individual can also be invalid for other reasons
            #so we always need to consider this in the fitness function
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind, points_train)
                
                
        #Update population for next generation
        population[:] = offspring
        #Include in the population the elitist individuals
        for i in range(elite_size):
            population.append(halloffame.items[i])
            
        valid0 = [ind for ind in population if not ind.invalid]
        valid = [ind for ind in valid0 if not math.isnan(ind.fitness.values[0])]
        if len(valid0) != len(valid):
            warnings.warn("Warning: There are valid individuals with fitness = NaN in the population. We will avoid in the statistics.")
        invalid = len(population) - len(valid0) #We use the original number of invalids in this case, because we just want to count the completely mapped individuals
        
        list_structures = []
        if 'fitness_diversity' in report_items:
            list_fitnesses = []
        if 'behavioural_diversity' in report_items:
            behaviours = np.zeros([len(valid), len(valid[0].fitness_each_sample)], dtype=float)
        
        #for ind in offspring:
        for idx, ind in enumerate(valid):
            #if ind.invalid == True:
            #    invalid += 1
            #else:
            list_structures.append(str(ind.structure))
            if 'fitness_diversity' in report_items:
                list_fitnesses.append(str(ind.fitness.values[0]))
            if 'behavioural_diversity' in report_items:
                behaviours[idx, :] = ind.fitness_each_sample
                
        unique_structures = np.unique(list_structures, return_counts=False)  
        if 'fitness_diversity' in report_items:
            unique_fitnesses = np.unique(list_fitnesses, return_counts=False) 
        if 'behavioural_diversity' in report_items:
            unique_behaviours = np.unique(behaviours, axis=0)
        
        structural_diversity = len(unique_structures)/len(population)
        fitness_diversity = len(unique_fitnesses)/(len(points_train[1])+1) if 'fitness_diversity' in report_items else 0 #TODO generalise for other problems, because it only works if the fitness is proportional to the number of testcases correctly predicted
        behavioural_diversity = len(unique_behaviours)/len(population) if 'behavioural_diversity' in report_items else 0
        
        length_list = [[len(ind.genome[i]) for ind in valid] for i in range(n_genomes)]
        avg_length = [sum(length)/len(length) for length in length_list]
        
        nodes_list = [[ind.nodes[i] for ind in valid] for i in range(n_genomes)]
        avg_nodes = [sum(nodes)/len(nodes) for nodes in nodes_list]
        
        depth_list = [[ind.depth[i] for ind in valid] for i in range(n_genomes)]
        avg_depth = [sum(depth)/len(depth) for depth in depth_list]
        
        used_codons_list = [[ind.used_codons[i] for ind in valid] for i in range(n_genomes)]
        avg_used_codons = [sum(used_codons)/len(used_codons) for used_codons in used_codons_list]
            
        end_gen = time.time()
        generation_time = end_gen-start_gen
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(valid)
            best_ind_length = len(halloffame.items[0].genome)
            best_ind_nodes = halloffame.items[0].nodes
            best_ind_depth = halloffame.items[0].depth
            best_ind_used_codons = halloffame.items[0].used_codons
            if not verbose:
                print("gen =", gen, ", Fitness =", halloffame.items[0].fitness.values, ", Invalids =", invalid, ", generation_time =", generation_time)
            if points_test:
                if gen < ngen:
                    fitness_test = np.NaN
                else:
                    fitness_test = toolbox.evaluate(halloffame.items[0], points_test)[0]
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        if points_test: 
            logbook.record(gen=gen, invalid=invalid, **record, 
                       fitness_test=fitness_test,
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time, 
                       generation_time=generation_time)
        else:
            logbook.record(gen=gen, invalid=invalid, **record, 
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time, 
                       generation_time=generation_time)
                
        if verbose:
            print(logbook.stream)

    return population, logbook
def ge_ea3pops(population0, population1, population2, toolbox, cxpb, mutpb, ngen, 
                bnf_grammar, codon_size, max_tree_depth, 
                max_genome_length=None,
                points_train=None, points_test=None, codon_consumption='eager', 
                report_items=None,
                genome_representation='list',
                stats=None, halloffame=None, 
                verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_, with some adaptations to run GE
    on GRAPE.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param elite_size: The number of best individuals to be copied to the 
                    next generation.
    :params bnf_grammar, codon_size, max_tree_depth: Parameters 
                    used to mapper the individuals after crossover and
                    mutation in order to check if they are valid.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    """
    
    logbook = tools.Logbook()
    
    if halloffame is None:
        warnings.warn('You will not register results of the best individual while not using a hof object.', hofWarning)
        logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['avg_length', 'avg_nodes', 'avg_depth', 'avg_used_codons', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']
    else:
        if halloffame.maxsize < 1:
            raise ValueError("HALLOFFAME_SIZE should be greater or equal to 1")
        if points_test:
            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['fitness_test', 'best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes', 'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']
        else:
            logbook.header = ['gen', 'invalid'] + (stats.fields if stats else []) + ['best_ind_length', 'avg_length', 'best_ind_nodes', 'avg_nodes', 'best_ind_depth', 'avg_depth', 'avg_used_codons', 'best_ind_used_codons', 'behavioural_diversity', 'structural_diversity', 'fitness_diversity', 'selection_time', 'generation_time']

    start_gen = time.time()        
    # Evaluate the individuals with an invalid fitness
    idx = 0
    for ind0 in population0:
        ind1 = population1[idx]
        ind2 = population2[idx]
        idx += 1
        if not ind0.fitness.valid:
            ind0.fitness.values = toolbox.evaluate(ind0, ind1, ind2, points_train)
            ind1.fitness.values = ind0.fitness.values
            ind2.fitness.values = ind0.fitness.values
        
    valid = []
    for i in range(len(population0)):
        if not population0[i].invalid:
            if not population1[i].invalid:
                if not population2[i].invalid:
                    valid.append(population0[i])
            
    invalid = len(population0) - len(valid)
    
    list_structures = []
    if 'fitness_diversity' in report_items:
        list_fitnesses = []
    if 'behavioural_diversity' in report_items:
        behaviours = np.zeros([len(valid), len(valid[0].fitness_each_sample)], dtype=float)
    
    #for ind in offspring:
    for idx, ind in enumerate(valid):
        #if ind.invalid == True:
        #    invalid += 1
        #else:
        list_structures.append(str(ind.structure))
        if 'fitness_diversity' in report_items:
            list_fitnesses.append(str(ind.fitness.values[0]))
        if 'behavioural_diversity' in report_items:
            behaviours[idx, :] = ind.fitness_each_sample
            
    unique_structures = np.unique(list_structures, return_counts=False)  
    if 'fitness_diversity' in report_items:
        unique_fitnesses = np.unique(list_fitnesses, return_counts=False) 
    if 'behavioural_diversity' in report_items:
        unique_behaviours = np.unique(behaviours, axis=0)
    
    structural_diversity = len(unique_structures)/len(population0)
    fitness_diversity = len(unique_fitnesses)/(len(points_train[1])+1) if 'fitness_diversity' in report_items else 0 #TODO generalise for other problems, because it only works if the fitness is proportional to the number of testcases correctly predicted
    behavioural_diversity = len(unique_behaviours)/len(population0) if 'behavioural_diversity' in report_items else 0
 
    length = [len(ind.genome) for ind in valid]
    avg_length = sum(length)/len(length)
    
    nodes = [ind.nodes for ind in valid]
    avg_nodes = sum(nodes)/len(nodes)
    
    depth = [ind.depth for ind in valid]
    avg_depth = sum(depth)/len(depth)
    
    used_codons = [ind.used_codons for ind in valid]
    avg_used_codons = sum(used_codons)/len(used_codons)
    
    end_gen = time.time()
    generation_time = end_gen-start_gen
    
    # Update the hall of fame with the generated individuals
    if halloffame is not None:
        halloffame.update(valid)
        best_ind_length = len(halloffame.items[0].genome) 
        best_ind_nodes = halloffame.items[0].nodes
        best_ind_depth = halloffame.items[0].depth
        best_ind_used_codons = halloffame.items[0].used_codons
        if not verbose:
            print("gen =", 0, ", Fitness =", halloffame.items[0].fitness.values, ", generation_time =", generation_time)
        
    selection_time = 0
    
    if points_test:
        fitness_test = np.NaN
    
    record = stats.compile(population0) if stats else {}
    if points_test: 
        logbook.record(gen=0, invalid=invalid, **record, 
                       fitness_test=fitness_test,
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time, 
                       generation_time=generation_time)
    else:
        logbook.record(gen=0, invalid=invalid, **record, 
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time, 
                       generation_time=generation_time)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(logbook.select("gen")[-1]+1, ngen + 1):
        start_gen = time.time()    
    
        # Select the next generation individuals
        start = time.time()    
        offspring0 = toolbox.select(population0, len(population0))
        offspring1 = toolbox.select(population1, len(population1))
        offspring2 = toolbox.select(population2, len(population2))
        end = time.time()
        selection_time = end-start
        # Vary the pool of individuals
        offspring0 = varAnd(offspring0, toolbox, cxpb, mutpb,
                           bnf_grammar, codon_size, max_tree_depth, 
                           codon_consumption, genome_representation,
                           max_genome_length)
#        offspring0.sort(key=lambda x: x.idx, reverse=False)
        offspring1 = varAnd(offspring1, toolbox, cxpb, mutpb,
                           bnf_grammar, codon_size, max_tree_depth, 
                           codon_consumption, genome_representation,
                           max_genome_length)
#        offspring1.sort(key=lambda x: x.idx, reverse=False)
        offspring2 = varAnd(offspring2, toolbox, cxpb, mutpb,
                           bnf_grammar, codon_size, max_tree_depth, 
                           codon_consumption, genome_representation,
                           max_genome_length)
#        offspring2.sort(key=lambda x: x.idx, reverse=False)
            
        idx = 0
        for ind0 in offspring0:
            ind1 = offspring1[idx]
            ind2 = offspring2[idx]
            idx += 1
            if not ind0.fitness.valid:
                ind0.fitness.values = toolbox.evaluate(ind0, ind1, ind2, points_train)
                ind1.fitness.values = ind0.fitness.values
                ind2.fitness.values = ind0.fitness.values
                
                
        #Update population for next generation
        population0[:] = offspring0
        population1[:] = offspring1
        population2[:] = offspring2
        
        valid = []
        for i in range(len(population0)):
            if not population0[i].invalid:
                if not population1[i].invalid:
                    if not population2[i].invalid:
                        valid.append(population0[i])
                
        invalid = len(population0) - len(valid)
        
        list_structures = []
        if 'fitness_diversity' in report_items:
            list_fitnesses = []
        if 'behavioural_diversity' in report_items:
            behaviours = np.zeros([len(valid), len(valid[0].fitness_each_sample)], dtype=float)
        
        for idx, ind in enumerate(valid):
            list_structures.append(str(ind.structure))
            if 'fitness_diversity' in report_items:
                list_fitnesses.append(str(ind.fitness.values[0]))
            if 'behavioural_diversity' in report_items:
                behaviours[idx, :] = ind.fitness_each_sample
                
        unique_structures = np.unique(list_structures, return_counts=False)  
        if 'fitness_diversity' in report_items:
            unique_fitnesses = np.unique(list_fitnesses, return_counts=False) 
        if 'behavioural_diversity' in report_items:
            unique_behaviours = np.unique(behaviours, axis=0)
        
        structural_diversity = len(unique_structures)/len(population0)
        fitness_diversity = len(unique_fitnesses)/(len(points_train[1])+1) if 'fitness_diversity' in report_items else 0 #TODO generalise for other problems, because it only works if the fitness is proportional to the number of testcases correctly predicted
        behavioural_diversity = len(unique_behaviours)/len(population0) if 'behavioural_diversity' in report_items else 0
        
        length = [len(ind.genome) for ind in valid]
        avg_length = sum(length)/len(length)
        
        nodes = [ind.nodes for ind in valid]
        avg_nodes = sum(nodes)/len(nodes)
        
        depth = [ind.depth for ind in valid]
        avg_depth = sum(depth)/len(depth)
        
        used_codons = [ind.used_codons for ind in valid]
        avg_used_codons = sum(used_codons)/len(used_codons)
        
        end_gen = time.time()
        generation_time = end_gen-start_gen
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(valid)
            best_ind_length = len(halloffame.items[0].genome)
            best_ind_nodes = halloffame.items[0].nodes
            best_ind_depth = halloffame.items[0].depth
            best_ind_used_codons = halloffame.items[0].used_codons
            if not verbose:
                print("gen =", gen, ", Fitness =", halloffame.items[0].fitness.values, ", Invalids =", invalid, ", generation_time =", generation_time)
            if points_test:
                if gen < ngen:
                    fitness_test = np.NaN
                else:
                    fitness_test = toolbox.evaluate(halloffame.items[0], points_test)[0]
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population0) if stats else {}
        if points_test: 
            logbook.record(gen=gen, invalid=invalid, **record, 
                       fitness_test=fitness_test,
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time, 
                       generation_time=generation_time)
        else:
            logbook.record(gen=gen, invalid=invalid, **record, 
                       best_ind_length=best_ind_length, avg_length=avg_length, 
                       best_ind_nodes=best_ind_nodes,
                       avg_nodes=avg_nodes,
                       best_ind_depth=best_ind_depth,
                       avg_depth=avg_depth,
                       avg_used_codons=avg_used_codons,
                       best_ind_used_codons=best_ind_used_codons,
                       behavioural_diversity=behavioural_diversity,
                       structural_diversity=structural_diversity,
                       fitness_diversity=fitness_diversity,
                       selection_time=selection_time, 
                       generation_time=generation_time)
                
        if verbose:
            print(logbook.stream)

    return population0, logbook