# -*- coding: utf-8 -*-
"""
Created on Tue May 10 06:53:28 2022

@author: allan
"""

import re
import math
from operator import attrgetter
import numpy as np
import random
import copy

#from scipy.stats import median_abs_deviation
#from statsmodels import robust   

def median_abs_deviation(arr, axis=0):
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Calculate the median along axis 0
    median = np.median(arr, axis=0)

    # Calculate the absolute deviations from the median along axis 0
    abs_deviations = np.abs(arr - median)

    # Calculate the median of the absolute deviations along axis 0
    mad = np.median(abs_deviations, axis=0)

    return mad

class Individual(object):
    """
    A GE individual.
    """

    def __init__(self, genome, grammar, max_depth, codon_consumption):
        """
        """
        
        self.genome = genome
        self.length = len(genome)
        if codon_consumption == 'lazy':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_lazy(genome, grammar, max_depth)
        elif codon_consumption == 'eager':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_eager(genome, grammar, max_depth)
        elif codon_consumption == 'leap':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure, self.tile_size, \
            self.effective_positions = mapper_leap(genome, grammar, max_depth)
        elif codon_consumption == 'leap2' or codon_consumption == 'leap3':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure, self.tile_size, \
            self.effective_positions = mapper_leap2(genome, grammar, max_depth)
        elif codon_consumption == 'multiGE':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_multi(genome, grammar, max_depth)
        elif codon_consumption == 'multichromosomalGE':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_multichromosomal(genome, grammar, max_depth)            
        elif codon_consumption == 'parameterised':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure = mapper_parameterised(genome, grammar, max_depth)    
        elif codon_consumption == 'cosmo_eager':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure, self.cosmo = mapper_cosmo(genome, grammar, max_depth)
        elif codon_consumption == 'cosmo_eager_depth':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure, self.cosmo = mapper_cosmo_ext(genome, grammar, max_depth)       
        elif codon_consumption == 'cosmo_total':
            self.phenotype, self.nodes, self.depth, \
            self.used_codons, self.invalid, self.n_wraps, \
            self.structure, self.cosmo = mapper_cosmo_total(genome, grammar, max_depth)     
        else:
            raise ValueError("Unknown mapper")
            
        self.fitness_each_sample = []
        self.n_cases = 0

def mutation_one_codon_leap(ind, mut_probability, codon_size, bnf_grammar, 
                                     max_depth, codon_consumption,
                                     invalidate_max_depth,
                                     max_genome_length): #TODO include code for this one
    """

    """
    if ind.invalid: #used_codons = 0
        possible_mutation_codons = len(ind.genome)
        continue_ = True
        genome = ind.genome.copy()
        while continue_:
            genome_mutated = genome.copy()
            codon_to_mutate = random.randint(0, possible_mutation_codons-1)
            genome_mutated[codon_to_mutate] = random.randint(0, codon_size)
            new_ind = reMap(ind, genome_mutated, bnf_grammar, max_depth, codon_consumption)
            if invalidate_max_depth: # In the mapping, if a ind surpasses max depth, it is invalid, and we won't redo crossover
                continue_ = False
            else: # We check if a ind surpasses max depth, and if so we will redo crossover
                continue_ = new_ind.depth > max_depth
        del new_ind.fitness.values
        return new_ind,        
    else:
        n_effective_codons = sum(ind.effective_positions)
        continue_ = True
        genome = ind.genome.copy()
        effective_positions = ind.effective_positions.copy()
        used_codons = ind.used_codons
        while continue_:
            genome_mutated = genome.copy()
            codon_to_mutate = random.randint(0, n_effective_codons-1)
            idx_ = 0
            for i in range(used_codons):
                if effective_positions[i]:
                    if codon_to_mutate == idx_:
                        genome_mutated[i] = random.randint(0, codon_size)
                        new_ind = reMap(ind, genome_mutated, bnf_grammar, max_depth, codon_consumption)
                        if invalidate_max_depth: # In the mapping, if a ind surpasses max depth, it is invalid, and we won't redo crossover
                            continue_ = False
                        else: # We check if a ind surpasses max depth, and if so we will redo crossover
                            continue_ = new_ind.depth > max_depth
                        break
                    idx_ += 1
        del new_ind.fitness.values
        return new_ind,        

def mutation_int_flip_per_codon_leap(ind, mut_probability, codon_size, bnf_grammar, 
                                     max_depth, codon_consumption,
                                     invalidate_max_depth,
                                     max_genome_length): #TODO include code for this one
    """

    """
    # Operation mutation within the effective part of the genome
    if ind.invalid: #used_codons = 0
        possible_mutation_codons = len(ind.genome)
    else:
        possible_mutation_codons = min(len(ind.genome), ind.used_codons) #in case of wrapping, used_codons can be greater than genome's length
    continue_ = True
    mutated_ = False
    genome = ind.genome.copy()
    effective_positions = ind.effective_positions.copy()
    while continue_:
        genome_mutated = genome.copy()
        for i in range(possible_mutation_codons):
            if effective_positions[i] or ind.invalid:
                if random.random() < mut_probability:
                    genome_mutated[i] = random.randint(0, codon_size)
                    mutated_ = True
 #                   break
    
        new_ind = reMap(ind, genome_mutated, bnf_grammar, max_depth, codon_consumption)

        if invalidate_max_depth: # In the mapping, if a ind surpasses max depth, it is invalid, and we won't redo crossover
            continue_ = False
        else: # We check if a ind surpasses max depth, and if so we will redo crossover
            continue_ = new_ind.depth > max_depth

    if mutated_:
        del new_ind.fitness.values
    return new_ind,

def mutation_int_flip_per_codon_leap3(ind, mut_probability, codon_size, bnf_grammar, max_depth, codon_consumption):
    """
    In this approach, we do not use codon_size.
    
    """
    tile_size = 0
    tile_n_rules = [] #Number of choices (PRs) for each position of the tile
    for i in range(len(bnf_grammar.production_rules)):
        if len(bnf_grammar.production_rules[i]) != 1: #The PR has a single option
            tile_n_rules.append(len(bnf_grammar.production_rules[i]))
            tile_size += 1   
            
    # Operation mutation within the effective part of the genome
    if ind.invalid: #used_codons = 0
        possible_mutation_codons = len(ind.genome)
    else:
        possible_mutation_codons = min(len(ind.genome), ind.used_codons) #in case of wrapping, used_codons can be greater than genome's length
    continue_ = True

    while continue_:
        for i in range(possible_mutation_codons):
            if ind.effective_positions[i]:
                if random.random() < mut_probability:
                    position_ = i % tile_size
                    ind.genome[i] = random.randint(0, tile_n_rules[position_])
    
        new_ind = reMap(ind, ind.genome, bnf_grammar, max_depth, codon_consumption)
        
        continue_ = new_ind.depth > max_depth

    return new_ind,

def crossover_onepoint_leap2(parent0, parent1, bnf_grammar, max_depth, codon_consumption,
                             invalidate_max_depth,
                             genome_representation, max_genome_length): #TODO include code for these two
    """
    
    """
    if parent0.invalid: #used_codons = 0
        possible_crossover_tile0 = int(len(parent0.genome)/parent0.tile_size)
    else:
        possible_crossover_tile0 = min(int(len(parent0.genome)/parent0.tile_size), int(parent0.used_codons/parent0.tile_size)) #in case of wrapping, used_codons can be greater than genome's length
    if parent1.invalid:
        possible_crossover_tile1 = int(len(parent1.genome)/parent1.tile_size)
    else:
        possible_crossover_tile1 = min(int(len(parent1.genome)/parent1.tile_size), int(parent1.used_codons/parent1.tile_size)) #in case of wrapping, used_codons can be greater than genome's length
    
    continue_ = True
    while continue_:
        #Set points for crossover within the effective part of the genomes
        try:
            point0 = random.randint(1, possible_crossover_tile0) * parent0.tile_size
            point1 = random.randint(1, possible_crossover_tile1) * parent0.tile_size
        except:
            ValueError(print(possible_crossover_tile0, possible_crossover_tile1, parent0.tile_size, parent0.used_codons, parent1.used_codons, parent0.invalid, parent1.invalid))
            
        #Operate crossover
        new_genome0 = parent0.genome[0:point0] + parent1.genome[point1:]
        new_genome1 = parent1.genome[0:point1] + parent0.genome[point0:]
        
        new_ind0 = reMap(parent0, new_genome0, bnf_grammar, max_depth, codon_consumption)
        new_ind1 = reMap(parent1, new_genome1, bnf_grammar, max_depth, codon_consumption)
        
        if invalidate_max_depth: # In the mapping, if a ind surpasses max depth, it is invalid, and we won't redo crossover
            continue_ = False
        else: # We check if a ind surpasses max depth, and if so we will redo crossover
            continue_ = new_ind0.depth > max_depth or new_ind1.depth > max_depth

    del new_ind0.fitness.values, new_ind1.fitness.values
    return new_ind0, new_ind1   

def crossover_onepoint_leap(parent0, parent1, bnf_grammar, max_depth, codon_consumption):
    """
    
    """
    if parent0.invalid: #used_codons = 0
        possible_crossover_tile0 = int(len(parent0.genome)/parent0.tile_size)
    else:
        possible_crossover_tile0 = min(int(len(parent0.genome)/parent0.tile_size), int(parent0.used_codons/parent0.tile_size)) #in case of wrapping, used_codons can be greater than genome's length
    if parent1.invalid:
        possible_crossover_tile1 = int(len(parent1.genome)/parent1.tile_size)
    else:
        possible_crossover_tile1 = min(int(len(parent1.genome)/parent1.tile_size), int(parent1.used_codons/parent1.tile_size)) #in case of wrapping, used_codons can be greater than genome's length
    
    continue_ = True
    while continue_:
        tile_position = random.randint(0, parent0.tile_size-1) #tile_size is equal for two parents
        #Set points for crossover within the effective part of the genomes
        point0 = random.randint(1, possible_crossover_tile0) * parent0.tile_size + tile_position
        point1 = random.randint(1, possible_crossover_tile1) * parent0.tile_size + tile_position
        
        #Operate crossover
        new_genome0 = parent0.genome[0:point0] + parent1.genome[point1:]
        new_genome1 = parent1.genome[0:point1] + parent0.genome[point0:]
        
        new_ind0 = reMap(parent0, new_genome0, bnf_grammar, max_depth, codon_consumption)
        new_ind1 = reMap(parent1, new_genome1, bnf_grammar, max_depth, codon_consumption)
        
        continue_ = new_ind0.depth > max_depth or new_ind1.depth > max_depth

    return new_ind0, new_ind1   

def mapper_leap2(genome, grammar, max_depth):
    """
    Mapper for LEAP GE
    Lazy approach.
    It consumes more than one codon per tile
    """
    
    effective_positions = [False]*len(genome)
    
    tile_size = 0
    tile_idx = [] #Index of each grammar.production_rules in the tile
    for i in range(len(grammar.production_rules)):
        if len(grammar.production_rules[i]) == 1: #The PR has a single option
            tile_idx.append(False)
        else:
            tile_idx.append(tile_size)
            tile_size += 1
    
    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    #n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and (idx_genome + tile_size) <= len(genome):
        NT_index = grammar.non_terminals.index(next_NT)
        if grammar.n_rules[NT_index] == 1: #there is a single PR for this non-terminal
            index_production_chosen = 0    
        else: #we consume one codon, and add the index to the structure
            if effective_positions[idx_genome + tile_idx[NT_index]] == False:
                index_production_chosen = genome[idx_genome + tile_idx[NT_index]] % grammar.n_rules[NT_index]
                effective_positions[idx_genome + tile_idx[NT_index]] = True
                structure.append(index_production_chosen)
            elif len(genome) >= idx_genome + 2*tile_size: #To use the next tile, we need to check if the current idx is at least 2*tile_size far from the end of the genome
                idx_genome += tile_size
                index_production_chosen = genome[idx_genome + tile_idx[NT_index]] % grammar.n_rules[NT_index]
                effective_positions[idx_genome + tile_idx[NT_index]] = True
                structure.append(index_production_chosen)
            else: #It means that we are using the last tile, so it is not possible to finish the mapping
                break
        
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
            idx_depth += 1
            nodes += 1
        elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
            pass        
        else: #it is a PR with more than one NT
            arity = grammar.production_rules[NT_index][index_production_chosen][2]
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome + tile_size #Because we did not increment the idx_genome with the last tile used in the process 
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure, tile_size, effective_positions

def mapper_leap(genome, grammar, max_depth):
    """
    Mapper for LEAP GE
    Lazy approach.
    It consumes one codon per tile
    """
    
    effective_positions = [False]*len(genome)
    
    tile_size = 0
    tile_idx = [] #Index of each grammar.production_rules in the tile
    for i in range(len(grammar.production_rules)):
        if len(grammar.production_rules[i]) == 1: #The PR has a single option
            tile_idx.append(False)
        else:
            tile_idx.append(tile_size)
            tile_size += 1
            
#    print(tile_size)
#    print(tile_idx)
#    print(grammar.non_terminals)
#    print(grammar.n_rules)
    
    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    #n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and (idx_genome + tile_size) <= len(genome):
        NT_index = grammar.non_terminals.index(next_NT)
        #print(phenotype)
#        print("idx", idx_genome)
        
        if grammar.n_rules[NT_index] == 1: #there is a single PR for this non-terminal
            index_production_chosen = 0    
#            print("not consumed", NT_index)
        else: #we consume one codon, and add the index to the structure
#            print("consumed", NT_index, tile_idx[NT_index])
            index_production_chosen = genome[idx_genome + tile_idx[NT_index]] % grammar.n_rules[NT_index]
            effective_positions[idx_genome + tile_idx[NT_index]] = True
#            print("index", genome[idx_genome + tile_idx[NT_index]])
            structure.append(index_production_chosen)
            idx_genome += tile_size
        
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
            idx_depth += 1
            nodes += 1
        elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
            pass        
        else: #it is a PR with more than one NT
            arity = grammar.production_rules[NT_index][index_production_chosen][2]
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
            
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure, tile_size, effective_positions

class Grammar(object):
    """
    Attributes:
    - non_terminals: list with each non-terminal (NT);
    - start_rule: first non-terminal;
    - production_rules: list with each production rule (PR), which contains in each position:
        - the PR itself as a string
        - 'non-terminal' or 'terminal'
        - the arity (number of NTs in the PR)
        - production choice label
        - True, if it is recursive, and False, otherwise
        - the minimum depth to terminate the mapping of all NTs of this PR
        - the minimum number of codons to terminate the mapping of all NTs of this PR
    - n_rules: df
    - max_codons_each_PR
    - min_codons_each_PR
    - max_depth_each_PR
    - initial_next_NT
    - initial_list_depth
    - initial_list_codons
    
    """
    def __init__(self, file_address):
        #Reading the file
        with open(file_address, "r") as text_file:
            bnf_grammar = text_file.read()
        #Getting rid of all the duplicate spaces
        bnf_grammar = re.sub(r"\s+", " ", bnf_grammar)

        #self.non_terminals = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>\s*::=",bnf_grammar)]
        self.non_terminals = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>\s*::=",bnf_grammar)]
        self.start_rule = self.non_terminals[0]
        for i in range(len(self.non_terminals)):
            bnf_grammar = bnf_grammar.replace(self.non_terminals[i] + " ::=", "  ::=")
        rules = bnf_grammar.split("::=")
        del rules[0]
        rules = [item.replace('\n',"") for item in rules]
        rules = [item.replace('\t',"") for item in rules]
        
        #list of lists (set of production rules for each non-terminal)
        self.production_rules = [i.split('|') for i in rules]
        for i in range(len(self.production_rules)):
            #Getting rid of all leading and trailing whitespaces
            self.production_rules[i] = [item.strip() for item in self.production_rules[i]]
            for j in range(len(self.production_rules[i])):
                #Include in the list the PR itself, NT or T, arity and the production choice label
                #if re.findall(r"\<(\w+)\>",self.production_rules[i][j]):
                if re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j]):                    
                    #arity = len(re.findall(r"\<(\w+)\>",self.production_rules[i][j]))
                    arity = len(re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j]))
                    self.production_rules[i][j] = [self.production_rules[i][j] , "non-terminal", arity, j]
                else:
                    self.production_rules[i][j] = [self.production_rules[i][j] , "terminal", 0, j] #arity 0
        #number of production rules for each non-terminal
        self.n_rules = [len(list_) for list_ in self.production_rules]
        


#        check_recursiveness = []
#        for PR in reversed(self.production_rules):
#            idx = self.production_rules.index(PR)
#            for j in range(len(PR)):
#                items = re.findall(r"\<([\(\)\w,-.]+)\>", PR[j][0])
#                for NT in items:
#                    if (NT not in check_recursiveness):
#                        for k in range(0, idx): #len(self.production_rules)
#                            for l in range(len(self.production_rules[k])):
#                                if NT in self.production_rules[k][l][0]:
#                                    check_recursiveness.append(self.non_terminals[k])


        
#        check_recursiveness = []
##        for i in range(len(self.production_rules)):
 #           for j in range(len(self.production_rules[i])):
 #               items = re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j][0])
                #n_non_terminals = len(items)
 #               for NT in items:
 #                   if (NT not in check_recursiveness) and (i < len(self.production_rules) - 1):
 #                       for k in range(i + 1, len(self.production_rules)):
 #                           for l in range(0, len(self.production_rules[k])):
 #                               if NT in self.production_rules[k][l][0]:
 #                                   check_recursiveness.append(self.non_terminals[i])
                
            
        
#begin        
        #Building list of non-terminals with recursive production-rules
#        check_recursiveness = []
#        try_recursiveness = self.non_terminals.copy()
#        recursive_indexes = []
#        for i in range(len(self.production_rules)):
#            check = 0
#            for j in range(len(self.production_rules[i])):
#                if self.production_rules[i][j][1] == 'non-terminal':
 #                   check += 1
 #           if check == 0: #if the PR has only terminals, it is not possible to be recursive
 #               if self.non_terminals[i] in try_recursiveness:
 #                   try_recursiveness.remove(self.non_terminals[i])
 #           else: #the PR has at least one recursive choice and it is therefore recursive
 #               recursive_indexes.append(i)
        
#        for item in reversed(try_recursiveness):
#            idx = self.non_terminals.index(item)
#            for i in range(len(self.production_rules[idx])):
#                if item in self.production_rules[idx][i][0]:
#                    if item not in check_recursiveness:
#                        check_recursiveness.append(item)
#                for recursive_item in check_recursiveness:
#                    if recursive_item in self.production_rules[idx][i][0]:
#                        if recursive_item not in check_recursiveness:
#                            check_recursiveness.append(recursive_item)
                
#        check_size = len(check_recursiveness) - 1
#        while check_size != len(check_recursiveness):
#            check_size = len(check_recursiveness)
 #           for item in check_recursiveness:
  
 #               for i in range(len(self.production_rules)):
 #                   for j in range(len(self.production_rules[i])):
 #                       if item in self.production_rules[i][j][0]:
 #                           if self.non_terminals[i] not in check_recursiveness:
 #                               check_recursiveness.append(self.non_terminals[i])
#end

#                        for recursive_item in check_recursiveness:
#                            if recursive_item in self.production_rules[idx][i][0]:
#                                if recursive_item not in check_recursiveness:
#                                    check_recursiveness.append(recursive_item)
                
            
        
        
#        check_recursiveness = []
#        check_size = len(try_recursiveness)
#        continue_ = True
#        while continue_:
#            for k in range(len(try_recursiveness)):
#                for i in range(len(self.production_rules)):
#                    for j in range(len(self.production_rules[i])):
#                        if i >= k:
#                            if try_recursiveness[k] in self.production_rules[i][j][0]:
#                                if self.non_terminals[i] not in check_recursiveness:
#                                    check_recursiveness.append(self.non_terminals[i])
#                                    if self.non_terminals[i] == '<nonboolean_feature>':
#                                        pass
#            if len(check_recursiveness) != check_size:
#                check_size = len(check_recursiveness)
#            else:
#                continue_ = False
                
        #Building list of non-terminals with recursive production-rules
#        try_recursiveness = self.non_terminals
#        check_recursiveness = []
#        check_size = len(try_recursiveness)
#        continue_ = True
#        while continue_:
#            for k in range(len(try_recursiveness)):
#                for i in range(len(self.production_rules)):
#                    for j in range(len(self.production_rules[i])):
#                        if i >= k:
#                            if try_recursiveness[k] in self.production_rules[i][j][0]:
#                                if self.non_terminals[i] not in check_recursiveness:
#                                    check_recursiveness.append(self.non_terminals[i])
#                                    if self.non_terminals[i] == '<nonboolean_feature>':
#                                        pass
#            if len(check_recursiveness) != check_size:
#                check_size = len(check_recursiveness)
#            else:
#                continue_ = False


            
        for i in range(len(self.production_rules)):
            for j in range(len(self.production_rules[i])):
                NTs_to_check_recursiveness = re.findall(r"\<([\(\)\w,-.]+)\>", self.production_rules[i][j][0])
                NTs_to_check_recursiveness = ['<' + item_ + '>' for item_ in NTs_to_check_recursiveness]
                unique_NTs = np.unique(NTs_to_check_recursiveness, return_counts=False) 
                recursive = False
      #          while unique_NTs.size and not recursive:
                for NT_to_check in unique_NTs:
                    stack = [self.non_terminals[i]]  
                    if NT_to_check in stack:
                        recursive = True
                        break
                    else:
                        #check_recursiveness.append(NT_to_check)
                        stack.append(NT_to_check)
                        recursive = check_recursiveness(self, NT_to_check, stack)
                        if recursive:
                            break
                        stack.pop()
                self.production_rules[i][j].append(recursive)
                

    
        

#finished

#        check_recursiveness = [self.start_rule]
#        check_size = len(check_recursiveness)
#        continue_ = True
#        while continue_:
#            for i in range(len(self.production_rules)):
#                for j in range(len(self.production_rules[i])):
#                    for k in range(len(check_recursiveness)):
#                        if check_recursiveness[k] in self.production_rules[i][j][0]:
#                            if self.non_terminals[i] not in check_recursiveness:
#                                check_recursiveness.append(self.non_terminals[i])
#            if len(check_recursiveness) != check_size:
#                check_size = len(check_recursiveness)
#            else:
#                continue_ = False

        
        #Including information of recursiveness in each production-rule
        #True if the respective non-terminal has recursive production-rules. False, otherwise
#        for i in range(len(self.production_rules)):
#            for j in range(len(self.production_rules[i])):
#                if self.production_rules[i][j][1] == 'terminal': #a terminal is never recursive
#                    recursive = False
#                    self.production_rules[i][j].append(recursive)
#                else: #a non-terminal can be recursive
#                    for k in range(len(check_recursiveness)):
#                        #Check if a recursive NT is in the current list of PR
                        #TODO I just changed from self.non_terminals[k] to check_recursiveness[k]
#                        if check_recursiveness[k] in self.production_rules[i][j][0]:
#                            recursive = True
#                            break #since we already found a recursive NT in this PR, we can stop
#                        else:
#                            recursive = False #if there is no recursive NT in this PR
#                    self.production_rules[i][j].append(recursive)
        
        #minimum depth from each non-terminal to terminate the mapping of all symbols
        NT_depth_to_terminate = [None]*len(self.non_terminals)
        #minimum depth from each production rule to terminate the mapping of all symbols
        part_PR_depth_to_terminate = list() #min depth for each non-terminal or terminal to terminate
        isolated_non_terminal = list() #None, if the respective position has a terminal
        #Separating the non-terminals within the same production rule
        for i in range(len(self.production_rules)):
            part_PR_depth_to_terminate.append( list() )
            isolated_non_terminal.append( list() )
            for j in range(len(self.production_rules[i])):
                part_PR_depth_to_terminate[i].append( list() )
                isolated_non_terminal[i].append( list() )
                if self.production_rules[i][j][1] == 'terminal':
                    isolated_non_terminal[i][j].append(None)
                    part_PR_depth_to_terminate[i][j] = 1
                    if not NT_depth_to_terminate[i]:
                        NT_depth_to_terminate[i] = 1
                else:
                    for k in range(self.production_rules[i][j][2]): #arity
                        part_PR_depth_to_terminate[i][j].append( list() )
                        #term = re.findall(r"\<(\w+)\>",self.production_rules[i][j][0])[k]
                        term = re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j][0])[k]
                        isolated_non_terminal[i][j].append('<' + term + '>')
        continue_ = True
        while continue_:
            #after filling up NT_depth_to_terminate, we need to run the loop one more time to
            #fill up part_PR_depth_to_terminate, so we check in the beginning
            if None not in NT_depth_to_terminate:
                continue_ = False 
            for i in range(len(self.non_terminals)):
                for j in range(len(self.production_rules)):
                    for k in range(len(self.production_rules[j])):
                        for l in range(len(isolated_non_terminal[j][k])):
                            if self.non_terminals[i] == isolated_non_terminal[j][k][l]:
                                if NT_depth_to_terminate[i]:
                                    if not part_PR_depth_to_terminate[j][k][l]:
                                        part_PR_depth_to_terminate[j][k][l] = NT_depth_to_terminate[i] + 1
                                        if [] not in part_PR_depth_to_terminate[j][k]:
                                            if not NT_depth_to_terminate[j]:
                                                NT_depth_to_terminate[j] = part_PR_depth_to_terminate[j][k][l]
        PR_depth_to_terminate = []
        for i in range(len(part_PR_depth_to_terminate)):
            for j in range(len(part_PR_depth_to_terminate[i])):
                #the min depth to terminate a PR is the max depth within the items of that PR
                if type(part_PR_depth_to_terminate[i][j]) == int:
                    depth_ = part_PR_depth_to_terminate[i][j]
                    PR_depth_to_terminate.append(depth_)
                    self.production_rules[i][j].append(depth_)
                else:
                    depth_ = max(part_PR_depth_to_terminate[i][j])
                    PR_depth_to_terminate.append(depth_)
                    self.production_rules[i][j].append(depth_)
                    
        #minimum number of codons from each non-terminal to terminate the mapping of all symbols
        NT_codons_to_terminate = [None]*len(self.non_terminals)
        #minimum number of codons from each production rule to terminate the mapping of all symbols
        part_PR_codons_to_terminate = list() #min number of codons for each non-terminal or terminal to terminate
        codons_non_terminal = list() #None, if the respective position has a terminal
        #Separating the non-terminals within the same production rule
 #       for i in range(len(self.production_rules)):
 #           part_PR_codons_to_terminate.append( list() )
 #           codons_non_terminal.append( list() )
 #           for j in range(len(self.production_rules[i])):
 #               part_PR_codons_to_terminate[i].append( list() )
 #               codons_non_terminal[i].append( list() )
 #               if self.production_rules[i][j][1] == 'terminal':
 #                   codons_non_terminal[i][j].append(None)
 #                   part_PR_codons_to_terminate[i][j] = 1
 #                   if not NT_codons_to_terminate[i]:
 #                       NT_codons_to_terminate[i] = 1
 #               else:
 #                   for k in range(self.production_rules[i][j][2]): #arity
 #                       part_PR_codons_to_terminate[i][j].append( list() )
 #                       term = re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j][0])[k]
 #                       codons_non_terminal[i][j].append('<' + term + '>')
 #       continue_ = True
 #       while continue_:
            #after filling up NT_depth_to_terminate, we need to run the loop one more time to
            #fill up part_PR_depth_to_terminate, so we check in the beginning
 #           if None not in NT_codons_to_terminate:
 #               continue_ = False 
 #           for i in range(len(self.non_terminals)):
 #               for j in range(len(self.production_rules)):
 #                   for k in range(len(self.production_rules[j])):
 #                       for l in range(len(codons_non_terminal[j][k])):
 #                           if self.non_terminals[i] == codons_non_terminal[j][k][l]:
 #                               if NT_codons_to_terminate[i]:
 #                                   if not part_PR_codons_to_terminate[j][k][l]:
 #                                       part_PR_codons_to_terminate[j][k][l] = NT_codons_to_terminate[i] + 1
 #                                       if [] not in part_PR_codons_to_terminate[j][k]:
 #                                           if not NT_codons_to_terminate[j]:
 #                                               NT_codons_to_terminate[j] = part_PR_codons_to_terminate[j][k][l]
 #       PR_codons_to_terminate = []
 #       for i in range(len(part_PR_codons_to_terminate)):
 #           for j in range(len(part_PR_codons_to_terminate[i])):
                #the min depth to terminate a PR is the max depth within the items of that PR
 #               if type(part_PR_codons_to_terminate[i][j]) == int:
 #                   codons_ = part_PR_codons_to_terminate[i][j]
 #                   PR_codons_to_terminate.append(codons_)
 #                   self.production_rules[i][j].append(codons_)
 #               else:
 #                   codons_ = sum(part_PR_codons_to_terminate[i][j])# - 1
 #                   PR_codons_to_terminate.append(codons_)
 #                   self.production_rules[i][j].append(codons_)
                    
        for i in range(len(self.production_rules)):
            part_PR_codons_to_terminate.append( list() )
            codons_non_terminal.append( list() )
            for j in range(len(self.production_rules[i])):
                part_PR_codons_to_terminate[i].append( list() )
                codons_non_terminal[i].append( list() )
                if self.production_rules[i][j][1] == 'terminal':
                    codons_non_terminal[i][j].append(None)
                    part_PR_codons_to_terminate[i][j] = 0 #part_PR_codons_to_terminate[i][j] represents the number of codons to terminate the remaining choices, then if this is a terminal, there are no remaining choices
                    if NT_codons_to_terminate[i] != 0:
                        NT_codons_to_terminate[i] = 0 #following the same idea from part_PR_codons_to_terminate[i][j]
                else:
                    for k in range(self.production_rules[i][j][2]): #arity
                        part_PR_codons_to_terminate[i][j].append( list() )
                        term = re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j][0])[k]
                        codons_non_terminal[i][j].append('<' + term + '>')
        continue_ = True
        while continue_:
            #after filling up NT_depth_to_terminate, we need to run the loop one more time to
            #fill up part_PR_depth_to_terminate, so we check in the beginning
            if None not in NT_codons_to_terminate:
                continue_ = False 
            for i in range(len(self.non_terminals)):
                for j in range(len(self.production_rules)):
                    for k in range(len(self.production_rules[j])):
                        for l in range(len(codons_non_terminal[j][k])):
                            if self.non_terminals[i] == codons_non_terminal[j][k][l]:
                                if NT_codons_to_terminate[i] != None:
                                    if not part_PR_codons_to_terminate[j][k][l]:
                                        part_PR_codons_to_terminate[j][k][l] = NT_codons_to_terminate[i] + 1
                                        if [] not in part_PR_codons_to_terminate[j][k]:
                                            if not NT_codons_to_terminate[j]:
                                                NT_codons_to_terminate[j] = sum(part_PR_codons_to_terminate[j][k])
        PR_codons_to_terminate = []
        for i in range(len(part_PR_codons_to_terminate)):
            for j in range(len(part_PR_codons_to_terminate[i])):
                #the min depth to terminate a PR is the max depth within the items of that PR
                if type(part_PR_codons_to_terminate[i][j]) == int:
                    codons_ = part_PR_codons_to_terminate[i][j] + 1 #part_PR_codons_to_terminate[i][j] represents the number of codons to terminate the remaining choices, then we add 1 regarding the current choice
                    PR_codons_to_terminate.append(codons_)
                    self.production_rules[i][j].append(codons_)
                else:
                    codons_ = sum(part_PR_codons_to_terminate[i][j]) + 1 #part_PR_codons_to_terminate[i][j] represents the number of codons to terminate the remaining choices, then we add 1 regarding the current choice
                    PR_codons_to_terminate.append(codons_)
                    self.production_rules[i][j].append(codons_)
                    
        #New attributes
        self.max_codons_each_PR = []
        self.min_codons_each_PR = []
        for PR in self.production_rules:
            choices_ = []
            for choice in PR:
                choices_.append(choice[6])
            self.max_codons_each_PR.append(max(choices_))
            self.min_codons_each_PR.append(min(choices_))
        self.max_depth_each_PR = []
        self.min_depth_each_PR = []
        for PR in self.production_rules:
            choices_ = []
            for choice in PR:
                choices_.append(choice[5])
            self.max_depth_each_PR.append(max(choices_))
            self.min_depth_each_PR.append(min(choices_))
            
        self.initial_next_NT = re.search(r"\<(\w+)\>",self.start_rule).group()
        n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>",self.start_rule)])
        self.initial_list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
        self.initial_list_codons = [] #it keeps the minimum number of codons necessary to terminate the mapping of each branch
        for term in re.findall(r"\<([\(\)\w,-.]+)\>",self.start_rule):
            NT_index = self.non_terminals.index('<' + term + '>')
            minimum_n_codons = []
            for PR in self.production_rules[NT_index]:
                minimum_n_codons.append(PR[6])
            self.initial_list_codons.append(min(minimum_n_codons))

class Grammar_parameterised(object):
    """
    Version of the class Grammar for reading parameterised grammars.
    Attributes:
    - non_terminals: list with each non-terminal (NT);
    - start_rule: first non-terminal;
    - production_rules: list with each production rule (PR), which contains in each position:
        - the PR itself as a string
        - 'non-terminal' or 'terminal'
        - the arity (number of NTs in the PR)
        - production choice label
        - True, if it is recursive, and False, otherwise
        - the minimum depth to terminate the mapping of all NTs of this PR
        - False, if it is not parameterised, and the parameter if True
    - n_rules: df
    - parameterised_non_terminals: list of parameterised NTs (first position shows NT, and second position shows parameter)
    
    """
    def __init__(self, file_address):
        #Reading the file
        with open(file_address, "r") as text_file:
            bnf_grammar = text_file.read()
        
        #Including a space before ::=, just in case we don't have anyone
        bnf_grammar = re.sub(r"::=", " ::=", bnf_grammar)
        #Getting rid of all the duplicate spaces
        bnf_grammar = re.sub(r"\s+", " ", bnf_grammar)

        #self.non_terminals = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>\s*::=",bnf_grammar)]
        #self.non_terminals = [term for term in re.findall(r"(\S*)\s*::=",bnf_grammar)]
        #self.non_terminals = [term for term in re.findall(r"(\<[\(\)\w,-.]+\>)\s*::=|(\<[\(\)\w,-.]+\>\(\w\))\s*::=",bnf_grammar)]
        self.non_terminals = [term for term in re.findall(r"(\<[\(\)\w,-.]+\>|\<[\(\)\w,-.]+\>\(\w\))\s*::=",bnf_grammar)]
        #volta
        for i in range(len(self.non_terminals)):
            bnf_grammar = bnf_grammar.replace(self.non_terminals[i] + " ::=", "  ::=")
            
        self.parameterised_non_terminals = []
        parameterised = []        
        for i in range(len(self.non_terminals)):
            if re.findall(r"(\<[\(\)\w,-.]+\>\(\w\))", self.non_terminals[i]):
                self.parameterised_non_terminals.append(self.non_terminals[i])
                parameterised.append(True)
                split_ = self.non_terminals[i].split('>(')
                self.non_terminals[i] = split_[0] + '>'
            else:
                parameterised.append(False)
                self.parameterised_non_terminals.append(False)
            
            
        #self.parameterised_non_terminals = [term for term in re.findall(r"(\<[\(\)\w,-.]+\>\(\w\))\s*::=",bnf_grammar)]
        
        
        #for i in range(len(self.parameterised_non_terminals)):
        #    bnf_grammar = bnf_grammar.replace(self.parameterised_non_terminals[i] + " ::=", "  ::=")
        
        for i in range(len(self.parameterised_non_terminals)):
            if self.parameterised_non_terminals[i]:
                self.parameterised_non_terminals[i] = self.parameterised_non_terminals[i].split('>(')
                self.parameterised_non_terminals[i][0] = self.parameterised_non_terminals[i][0] + '>'
                self.parameterised_non_terminals[i][1] = self.parameterised_non_terminals[i][1][:-1] 
        #    self.non_terminals.append(self.parameterised_non_terminals[i][0])
        
        
        
        

        #self.non_terminals = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>\s*::=",bnf_grammar)]
#        self.non_terminals = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>\s*::=",bnf_grammar)]
        self.start_rule = self.non_terminals[0]
#        for i in range(len(self.non_terminals)):
#            bnf_grammar = bnf_grammar.replace(self.non_terminals[i] + " ::=", "  ::=")
        rules = bnf_grammar.split("::=")
        del rules[0]
        rules = [item.replace('\n',"") for item in rules]
        rules = [item.replace('\t',"") for item in rules]
        
        #list of lists (set of production rules for each non-terminal)
        self.production_rules = [i.split('|') for i in rules]
        for i in range(len(self.production_rules)):
            #Getting rid of all leading and trailing whitespaces
            self.production_rules[i] = [item.strip() for item in self.production_rules[i]]
            for j in range(len(self.production_rules[i])):
                #Include in the list the PR itself, NT or T, arity and the production choice label
                #if re.findall(r"\<(\w+)\>",self.production_rules[i][j]):
                if re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j]):                    
                    #arity = len(re.findall(r"\<(\w+)\>",self.production_rules[i][j]))
                    arity = len(re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j]))
                    self.production_rules[i][j] = [self.production_rules[i][j] , "non-terminal", arity, j]
                else:
                    self.production_rules[i][j] = [self.production_rules[i][j] , "terminal", 0, j] #arity 0
        #number of production rules for each non-terminal
        self.n_rules = [len(list_) for list_ in self.production_rules]
        
        for i in range(len(self.production_rules)):
            for j in range(len(self.production_rules[i])):
                NTs_to_check_recursiveness = re.findall(r"\<([\(\)\w,-.]+)\>", self.production_rules[i][j][0])
                NTs_to_check_recursiveness = ['<' + item_ + '>' for item_ in NTs_to_check_recursiveness]
                unique_NTs = np.unique(NTs_to_check_recursiveness, return_counts=False) 
                recursive = False
                for NT_to_check in unique_NTs:
                    stack = [self.non_terminals[i]]  
                    if NT_to_check in stack:
                        recursive = True
                        break
                    else:
                        stack.append(NT_to_check)
                        recursive = check_recursiveness(self, NT_to_check, stack)
                        if recursive:
                            break
                        stack.pop()
                self.production_rules[i][j].append(recursive)

        #minimum depth from each non-terminal to terminate the mapping of all symbols
        NT_depth_to_terminate = [None]*len(self.non_terminals)
        #minimum depth from each production rule to terminate the mapping of all symbols
        part_PR_depth_to_terminate = list() #min depth for each non-terminal or terminal to terminate
        isolated_non_terminal = list() #None, if the respective position has a terminal
        #Separating the non-terminals within the same production rule
        for i in range(len(self.production_rules)):
            part_PR_depth_to_terminate.append( list() )
            isolated_non_terminal.append( list() )
            for j in range(len(self.production_rules[i])):
                part_PR_depth_to_terminate[i].append( list() )
                isolated_non_terminal[i].append( list() )
                if self.production_rules[i][j][1] == 'terminal':
                    isolated_non_terminal[i][j].append(None)
                    part_PR_depth_to_terminate[i][j] = 1
                    if not NT_depth_to_terminate[i]:
                        NT_depth_to_terminate[i] = 1
                else:
                    for k in range(self.production_rules[i][j][2]): #arity
                        part_PR_depth_to_terminate[i][j].append( list() )
                        #term = re.findall(r"\<(\w+)\>",self.production_rules[i][j][0])[k]
                        term = re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j][0])[k]
                        isolated_non_terminal[i][j].append('<' + term + '>')
        continue_ = True
        while continue_:
            #after filling up NT_depth_to_terminate, we need to run the loop one more time to
            #fill up part_PR_depth_to_terminate, so we check in the beginning
            if None not in NT_depth_to_terminate:
                continue_ = False 
            for i in range(len(self.non_terminals)):
                for j in range(len(self.production_rules)):
                    for k in range(len(self.production_rules[j])):
                        for l in range(len(isolated_non_terminal[j][k])):
                            if self.non_terminals[i] == isolated_non_terminal[j][k][l]:
                                if NT_depth_to_terminate[i]:
                                    if not part_PR_depth_to_terminate[j][k][l]:
                                        part_PR_depth_to_terminate[j][k][l] = NT_depth_to_terminate[i] + 1
                                        if [] not in part_PR_depth_to_terminate[j][k]:
                                            if not NT_depth_to_terminate[j]:
                                                NT_depth_to_terminate[j] = part_PR_depth_to_terminate[j][k][l]
        PR_depth_to_terminate = []
        for i in range(len(part_PR_depth_to_terminate)):
            for j in range(len(part_PR_depth_to_terminate[i])):
                #the min depth to terminate a PR is the max depth within the items of that PR
                if type(part_PR_depth_to_terminate[i][j]) == int:
                    depth_ = part_PR_depth_to_terminate[i][j]
                    PR_depth_to_terminate.append(depth_)
                    self.production_rules[i][j].append(depth_)
                else:
                    depth_ = max(part_PR_depth_to_terminate[i][j])
                    PR_depth_to_terminate.append(depth_)
                    self.production_rules[i][j].append(depth_)
                    
        for i in range(len(self.production_rules)):
            for j in range(len(self.production_rules[i])):
                if self.parameterised_non_terminals[i]:
                    self.production_rules[i][j].append(self.parameterised_non_terminals[i][1])
                else:
                    self.production_rules[i][j].append(False)
                #parameterised_ = False
                #for k in range(len(self.parameterised_non_terminals)):
                #    if self.parameterised_non_terminals[k]:
                #        if re.findall(self.parameterised_non_terminals[k][0], self.production_rules[i][j][0]):
                #            self.production_rules[i][j].append(self.parameterised_non_terminals[k][1])
                #            parameterised_ = True
                #if not parameterised_:
                #    self.production_rules[i][j].append(False)
                    
                    
class Grammar_parameterised_errado(object):
    """
    Version of the class Grammar for reading parameterised grammars.
    Attributes:
    - non_terminals: list with each non-terminal (NT);
    - start_rule: first non-terminal;
    - production_rules: list with each production rule (PR), which contains in each position:
        - the PR itself as a string
        - 'non-terminal' or 'terminal'
        - the arity (number of NTs in the PR)
        - production choice label
        - True, if it is recursive, and False, otherwise
        - the minimum depth to terminate the mapping of all NTs of this PR
    - n_rules: df
    
    """
    def __init__(self, file_address):
        #Reading the file
        with open(file_address, "r") as text_file:
            bnf_grammar = text_file.read()
        #Getting rid of all the duplicate spaces
        bnf_grammar = re.sub(r"\s+", " ", bnf_grammar)

        self.non_terminals = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>\s*::=",bnf_grammar)]
        self.parameterised_non_terminals = [term for term in re.findall(r"(\<[\(\)\w,-.]+\>\(\w\))\s*::=",bnf_grammar)]
        
        for i in range(len(self.non_terminals)):
            bnf_grammar = bnf_grammar.replace(self.non_terminals[i] + " ::=", "  ::=")
        for i in range(len(self.parameterised_non_terminals)):
            bnf_grammar = bnf_grammar.replace(self.parameterised_non_terminals[i] + " ::=", "  ::=")
        
        for i in range(len(self.parameterised_non_terminals)):
            self.parameterised_non_terminals[i] = self.parameterised_non_terminals[i].split('>(')
            self.parameterised_non_terminals[i][0] = self.parameterised_non_terminals[i][0] + '>'
            self.parameterised_non_terminals[i][1] = self.parameterised_non_terminals[i][1][:-1]
        self.start_rule = self.non_terminals[0]
        
        rules = bnf_grammar.split("::=")
        del rules[0]
        rules = [item.replace('\n',"") for item in rules]
        rules = [item.replace('\t',"") for item in rules]
        
        #list of lists (set of production rules for each non-terminal)
        self.production_rules = [i.split('|') for i in rules]
        for i in range(len(self.production_rules)):
            #Getting rid of all leading and trailing whitespaces
            self.production_rules[i] = [item.strip() for item in self.production_rules[i]]
            for j in range(len(self.production_rules[i])):
                #Include in the list the PR itself, NT or T, arity and the production choice label
                #if re.findall(r"\<(\w+)\>",self.production_rules[i][j]):
                if re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j]):                    
                    #arity = len(re.findall(r"\<(\w+)\>",self.production_rules[i][j]))
                    arity = len(re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j]))
                    self.production_rules[i][j] = [self.production_rules[i][j] , "non-terminal", arity, j]
                else:
                    self.production_rules[i][j] = [self.production_rules[i][j] , "terminal", 0, j] #arity 0
        #number of production rules for each non-terminal
        self.n_rules = [len(list_) for list_ in self.production_rules]
             
        for i in range(len(self.production_rules)):
            for j in range(len(self.production_rules[i])):
                NTs_to_check_recursiveness = re.findall(r"\<([\(\)\w,-.]+)\>", self.production_rules[i][j][0])
                NTs_to_check_recursiveness = ['<' + item_ + '>' for item_ in NTs_to_check_recursiveness]
                unique_NTs = np.unique(NTs_to_check_recursiveness, return_counts=False) 
                recursive = False
                for NT_to_check in unique_NTs:
                    stack = [self.non_terminals[i]]  
                    if NT_to_check in stack:
                        recursive = True
                        break
                    else:
                        stack.append(NT_to_check)
                        recursive = check_recursiveness(self, NT_to_check, stack)
                        if recursive:
                            break
                        stack.pop()
                self.production_rules[i][j].append(recursive)
                #Repeat for parameterised rules
                for NT_to_check in unique_NTs:
                    stack = [self.parameterised_non_terminals[i][0]]  
                    if NT_to_check in stack:
                        recursive = True
                        break
                    else:
                        stack.append(NT_to_check)
                        recursive = check_recursiveness_parameterised(self, NT_to_check, stack)
                        if recursive:
                            break
                        stack.pop()
                self.production_rules[i][j].append(recursive)
      
        #minimum depth from each non-terminal to terminate the mapping of all symbols
        NT_depth_to_terminate = [None]*len(self.non_terminals)
        #minimum depth from each production rule to terminate the mapping of all symbols
        part_PR_depth_to_terminate = list() #min depth for each non-terminal or terminal to terminate
        isolated_non_terminal = list() #None, if the respective position has a terminal
        #Separating the non-terminals within the same production rule
        for i in range(len(self.production_rules)):
            part_PR_depth_to_terminate.append( list() )
            isolated_non_terminal.append( list() )
            for j in range(len(self.production_rules[i])):
                part_PR_depth_to_terminate[i].append( list() )
                isolated_non_terminal[i].append( list() )
                if self.production_rules[i][j][1] == 'terminal':
                    isolated_non_terminal[i][j].append(None)
                    part_PR_depth_to_terminate[i][j] = 1
                    if not NT_depth_to_terminate[i]:
                        NT_depth_to_terminate[i] = 1
                else:
                    for k in range(self.production_rules[i][j][2]): #arity
                        part_PR_depth_to_terminate[i][j].append( list() )
                        term = re.findall(r"\<([\(\)\w,-.]+)\>",self.production_rules[i][j][0])[k]
                        isolated_non_terminal[i][j].append('<' + term + '>')
        continue_ = True
        while continue_:
            #after filling up NT_depth_to_terminate, we need to run the loop one more time to
            #fill up part_PR_depth_to_terminate, so we check in the beginning
            if None not in NT_depth_to_terminate:
                continue_ = False 
            for i in range(len(self.non_terminals)):
                for j in range(len(self.production_rules)):
                    for k in range(len(self.production_rules[j])):
                        for l in range(len(isolated_non_terminal[j][k])):
                            if self.non_terminals[i] == isolated_non_terminal[j][k][l]:
                                if NT_depth_to_terminate[i]:
                                    if not part_PR_depth_to_terminate[j][k][l]:
                                        part_PR_depth_to_terminate[j][k][l] = NT_depth_to_terminate[i] + 1
                                        if [] not in part_PR_depth_to_terminate[j][k]:
                                            if not NT_depth_to_terminate[j]:
                                                NT_depth_to_terminate[j] = part_PR_depth_to_terminate[j][k][l]
        PR_depth_to_terminate = []
        for i in range(len(part_PR_depth_to_terminate)):
            for j in range(len(part_PR_depth_to_terminate[i])):
                #the min depth to terminate a PR is the max depth within the items of that PR
                if type(part_PR_depth_to_terminate[i][j]) == int:
                    depth_ = part_PR_depth_to_terminate[i][j]
                    PR_depth_to_terminate.append(depth_)
                    self.production_rules[i][j].append(depth_)
                else:
                    depth_ = max(part_PR_depth_to_terminate[i][j])
                    PR_depth_to_terminate.append(depth_)
                    self.production_rules[i][j].append(depth_)
                    
def check_recursiveness(self, NT, stack):
    idx_NT = self.non_terminals.index(NT)
    for j in range(len(self.production_rules[idx_NT])):
        NTs_to_check_recursiveness = re.findall(r"\<([\(\)\w,-.]+)\>", self.production_rules[idx_NT][j][0])
        NTs_to_check_recursiveness = ['<' + item_ + '>' for item_ in NTs_to_check_recursiveness]
        unique_NTs = np.unique(NTs_to_check_recursiveness, return_counts=False) 
        recursive = False
  #      while unique_NTs.size and not recursive:
        for NT_to_check in unique_NTs:
            if NT_to_check in stack:
                recursive = True
                return recursive
            else:
                stack.append(NT_to_check) #Include the current NT to check it recursively
                recursive = check_recursiveness(self, NT_to_check, stack)
                if recursive:
                    return recursive
                stack.pop() #If the inclusion didn't show recursiveness, remove it before continuing
            #    return recursive
    return recursive#, stack

def check_recursiveness_parameterised(self, NT, stack):
    idx_NT = self.parameterised_non_terminals[0].index(NT)
    for j in range(len(self.production_rules[idx_NT])):
        NTs_to_check_recursiveness = re.findall(r"\<([\(\)\w,-.]+)\>", self.production_rules[idx_NT][j][0])
        NTs_to_check_recursiveness = ['<' + item_ + '>' for item_ in NTs_to_check_recursiveness]
        unique_NTs = np.unique(NTs_to_check_recursiveness, return_counts=False) 
        recursive = False
  #      while unique_NTs.size and not recursive:
        for NT_to_check in unique_NTs:
            if NT_to_check in stack:
                recursive = True
                return recursive
            else:
                stack.append(NT_to_check) #Include the current NT to check it recursively
                recursive = check_recursiveness(self, NT_to_check, stack)
                if recursive:
                    return recursive
                stack.pop() #If the inclusion didn't show recursiveness, remove it before continuing
            #    return recursive
    return recursive#, stack

def mapper_parameterised(genome, grammar, max_depth):
    """
    Lazy
    """
    
    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"(\<(\w+)\>|\<(\w+)\>\(\w+\))",phenotype).group()
    #n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    n_starting_NTs = len([term for term in re.findall(r"(\<([\(\)\w,-.]+)\>|\<([\(\)\w,-.]+)\>\(\w+\))",phenotype)])  
#    self.non_terminals = [term for term in re.findall(r"(\<[\(\)\w,-.]+\>|\<[\(\)\w,-.]+\>\(\w\))\s*::=",bnf_grammar)]
        
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < len(genome):
        parameterised_ = False
        NT_index = grammar.non_terminals.index(next_NT)
        if grammar.n_rules[NT_index] == 1: #there is a single PR for this non-terminal
            index_production_chosen = 0        
        else: #we consume one codon, and add the index to the structure
            index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
            structure.append(index_production_chosen)
            idx_genome += 1
            
        if grammar.parameterised_non_terminals[NT_index]:
            parameterised_ = True
        if parameterised_:
            next_NT = re.search(next_NT + r"\(\S*\)", phenotype).group()
            next_NT_parameterised = next_NT.split('>(')
            next_NT_parameterised[0] = next_NT_parameterised[0] + '>'
            next_NT_parameterised[1] = next_NT_parameterised[1][:-1] #remove the last parenthesis
            #next_NT_parameterised[1] is the current level (integer number)
            #grammar.parameterised_non_terminals[NT_index][1] is the parameter
            exec(grammar.parameterised_non_terminals[NT_index][1] + '=' + next_NT_parameterised[1])
            PR_replace_ = grammar.production_rules[NT_index][index_production_chosen][0]
            replace_levels_ = re.findall(r"\(\S*\)",PR_replace_)
            for i in range(len(replace_levels_)):
                level_ = eval(replace_levels_[i])
                PR_replace_ = PR_replace_.replace(replace_levels_[i], '(' + str(level_) + ')')
            phenotype = phenotype.replace(next_NT, PR_replace_, 1)
        #volta
        else:
            phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
            idx_depth += 1
            nodes += 1
        elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
            pass        
        else: #it is a PR with more than one NT
            arity = grammar.production_rules[NT_index][index_production_chosen][2]
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]

        next_ = re.search(r"\<(\w+)\>",phenotype)
        
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        #idx_genome += 1
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure

def mapper(genome, grammar, max_depth):
    
    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    #n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)])    
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < len(genome):
        NT_index = grammar.non_terminals.index(next_NT)
        index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
        structure.append(index_production_chosen)
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
            idx_depth += 1
            nodes += 1
        elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
            pass        
        else: #it is a PR with more than one NT
            arity = grammar.production_rules[NT_index][index_production_chosen][2]
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        idx_genome += 1
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure

def mapper_eager(genome, grammar, max_depth):
    """
    Identical to the previous one.
    TODO Solve the names later.
    """    

    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    #n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < len(genome):
        NT_index = grammar.non_terminals.index(next_NT)
        index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
        structure.append(index_production_chosen)
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
            idx_depth += 1
            nodes += 1
        elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
            pass        
        else: #it is a PR with more than one NT
            arity = grammar.production_rules[NT_index][index_production_chosen][2]
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        idx_genome += 1
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure

def mapper_lazy(genome, grammar, max_depth):
    """
    This mapper is similar to the previous one, but it does not consume codons
    when mapping a production rule with a single option."""
    
    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    #n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < len(genome):
        NT_index = grammar.non_terminals.index(next_NT)
        if grammar.n_rules[NT_index] == 1: #there is a single PR for this non-terminal
            index_production_chosen = 0        
        else: #we consume one codon, and add the index to the structure
            index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
            structure.append(index_production_chosen)
            idx_genome += 1
        
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
            idx_depth += 1
            nodes += 1
        elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
            pass        
        else: #it is a PR with more than one NT
            arity = grammar.production_rules[NT_index][index_production_chosen][2]
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
            
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure

def filter_options(grammar, list_codons, genome, idx_genome, NT_index):
    filtered_options = []
    #remaining = abs(idx_genome)
    #necessary = sum(list_codons)
    boundary = abs(idx_genome) - sum(list_codons)
    #return [PR for PR in grammar.production_rules[NT_index] if PR[6] <= boundary]
    for PR in grammar.production_rules[NT_index]:
        if PR[6] <= boundary:
            filtered_options.append(PR)
    return filtered_options

    #actual_options = filter_options(total_options, list_codons, genome, idx_genome)
    
def filter_options2(grammar, list_codons, genome, idx_genome, NT_index):
    #remaining = abs(idx_genome)
    #necessary = sum(list_codons)
    boundary = abs(idx_genome) - sum(list_codons)
    return [PR for PR in grammar.production_rules[NT_index] if PR[6] <= boundary]

def mapper_cosmo_total(genome, grammar, max_depth):
    """
    A eager cosmo mapping, which extends the elimination of choices to those ones deeper than max depth.
    This version maps with Cosmo all the time"""
    
    idx_genome = -len(genome)
    phenotype = grammar.start_rule
    next_NT = grammar.initial_next_NT
    list_depth = grammar.initial_list_depth.copy()
    list_codons = grammar.initial_list_codons.copy()
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < 0:
        changed_ = False
        
        NT_index = grammar.non_terminals.index(next_NT)
        
        boundary_ = abs(idx_genome) - sum(list_codons) + list_codons[idx_depth]
        boundary_depth_ = max_depth - list_depth[idx_depth]
        if grammar.max_codons_each_PR[NT_index] <= boundary_ and grammar.max_depth_each_PR[NT_index] <= boundary_depth_: #it goes normal
            index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
            Ch = grammar.production_rules[NT_index][index_production_chosen]
 #           print(list_codons, idx_genome, boundary_, grammar.n_rules[NT_index], phenotype)
        else: #we reduce the possible choices
            if grammar.max_codons_each_PR[NT_index] > boundary_:
                actual_options = [PR for PR in grammar.production_rules[NT_index] if PR[6] <= boundary_]
                changed_ = True
            if grammar.max_depth_each_PR[NT_index] > boundary_depth_:
                if changed_:
                    actual_options = [PR for PR in actual_options if PR[5] <= boundary_depth_]
                else:
                    actual_options = [PR for PR in grammar.production_rules[NT_index] if PR[5] <= boundary_depth_]

            try:
                index_production_chosen = genome[idx_genome] % len(actual_options)
            except(ZeroDivisionError):
    #            print("invalid cosmo children generated, length = ", len(genome), list_depth)
                break
            Ch = actual_options[index_production_chosen]
  #          print(list_codons, idx_genome, boundary_, len(actual_options), phenotype)

        structure.append(Ch[3])
        phenotype = phenotype.replace(next_NT, Ch[0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if Ch[2] == 0: #arity 0 (T)
            list_codons[idx_depth] = 0
            idx_depth += 1
            nodes += 1
        elif Ch[2] == 1: #arity 1 (PR with one NT)
            list_codons[idx_depth] = Ch[6] - 1 #-1 because one codon was already consumed to make the choice
        else: #it is a PR with more than one NT
            #to use with depth
            arity = Ch[2]
            #to use with codons
            extra_codons = [] #it keeps the minimum number of codons necessary to terminate the mapping of each branch
            for term in re.findall(r"\<([\(\)\w,-.]+)\>", Ch[0]):
                NT_index = grammar.non_terminals.index('<' + term + '>')
                extra_codons.append(grammar.min_codons_each_PR[NT_index]) 
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
                list_codons = extra_codons + list_codons[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
                list_codons = list_codons[0:idx_depth] + extra_codons + list_codons[idx_depth+1:]
                
        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        idx_genome += 1
 
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = len(genome) - abs(idx_genome)
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure, True

def mapper_cosmo_ext(genome, grammar, max_depth):
    """
    A eager cosmo mapping, which extends the elimination of choices to those ones deeper than max depth."""
    
    phenotype, nodes, depth, used_codons, invalid, n_wraps, structure = mapper_eager(genome, grammar, max_depth)
    
    if not invalid:
        return phenotype, nodes, depth, used_codons, invalid, n_wraps, structure, False
    
    idx_genome = -len(genome)
    phenotype = grammar.start_rule
    next_NT = grammar.initial_next_NT
    list_depth = grammar.initial_list_depth.copy()
    list_codons = grammar.initial_list_codons.copy()
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < 0:
        changed_ = False
        
        NT_index = grammar.non_terminals.index(next_NT)
        
        boundary_ = abs(idx_genome) - sum(list_codons) + list_codons[idx_depth]
        boundary_depth_ = max_depth - list_depth[idx_depth]
        if grammar.max_codons_each_PR[NT_index] <= boundary_ and grammar.max_depth_each_PR[NT_index] <= boundary_depth_: #it goes normal
            index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
            Ch = grammar.production_rules[NT_index][index_production_chosen]
 #           print(list_codons, idx_genome, boundary_, grammar.n_rules[NT_index], phenotype)
        else: #we reduce the possible choices
            if grammar.max_codons_each_PR[NT_index] > boundary_:
                actual_options = [PR for PR in grammar.production_rules[NT_index] if PR[6] <= boundary_]
                changed_ = True
            if grammar.max_depth_each_PR[NT_index] > boundary_depth_:
                if changed_:
                    actual_options = [PR for PR in actual_options if PR[5] <= boundary_depth_]
                else:
                    actual_options = [PR for PR in grammar.production_rules[NT_index] if PR[5] <= boundary_depth_]

            try:
                index_production_chosen = genome[idx_genome] % len(actual_options)
            except(ZeroDivisionError):
    #            print("invalid cosmo children generated, length = ", len(genome), list_depth)
                break
            Ch = actual_options[index_production_chosen]
  #          print(list_codons, idx_genome, boundary_, len(actual_options), phenotype)

        structure.append(Ch[3])
        phenotype = phenotype.replace(next_NT, Ch[0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if Ch[2] == 0: #arity 0 (T)
            list_codons[idx_depth] = 0
            idx_depth += 1
            nodes += 1
        elif Ch[2] == 1: #arity 1 (PR with one NT)
            list_codons[idx_depth] = Ch[6] - 1 #-1 because one codon was already consumed to make the choice
        else: #it is a PR with more than one NT
            #to use with depth
            arity = Ch[2]
            #to use with codons
            extra_codons = [] #it keeps the minimum number of codons necessary to terminate the mapping of each branch
            for term in re.findall(r"\<([\(\)\w,-.]+)\>", Ch[0]):
                NT_index = grammar.non_terminals.index('<' + term + '>')
                extra_codons.append(grammar.min_codons_each_PR[NT_index]) 
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
                list_codons = extra_codons + list_codons[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
                list_codons = list_codons[0:idx_depth] + extra_codons + list_codons[idx_depth+1:]
                
        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        idx_genome += 1
 
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = len(genome) - abs(idx_genome)
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure, True

def mapper_cosmo(genome, grammar, max_depth):
    """
    Fifth implementation of the eager cosmo mapping."""
    
    phenotype, nodes, depth, used_codons, invalid, n_wraps, structure = mapper_eager(genome, grammar, max_depth)
    
    if not invalid:
        return phenotype, nodes, depth, used_codons, invalid, n_wraps, structure, False
    
    idx_genome = -len(genome)
    phenotype = grammar.start_rule
    next_NT = grammar.initial_next_NT
    list_depth = grammar.initial_list_depth.copy()
    list_codons = grammar.initial_list_codons.copy()
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < 0:
        
        NT_index = grammar.non_terminals.index(next_NT)
        
        #console 12
        boundary_ = abs(idx_genome) - sum(list_codons) + list_codons[idx_depth]
        if grammar.max_codons_each_PR[NT_index] <= boundary_: #it goes normal
            index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
            Ch = grammar.production_rules[NT_index][index_production_chosen]
 #           print(list_codons, idx_genome, boundary_, grammar.n_rules[NT_index], phenotype)
        else: #we reduce the possible choices
            
            actual_options = [PR for PR in grammar.production_rules[NT_index] if PR[6] <= boundary_]

            try:
                index_production_chosen = genome[idx_genome] % len(actual_options)
            except(ZeroDivisionError):
    #            print("invalid cosmo children generated, length = ", len(genome), list_depth)
                break
            Ch = actual_options[index_production_chosen]
  #          print(list_codons, idx_genome, boundary_, len(actual_options), phenotype)

        structure.append(Ch[3])
        phenotype = phenotype.replace(next_NT, Ch[0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if Ch[2] == 0: #arity 0 (T)
            list_codons[idx_depth] = 0
            idx_depth += 1
            nodes += 1
        elif Ch[2] == 1: #arity 1 (PR with one NT)
            list_codons[idx_depth] = Ch[6] - 1 #-1 because one codon was already consumed to make the choice
        else: #it is a PR with more than one NT
            #to use with depth
            arity = Ch[2]
            #to use with codons
            extra_codons = [] #it keeps the minimum number of codons necessary to terminate the mapping of each branch
            for term in re.findall(r"\<([\(\)\w,-.]+)\>", Ch[0]):
                NT_index = grammar.non_terminals.index('<' + term + '>')
                extra_codons.append(grammar.min_codons_each_PR[NT_index]) 
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
                list_codons = extra_codons + list_codons[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
                list_codons = list_codons[0:idx_depth] + extra_codons + list_codons[idx_depth+1:]
                
        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        idx_genome += 1
 
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = len(genome) - abs(idx_genome)
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure, True

def mapper_cosmo4(genome, grammar, max_depth):
    """
    Fourth implementation of the eager cosmo mapping."""
    
    phenotype, nodes, depth, used_codons, invalid, n_wraps, structure = mapper_eager(genome, grammar, max_depth)
    
    if not invalid:
        return phenotype, nodes, depth, used_codons, invalid, n_wraps, structure, False
    
    idx_genome = -len(genome)
    phenotype = copy.deepcopy(grammar.start_rule)
    next_NT = copy.deepcopy(grammar.initial_next_NT)
    list_depth = copy.deepcopy(grammar.initial_list_depth)
    list_codons = copy.deepcopy(grammar.initial_list_codons)
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < 0:
        
        NT_index = grammar.non_terminals.index(next_NT)
        
        #console 12
        boundary_ = abs(idx_genome) - sum(list_codons)
        if grammar.max_codons_each_PR[NT_index] <= boundary_: #it goes normal
            index_production_chosen = genome[idx_genome] % grammar.n_rules[NT_index]
            Ch = grammar.production_rules[NT_index][index_production_chosen]
 #           print(list_codons, idx_genome, boundary_, grammar.n_rules[NT_index], phenotype)
        else: #we reduce the possible choices
            
            actual_options = [PR for PR in grammar.production_rules[NT_index] if PR[6] <= boundary_]

            try:
                index_production_chosen = genome[idx_genome] % len(actual_options)
            except(ZeroDivisionError):
                break
            Ch = actual_options[index_production_chosen]
 #           print(list_codons, idx_genome, boundary_, len(actual_options), phenotype)

        structure.append(Ch[3])
        phenotype = phenotype.replace(next_NT, Ch[0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if Ch[2] == 0: #arity 0 (T)
            list_codons[idx_depth] = 0
            idx_depth += 1
            nodes += 1
        elif Ch[2] == 1: #arity 1 (PR with one NT)
            list_codons[idx_depth] = Ch[6] - 1 #-1 because one codon was already consumed to make the choice
        else: #it is a PR with more than one NT
            #to use with depth
            arity = Ch[2]
            #to use with codons
            extra_codons = [] #it keeps the minimum number of codons necessary to terminate the mapping of each branch
            for term in re.findall(r"\<([\(\)\w,-.]+)\>", Ch[0]):
                NT_index = grammar.non_terminals.index('<' + term + '>')
                minimum_n_codons = []
                for PR in grammar.production_rules[NT_index]:
                    minimum_n_codons.append(PR[6])
                extra_codons.append(min(minimum_n_codons)) 
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
                list_codons = extra_codons + list_codons[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
                list_codons = list_codons[0:idx_depth] + extra_codons + list_codons[idx_depth+1:]
                

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        idx_genome += 1
 
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = len(genome) - abs(idx_genome)
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure, True

def mapper_cosmo3(genome, grammar, max_depth):
    """
    Third implementation of the eager cosmo mapping."""
    
    phenotype, nodes, depth, used_codons, invalid, n_wraps, structure = mapper_eager(genome, grammar, max_depth)
    
#    if not invalid:
#        return phenotype, nodes, depth, used_codons, invalid, n_wraps, structure    
    
    idx_genome = -len(genome)
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    #n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    list_codons = [] #it keeps the minimum number of codons necessary to terminate the mapping of each branch
    for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype):
        NT_index = grammar.non_terminals.index('<' + term + '>')
        minimum_n_codons = []
        for PR in grammar.production_rules[NT_index]:
            minimum_n_codons.append(PR[6])
        list_codons.append(min(minimum_n_codons))
    idx_depth = 0
    nodes = 0
    structure = []
    
    while next_NT and idx_genome < 0:
        NT_index = grammar.non_terminals.index(next_NT)
        
        #console 8
        #boundary_ = abs(idx_genome) - sum(list_codons)
        #actual_options = [PR for PR in grammar.production_rules[NT_index] if PR[6] <= boundary_]#filter_options(grammar, list_codons, genome, idx_genome, NT_index)#[PR for PR in grammar.production_rules[NT_index] if (PR[6] + sum(list_codons)) <= abs(idx_genome)]
        
        #console 9
       # actual_options = filter_options(grammar, list_codons, genome, idx_genome, NT_index)#[PR for PR in grammar.production_rules[NT_index] if (PR[6] + sum(list_codons)) <= abs(idx_genome)]
       
        #console 10
        #actual_options = filter_options2(grammar, list_codons, genome, idx_genome, NT_index)#[PR for PR in grammar.production_rules[NT_index] if (PR[6] + sum(list_codons)) <= abs(idx_genome)]
        
        #console 11
        actual_options = [PR for PR in grammar.production_rules[NT_index] if (PR[6] + sum(list_codons)) <= abs(idx_genome)]
       
        try:
            index_production_chosen = genome[idx_genome] % len(actual_options)
        except(ZeroDivisionError):
            break
        Ch = actual_options[index_production_chosen]

        structure.append(Ch[3])
        phenotype = phenotype.replace(next_NT, Ch[0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if Ch[2] == 0: #arity 0 (T)
            list_codons[idx_depth] = 0
            idx_depth += 1
            nodes += 1
        elif Ch[2] == 1: #arity 1 (PR with one NT)
            list_codons[idx_depth] = Ch[6]
        else: #it is a PR with more than one NT
            #to use with depth
            arity = Ch[2]
            #to use with codons
            extra_codons = [] #it keeps the minimum number of codons necessary to terminate the mapping of each branch
            for term in re.findall(r"\<([\(\)\w,-.]+)\>", Ch[0]):
                NT_index = grammar.non_terminals.index('<' + term + '>')
                minimum_n_codons = []
                for PR in grammar.production_rules[NT_index]:
                    minimum_n_codons.append(PR[6])
                extra_codons.append(min(minimum_n_codons))
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
                list_codons = extra_codons + list_codons[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
                list_codons = list_codons[0:idx_depth] + extra_codons + list_codons[idx_depth+1:]
                

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        idx_genome += 1
 
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = len(genome) - abs(idx_genome)
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure

def mapper_cosmo2(genome, grammar, max_depth):
    """
    New attempt."""
    
    phenotype, nodes, depth, used_codons, invalid, n_wraps, structure = mapper_eager(genome, grammar, max_depth)
    
    if not invalid:
        return phenotype, nodes, depth, used_codons, invalid, n_wraps, structure
    
    genome_length = len(genome)
    
    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    #n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    list_codons = [] #it keeps the minimum number of codons necessary to terminate the mapping of each branch
    for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype):
        NT_index = grammar.non_terminals.index('<' + term + '>')
        minimum_n_codons = []
        for PR in grammar.production_rules[NT_index]:
            minimum_n_codons.append(PR[6])
        list_codons.append(min(minimum_n_codons)) #TODO
    idx_depth = 0
    nodes = 0
    structure = []
    
    codons_check = []
    respective_rules = []
    remove_ = []
    count_ = 0
    production_rules = copy.deepcopy(grammar.production_rules)
    non_terminals = copy.deepcopy(grammar.non_terminals)
    idx_NTs = [[i] for i in range(len(non_terminals))]
    for i in range(len(non_terminals)):
        idx_NTs[i].append(non_terminals[i])
    
    for i in range(len(production_rules)):
        for j in range(len(production_rules[i])):
            if not (production_rules[i][j][6] + sum(list_codons) - list_codons[idx_depth]) <= (genome_length - idx_genome):
                remove_.append([i,j])
            else:
                #if production_rules[i][j][6] > 1:
                codons_check.append(production_rules[i][j][6])
                respective_rules.append([i, j])
            count_ += 1
    #del production_rules[i][j]
    for ele in sorted(remove_, reverse = True):
        for j in range(len(production_rules[ele[0]]) - ele[1] - 1):
            respective_rules[ele[0] + j][1] -= 1
        del production_rules[ele[0]][ele[1]]
        
    codons_check_set = list(set(codons_check))
    codons_check_set = sorted(codons_check_set, reverse=True) #Use the reverse sorting to stop checking earlier
    
    #print(respective_rules)
            
    #total_options = [PR for PR in grammar.production_rules if (PR[6] + sum(list_codons) - list_codons[idx_depth]) <= (len(genome) - idx_genome + 1)]
    
    while next_NT and idx_genome < genome_length:
        try:
            NT_index = non_terminals.index(next_NT)
        except(ValueError):
            break

        #total_options = [PR for PR in grammar.production_rules[NT_index]]
        
        #actual_options = [PR for PR in total_options if (PR[6] + sum(list_codons) - list_codons[idx_depth]) <= (len(genome) - idx_genome + 1)]
        
        #actual_options = production_rules[NT_index]

     #   if not actual_options:
     #       break
        
        try: 
            index_production_chosen = genome[idx_genome] % len(production_rules[NT_index])
        except(ZeroDivisionError):
            break
        Ch = production_rules[NT_index][index_production_chosen]

        structure.append(Ch[3])
        phenotype = phenotype.replace(next_NT, Ch[0], 1)
        list_depth[idx_depth] += 1
        #list_codons[idx_depth] -= 1
        if list_depth[idx_depth] > max_depth:
            break
        if Ch[2] == 0: #arity 0 (T)
            list_codons[idx_depth] = 0
            idx_depth += 1
            nodes += 1
        elif Ch[2] == 1: #arity 1 (PR with one NT)
            list_codons[idx_depth] = Ch[6]
        else: #it is a PR with more than one NT
            #to use with depth
            arity = Ch[2]
            #to use with codons
            extra_codons = [] #it keeps the minimum number of codons necessary to terminate the mapping of each branch
            for term in re.findall(r"\<([\(\)\w,-.]+)\>", Ch[0]):
                NT_index = grammar.non_terminals.index('<' + term + '>')
                minimum_n_codons = []
                for PR in production_rules[NT_index]:
                    minimum_n_codons.append(PR[6])
                extra_codons.append(min(minimum_n_codons))
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
                list_codons = extra_codons + list_codons[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
                list_codons = list_codons[0:idx_depth] + extra_codons + list_codons[idx_depth+1:]
                

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        idx_genome += 1
        
        if next_NT and idx_genome < genome_length:
            remove_ = []
            remove2_ = []
            for i in range(len(codons_check_set)): #This list is sorted reversely. Then, when one passes the check, we don't need to continue
       #         print(codons_check_set, i)
                if codons_check_set[i] != 1 and (not (codons_check_set[i] + sum(list_codons)) <= (genome_length - idx_genome)): #+1 because idx_genome was already incremented
                    check = True
                    while check:
                        try:
                            index_rule = codons_check.index(codons_check_set[i])
                        except(ValueError):#, IndexError): #element is not in list
                            break #when there is no more rules with that quantity of codons in codons_check_set, it'll stop
                            
                        try:
                            del production_rules[respective_rules[index_rule][0]][respective_rules[index_rule][1]]
           #                 if len(production_rules[respective_rules[index_rule][0]]) == 0:
           #                     idx_ = idx_NTs[respective_rules[index_rule][0]]
           #                     idx_remove_ = non_terminals.index(idx_[1])
           #                     del non_terminals[idx_remove_]
           #                     del idx_NTs[respective_rules[index_rule][0]]
           #                     del production_rules[respective_rules[index_rule][0]]
                        except(IndexError):
                            raise ValueError("Error when removing PRs")
    
                        if len(production_rules[respective_rules[index_rule][0]]) == 0:
                            
                            idx_ = idx_NTs[respective_rules[index_rule][0]]
                            idx_remove_ = non_terminals.index(idx_[1])
                            del non_terminals[idx_remove_]
                            del idx_NTs[respective_rules[index_rule][0]]
                            del production_rules[respective_rules[index_rule][0]]
                            
                            #TODO check if this was correctly removed in 18-3
    #                        for j in range(len(respective_rules[index_rule:])):
    #                            if respective_rules[index_rule + j]:
    #                                respective_rules[index_rule + j][0] -= 1
                        else:
                            for j in range(len(production_rules[respective_rules[index_rule][0]])+1-respective_rules[index_rule][1]):
                                if respective_rules[index_rule + j]:
                                    respective_rules[index_rule + j][1] -= 1
                            
                        
                            
                        for j in range(len(production_rules)):
                            if len(production_rules[j]) == 0:
                                raise ValueError("Number of PRs is zero")
                        
                        respective_rules[index_rule] = None
                        codons_check[index_rule] = None
                        
                        for cd in respective_rules:
                            if cd is not None:
                                if cd[0] < 0 or cd[1] < 0:
                                    raise ValueError("Error when decrementing indexes")
                    codons_check_set[i] = None
                else:
                    break
      #      print("finished")
            codons_check_set = [cd for cd in codons_check_set if cd is not None]
            
                        #del production_rules[respective_rules[index_rule]]
                    
                    
                    
               #         del respective_rules[index_rule]
                    
            #except(IndexError):
            #    pass
#        for ele in sorted(remove_, reverse = True):
            #try:
#            del production_rules[ele[0]][ele[1]]
            #except(IndexError):
            #    pass
#        for ele in sorted(remove2_, reverse = True):
#            del respective_rules[ele]
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure

def mapper_cosmo1(genome, grammar, max_depth):
    """
    This mapper is similar to the eager approach."""
    
    phenotype, nodes, depth, used_codons, invalid, n_wraps, structure = mapper_eager(genome, grammar, max_depth)
    
    if not invalid:
        return phenotype, nodes, depth, used_codons, invalid, n_wraps, structure    
    
    idx_genome = 0
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    #n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    list_codons = [] #it keeps the minimum number of codons necessary to terminate the mapping of each branch
    for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype):
        NT_index = grammar.non_terminals.index('<' + term + '>')
        minimum_n_codons = []
        for PR in grammar.production_rules[NT_index]:
            minimum_n_codons.append(PR[6])
        list_codons.append(min(minimum_n_codons))
    idx_depth = 0
    nodes = 0
    structure = []
    
    NT_index = grammar.non_terminals.index(next_NT)
    total_options = [PR for PR in grammar.production_rules[NT_index]]
    actual_options = [PR for PR in total_options if (PR[6] + sum(list_codons) - list_codons[idx_depth]) <= len(genome)]
    
    while next_NT and idx_genome < len(genome):
        

        
        
     #   if not actual_options:
     #       break
        
        try:
            index_production_chosen = genome[idx_genome] % len(actual_options)
        except(ZeroDivisionError):
            break
        Ch = actual_options[index_production_chosen]

        structure.append(Ch[3])
        phenotype = phenotype.replace(next_NT, Ch[0], 1)
        list_depth[idx_depth] += 1
        #list_codons[idx_depth] -= 1
        if list_depth[idx_depth] > max_depth:
            break
        if Ch[2] == 0: #arity 0 (T)
            list_codons[idx_depth] = 0
            idx_depth += 1
            nodes += 1
        elif Ch[2] == 1: #arity 1 (PR with one NT)
            list_codons[idx_depth] = Ch[6]
        else: #it is a PR with more than one NT
            #to use with depth
            arity = Ch[2]
            #to use with codons
            extra_codons = [] #it keeps the minimum number of codons necessary to terminate the mapping of each branch
            for term in re.findall(r"\<([\(\)\w,-.]+)\>", Ch[0]):
                NT_index = grammar.non_terminals.index('<' + term + '>')
                minimum_n_codons = []
                for PR in grammar.production_rules[NT_index]:
                    minimum_n_codons.append(PR[6])
                extra_codons.append(min(minimum_n_codons))
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
                list_codons = extra_codons + list_codons[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
                list_codons = list_codons[0:idx_depth] + extra_codons + list_codons[idx_depth+1:]
                

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
        else:
            next_NT = None
        idx_genome += 1
        
        if next_NT and idx_genome < len(genome):
            NT_index = grammar.non_terminals.index(next_NT)
            total_options = [PR for PR in grammar.production_rules[NT_index]]
            actual_options = [PR for PR in total_options if (PR[6] + sum(list_codons)) <= (len(genome) - idx_genome)]
        
    if next_NT:
        invalid = True
        used_codons = 0
    else:
        invalid = False
        used_codons = idx_genome
    
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure

def mapper_multichromosomal(genome, grammar, max_depth):
    """
    Mapper for Multi-chromosomal Grammatical Evolution (hara2008).
    It uses the lazy approach.
    Genome should be a list of lists, each with an independent genotype"""
    
    n_genomes = len(genome)
    n_rules_check = 0
    idx_mapped_PRs = []
    for n_rules in grammar.n_rules:
        if n_rules > 1:
            #Filling up the idx of the genome considering the PRs with at least two choices
            idx_mapped_PRs.append(n_rules_check)
            n_rules_check += 1
        else:
            idx_mapped_PRs.append(None)
    if n_genomes != n_rules_check:
        raise ValueError("The number of genomes is different of the number of PRs with at least two choices")
        
    #idx_genome = [0, 0] #1st refers to the genome and 2nd to the codon being mapped
    idx_genome = [0]*n_genomes #idx refers to the genome and value refers to the position of the codon being mapped
    used_codons = np.zeros([n_genomes,], dtype=int)
    #phenotype_list = []
    #nodes_list = []
    #depth_list = []
    #used_codons_list = []
    #invalid_list = []
    #structure_list = []
    
    phenotype = grammar.start_rule
    next_NT = re.search(r"\<(\w+)\>",phenotype).group()
    #n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
    n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)])
    list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
    idx_depth = 0
    nodes = 0
    structure = []
    
    idx_available = True    
    NT_index = grammar.non_terminals.index(next_NT)
    while next_NT and idx_available:
        idx = idx_mapped_PRs[NT_index] #idx of the genome currently being mapped
        if grammar.n_rules[NT_index] == 1: #there is a single PR for this non-terminal
            index_production_chosen = 0        
        else: #we consume one codon, and add the index to the structure
 #           print("genome:", idx)
 #           print("codon position:", idx_genome[idx])
            index_production_chosen = genome[idx][idx_genome[idx]] % grammar.n_rules[NT_index]
            structure.append(index_production_chosen)
            used_codons[idx] += 1
            idx_genome[idx] += 1
        
        phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
        list_depth[idx_depth] += 1
        if list_depth[idx_depth] > max_depth:
            break
        if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
            idx_depth += 1
            nodes += 1
        elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
            pass        
        else: #it is a PR with more than one NT
            arity = grammar.production_rules[NT_index][index_production_chosen][2]
            if idx_depth == 0:
                list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
            else:
                list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]

        next_ = re.search(r"\<(\w+)\>",phenotype)
        if next_:
            next_NT = next_.group()
            NT_index = grammar.non_terminals.index(next_NT)
            idx_available = idx_genome[idx_mapped_PRs[NT_index]] < len(genome[idx_mapped_PRs[NT_index]])    
        else:
            next_NT = None
        
    if next_NT:
        invalid = True
        used_codons = np.zeros([n_genomes,], dtype=int)
    else:
        invalid = False
        #used_codons = 0
        #for codons in idx_genome:
        #    used_codons += codons
            
    depth = max(list_depth)
   
    return phenotype, nodes, depth, used_codons, invalid, 0, structure

def mapper_multi(genome, grammar, max_depth):
    """
    Mapper for multi GE used for FPTs (multiple outputs).
    It uses the lazy approach.
    Genome should be a list of lists, each with an independent genotype"""
    
    n_genomes = len(genome)
    phenotype_list = []
    nodes_list = []
    depth_list = []
    used_codons_list = []
    invalid_list = []
    structure_list = []
    
    for i in range(n_genomes):
        idx_genome = 0
        phenotype = grammar.start_rule
        next_NT = re.search(r"\<(\w+)\>",phenotype).group()
        #n_starting_NTs = len([term for term in re.findall(r"\<(\w+)\>",phenotype)])
        n_starting_NTs = len([term for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)])
        list_depth = [1]*n_starting_NTs #it keeps the depth of each branch
        idx_depth = 0
        nodes = 0
        structure = []
        
        while next_NT and idx_genome < len(genome[i]):
            NT_index = grammar.non_terminals.index(next_NT)
            if grammar.n_rules[NT_index] == 1: #there is a single PR for this non-terminal
                index_production_chosen = 0        
            else: #we consume one codon, and add the index to the structure
                index_production_chosen = genome[i][idx_genome] % grammar.n_rules[NT_index]
                structure.append(index_production_chosen)
                idx_genome += 1
            
            phenotype = phenotype.replace(next_NT, grammar.production_rules[NT_index][index_production_chosen][0], 1)
            list_depth[idx_depth] += 1
            if list_depth[idx_depth] > max_depth:
                break
            if grammar.production_rules[NT_index][index_production_chosen][2] == 0: #arity 0 (T)
                idx_depth += 1
                nodes += 1
            elif grammar.production_rules[NT_index][index_production_chosen][2] == 1: #arity 1 (PR with one NT)
                pass        
            else: #it is a PR with more than one NT
                arity = grammar.production_rules[NT_index][index_production_chosen][2]
                if idx_depth == 0:
                    list_depth = [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
                else:
                    list_depth = list_depth[0:idx_depth] + [list_depth[idx_depth],]*arity + list_depth[idx_depth+1:]
    
            next_ = re.search(r"\<(\w+)\>",phenotype)
            if next_:
                next_NT = next_.group()
            else:
                next_NT = None
                
            
        if next_NT:
            invalid = True
            used_codons = 0
        else:
            invalid = False
            used_codons = idx_genome
        
        depth = max(list_depth)
        
        phenotype_list.append(phenotype)
        nodes_list.append(nodes)
        depth_list.append(depth)
        used_codons_list.append(used_codons)
        invalid_list.append(invalid)
        structure_list.append(structure)
        
        invalid = True if True in invalid_list else False
   
    return phenotype_list, nodes_list, depth_list, used_codons_list, invalid, 0, structure_list

def random_initialisation(ind_class, pop_size, bnf_grammar, min_init_genome_length, max_init_genome_length, max_init_depth, codon_size, codon_consumption, genome_representation):
        """
        
        """
        population = []
        
        for i in range(pop_size):
            genome = []
            init_genome_length = random.randint(min_init_genome_length, max_init_genome_length)
            for j in range(init_genome_length):
                genome.append(random.randint(0, codon_size))
            ind = ind_class(genome, bnf_grammar, max_init_depth, codon_consumption)
            population.append(ind)
    
        if genome_representation == 'list':
            return population
        else:
            raise ValueError("Unkonwn genome representation")

def sensible_initialisation_multiGE(ind_class, pop_size, bnf_grammar, min_init_depth, 
                            max_init_depth, codon_size, codon_consumption,
                            genome_representation, n_genomes):
        """
        Lazy
        """
        if codon_consumption != 'multiGE':
            raise ValueError("wrong codon_consumption")
        else:
            #Calculate the number of individuals to be generated with each method
            is_odd = pop_size % 2
            n_grow = int(pop_size/2)
            
            n_sets_grow = max_init_depth - min_init_depth + 1
            set_size = int(n_grow/n_sets_grow)
            remaining = n_grow % n_sets_grow
            
            n_full = n_grow + is_odd + remaining #if pop_size is odd, generate an extra ind with "full"
            
            #TODO check if it is possible to generate inds with max_init_depth
            
            population = []
            #Generate inds using "Grow"
            for i in range(n_sets_grow):
                max_init_depth_ = min_init_depth + i
                for j in range(set_size):
                    genome_list = []
                    remainders_list = []
                    phenotype_list = []
                    depth_list = []
                    
                    for g in range(n_genomes):                    
                        remainders = [] #it will register the choices
                        possible_choices = [] #it will register the respective possible choices
                        
                        phenotype = bnf_grammar.start_rule
                        #remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)] #
                        remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)] #
                        depths = [1]*len(remaining_NTs) #it keeps the depth of each branch
                        idx_branch = 0 #index of the current branch being grown
                        while len(remaining_NTs) != 0:
                            idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[0])
                            total_options = [PR for PR in bnf_grammar.production_rules[idx_NT]]
                            actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if PR[5] + depths[idx_branch] <= max_init_depth_]
                            Ch = random.choice(actual_options)
                            phenotype = phenotype.replace(remaining_NTs[0], Ch[0], 1)
                            depths[idx_branch] += 1
                            if codon_consumption == 'eager':
                                remainders.append(Ch[3])
                                possible_choices.append(len(total_options))
                            elif codon_consumption == 'lazy' or codon_consumption == 'multiGE':
                                if len(total_options) > 1:
                                    remainders.append(Ch[3])
                                    possible_choices.append(len(total_options))
                                    
                            if Ch[2] > 1:
                                if idx_branch == 0:
                                    depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                                else:
                                    depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                            if Ch[1] == 'terminal':
                                idx_branch += 1
                            
                            remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)]
                        
                        #Generate the genome
                        genome = []
                        if codon_consumption == 'eager' or codon_consumption == 'lazy' or codon_consumption == 'multiGE':
                            for k in range(len(remainders)):
                                codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[k])) * possible_choices[k]) + remainders[k]
                                genome.append(codon)
                        else:
                            raise ValueError("Unknown mapper")
                            
                        #Include a tail with 50% of the genome's size
                        size_tail = max(int(0.5*len(genome)), 1) #Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.
                        for j in range(size_tail):
                            genome.append(random.randint(0,codon_size))
                            
                        genome_list.append(genome)
                        remainders_list.append(remainders)
                        phenotype_list.append(phenotype)
                        depth_list.append(max(depths))
                            
                    #Initialise the individual and include in the population
                    ind = ind_class(genome_list, bnf_grammar, max_init_depth_, codon_consumption)
                    
                    #Check if the individual was mapped correctly
                    if remainders_list != ind.structure or phenotype_list != ind.phenotype or depth_list != ind.depth:
                        raise Exception('error in the mapping')
                        
                    population.append(ind)    
                
            for i in range(n_full):
                genome_list = []
                remainders_list = []
                phenotype_list = []
                depth_list = []
                    
                for g in range(n_genomes):    
                    remainders = [] #it will register the choices
                    possible_choices = [] #it will register the respective possible choices
        
                    phenotype = bnf_grammar.start_rule
                    #remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)] #
                    remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)] #
                    depths = [1]*len(remaining_NTs) #it keeps the depth of each branch
                    idx_branch = 0 #index of the current branch being grown
        
                    while len(remaining_NTs) != 0:
                        idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[0])
                        total_options = [PR for PR in bnf_grammar.production_rules[idx_NT]]
                        actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if PR[5] + depths[idx_branch] <= max_init_depth]
                        recursive_options = [PR for PR in actual_options if PR[4]]
                        if len(recursive_options) > 0:
                            Ch = random.choice(recursive_options)
                        else:
                            Ch = random.choice(actual_options)
                        phenotype = phenotype.replace(remaining_NTs[0], Ch[0], 1)
                        depths[idx_branch] += 1
                        if codon_consumption == 'eager':
                            remainders.append(Ch[3])
                            possible_choices.append(len(total_options))
                        elif codon_consumption == 'lazy' or codon_consumption == 'multiGE':
                            if len(total_options) > 1:
                                remainders.append(Ch[3])
                                possible_choices.append(len(total_options))
                                
                        if Ch[2] > 1:
                            if idx_branch == 0:
                                depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                            else:
                                depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                        if Ch[1] == 'terminal':
                            idx_branch += 1
                        
                        remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)]
                    
                    #Generate the genome
                    genome = []
                    if codon_consumption == 'eager' or codon_consumption == 'lazy' or codon_consumption == 'multiGE':
                    	for j in range(len(remainders)):
                    		codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[j])) * possible_choices[j]) + remainders[j]
                    		genome.append(codon)
                    else:
                    	raise ValueError("Unknown mapper")
        
                    #Include a tail with 50% of the genome's size
                    if codon_consumption == 'eager' or codon_consumption == 'lazy' or codon_consumption == 'multiGE':
                        size_tail = max(int(0.5*len(genome)), 1) #Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.
                    
                    for j in range(size_tail):
                        genome.append(random.randint(0,codon_size))
                        
                    genome_list.append(genome)
                    remainders_list.append(remainders)
                    phenotype_list.append(phenotype)
                    depth_list.append(max(depths))
                        
                #Initialise the individual and include in the population
                ind = ind_class(genome_list, bnf_grammar, max_init_depth, codon_consumption)
                
                #Check if the individual was mapped correctly
                if remainders_list != ind.structure or phenotype_list != ind.phenotype or depth_list != ind.depth:
                    raise Exception('error in the mapping')
                        
                population.append(ind)    
        
            if genome_representation == 'list':
                return population
            else:
                raise ValueError("Unkonwn genome representation")
            
def sensible_initialisation(ind_class, pop_size, bnf_grammar, min_init_depth, 
                            max_init_depth, codon_size, codon_consumption,
                            genome_representation):
        """
        
        """
        
        if codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
            tile_size = 0
            tile_idx = [] #Index of each grammar.production_rules in the tile
            tile_n_rules = [] #Number of choices (PRs) for each position of the tile
            for i in range(len(bnf_grammar.production_rules)):
                if len(bnf_grammar.production_rules[i]) == 1: #The PR has a single option
                    tile_idx.append(False)
                else:
                    tile_idx.append(tile_size)
                    tile_n_rules.append(len(bnf_grammar.production_rules[i]))
                    tile_size += 1            
        
        #Calculate the number of individuals to be generated with each method
        is_odd = pop_size % 2
        n_grow = int(pop_size/2)
        
        n_sets_grow = max_init_depth - min_init_depth + 1
        set_size = int(n_grow/n_sets_grow)
        remaining = n_grow % n_sets_grow
        
        n_full = n_grow + is_odd + remaining #if pop_size is odd, generate an extra ind with "full"
        
        #TODO check if it is possible to generate inds with max_init_depth
        
        population = []
        #Generate inds using "Grow"
        for i in range(n_sets_grow):
            max_init_depth_ = min_init_depth + i
            for j in range(set_size):
                remainders = [] #it will register the choices
                possible_choices = [] #it will register the respective possible choices
                if codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                    PR_used_idx = [] #it will register the respective index of the PRs being used
    
                phenotype = bnf_grammar.start_rule
#                    remaining_NTs = [term for term in re.findall(r"(\<([\(\)\w,-.]+)\>|\<([\(\)\w,-.]+)\>\(\w+\))",phenotype)]
                #remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)] #
                remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)] #
                depths = [1]*len(remaining_NTs) #it keeps the depth of each branch
                idx_branch = 0 #index of the current branch being grown
                while len(remaining_NTs) != 0:
                    parameterised_ = False
                    idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[0])
                    total_options = [PR for PR in bnf_grammar.production_rules[idx_NT]]
                    actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if PR[5] + depths[idx_branch] <= max_init_depth_]
                    Ch = random.choice(actual_options)
                    if codon_consumption == 'parameterised':#voltasen
                        if bnf_grammar.parameterised_non_terminals[idx_NT]:
                            parameterised_ = True
                    if parameterised_:
                        next_NT = re.search(remaining_NTs[0] + r"\(\S*\)", phenotype).group()
                        next_NT_parameterised = next_NT.split('>(')
                        next_NT_parameterised[0] = next_NT_parameterised[0] + '>'
                        next_NT_parameterised[1] = next_NT_parameterised[1][:-1] #remove the last parenthesis
                        #next_NT_parameterised[1] is the current level (integer number)
                        #grammar.parameterised_non_terminals[NT_index][1] is the parameter
                        exec(bnf_grammar.parameterised_non_terminals[idx_NT][1] + '=' + next_NT_parameterised[1])
                        PR_replace_ = Ch[0]
                        replace_levels_ = re.findall(r"\(\S*\)",PR_replace_)
                        for i in range(len(replace_levels_)):
                            level_ = eval(replace_levels_[i])
                            PR_replace_ = PR_replace_.replace(replace_levels_[i], '(' + str(level_) + ')')
                        phenotype = phenotype.replace(next_NT, PR_replace_, 1)
                    else:
                        phenotype = phenotype.replace(remaining_NTs[0], Ch[0], 1)
                    depths[idx_branch] += 1
                    if codon_consumption == 'eager' or codon_consumption == 'cosmo_eager' or codon_consumption == 'cosmo_eager_depth':
                        remainders.append(Ch[3])
                        possible_choices.append(len(total_options))
                    elif codon_consumption == 'lazy' or codon_consumption == 'parameterised' or codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                        if len(total_options) > 1:
                            remainders.append(Ch[3])
                            possible_choices.append(len(total_options))
                            if codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                                PR_used_idx.append(idx_NT)                       
                    
                    if Ch[2] > 1:
                        if idx_branch == 0:
                            depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                        else:
                            depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                    if Ch[1] == 'terminal':
                        idx_branch += 1
                    
                    #remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)]
                    remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)]
                
                #Generate the genome
                genome = []
                if codon_consumption == 'eager' or codon_consumption == 'lazy' or codon_consumption == 'parameterised' or codon_consumption == 'cosmo_eager' or codon_consumption == 'cosmo_eager_depth':
                    for k in range(len(remainders)):
                        codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[k])) * possible_choices[k]) + remainders[k]
                        genome.append(codon)
                elif codon_consumption == 'leap':
                    for k in range(len(remainders)):
                        codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[k])) * possible_choices[k]) + remainders[k]
                        for l in range(tile_size):
                            if l == tile_idx[PR_used_idx[k]]:#volta
                                genome.append(codon)
                            else:
                                genome.append(random.randint(0, codon_size))
                elif codon_consumption == 'leap2' or codon_consumption == 'leap3':
                    #Firstly we need to know how many tiles we will have
                    tile_map = [[False]*tile_size]
                    n_tiles = 0
                    order_map_inside_tile = 1 #The first one to be used will receive the order 1
                    for k in range(len(remainders)):
                        for l in range(tile_size):
                            if l == tile_idx[PR_used_idx[k]]: 
                                if tile_map[n_tiles][l]: #If we already used this position, we open the next tile
                                    order_map_inside_tile = 1 #To check how position is being mapped firstly inside the tile
                                    n_tiles += 1
                                    tile_map.append([False]*tile_size)
                                    tile_map[n_tiles][l] = order_map_inside_tile
                                    order_map_inside_tile += 1
                                else: #If not, we keep in the same tile
                                    tile_map[n_tiles][l] = order_map_inside_tile
                                    order_map_inside_tile += 1
                    #Now we know how the tiles are distributed, so we can map
                    positions_used_each_tile = []
                    for k in range(len(tile_map)):
                        positions = 0
                        for l in range(tile_size):
                            if tile_map[k][l]:
                                positions += 1
                        positions_used_each_tile.append(positions)    
                        
                    id_mapping = 0
                    
                    for k in range(len(tile_map)):
                        for l in range(tile_size):
                            if tile_map[k][l]:
                                if codon_consumption == 'leap2':
                                    codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[id_mapping+tile_map[k][l]-1])) * possible_choices[id_mapping+tile_map[k][l]-1]) + remainders[id_mapping+tile_map[k][l]-1]
                                elif codon_consumption == 'leap3':
                                    codon = remainders[id_mapping+tile_map[k][l]-1]#possible_choices[id_mapping+tile_map[k][l]-1]                                    
                                genome.append(codon)
                                #id_mapping += 1
                            else:
                                if codon_consumption == 'leap2':
                                    genome.append(random.randint(0, codon_size))
                                elif codon_consumption == 'leap3':
                                    genome.append(random.randint(0, tile_n_rules[l]))
#                                print(genome)
                        id_mapping += positions_used_each_tile[k]
                    
#                    order_map_inside_tile = 0
#                    for k in range(len(remainders)):
#                        codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[k])) * possible_choices[k]) + remainders[k]
#                        for l in range(tile_size):
#                            if tile_map[n_tiles][l]: #If we already used this position, we open the next tile
#                                if tile_map[n_tiles][l] == order_map_inside_tile    
#                            n_tiles += 1
#                                tile_map.append([False]*tile_size)
#                                tile_map[n_tiles][l] == True
#                            else: #If not, we keep in the same tile
#                                if l == tile_idx[PR_used_idx[k]]: 
#                                    tile_map[n_tiles][l] == True
                else:
                    raise ValueError("Unknown mapper")
                    
                #Include a tail with 50% of the genome's size
                size_tail = max(int(0.5*len(genome)), 1) #Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.
                for k in range(size_tail):
                    genome.append(random.randint(0,codon_size))
                    
                #Initialise the individual and include in the population
                ind = ind_class(genome, bnf_grammar, max_init_depth_, codon_consumption)
                
   #             if ind.depth > 5:
   #                 print(ind.phenotype)
   #                 print(ind.depth)
   #                 input("enter")
   #             print(phenotype)
   #             print(remainders)
   #             print(ind.structure)
   #             print(ind.invalid)
#                input()
                
                #Check if the individual was mapped correctly
                if remainders != ind.structure or phenotype != ind.phenotype or max(depths) != ind.depth:
                    raise Exception('error in the mapping')
                    
                population.append(ind)    

            
        for i in range(n_full):
            remainders = [] #it will register the choices
            possible_choices = [] #it will register the respective possible choices
            if codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                PR_used_idx = [] #it will register the respective index of the PRs being used

            phenotype = bnf_grammar.start_rule
            #remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)] #
            remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)] #
            depths = [1]*len(remaining_NTs) #it keeps the depth of each branch
            idx_branch = 0 #index of the current branch being grown

            while len(remaining_NTs) != 0:
                parameterised_ = False
                idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[0])
                total_options = [PR for PR in bnf_grammar.production_rules[idx_NT]]
                actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if PR[5] + depths[idx_branch] <= max_init_depth]
                recursive_options = [PR for PR in actual_options if PR[4]]
                if len(recursive_options) > 0:
                    Ch = random.choice(recursive_options)
                else:
                    Ch = random.choice(actual_options)
                if codon_consumption == 'parameterised':#voltasen
                    if bnf_grammar.parameterised_non_terminals[idx_NT]:
                        parameterised_ = True
                if parameterised_:
                    next_NT = re.search(remaining_NTs[0] + r"\(\S*\)", phenotype).group()
                    next_NT_parameterised = next_NT.split('>(')
                    next_NT_parameterised[0] = next_NT_parameterised[0] + '>'
                    next_NT_parameterised[1] = next_NT_parameterised[1][:-1] #remove the last parenthesis
                    #next_NT_parameterised[1] is the current level (integer number)
                    #grammar.parameterised_non_terminals[NT_index][1] is the parameter
                    exec(bnf_grammar.parameterised_non_terminals[idx_NT][1] + '=' + next_NT_parameterised[1])
                    #PR_replace_ = bnf_grammar.production_rules[idx_NT][Ch[0]][0]
                    PR_replace_ = Ch[0]
                    replace_levels_ = re.findall(r"\(\S*\)",PR_replace_)
                    for i in range(len(replace_levels_)):
                        level_ = eval(replace_levels_[i])
                        PR_replace_ = PR_replace_.replace(replace_levels_[i], '(' + str(level_) + ')')
                    phenotype = phenotype.replace(next_NT, PR_replace_, 1)
                else:
                    phenotype = phenotype.replace(remaining_NTs[0], Ch[0], 1)
                depths[idx_branch] += 1
                if codon_consumption == 'eager' or codon_consumption == 'cosmo_eager' or codon_consumption == 'cosmo_eager_depth':
                    remainders.append(Ch[3])
                    possible_choices.append(len(total_options))
                elif codon_consumption == 'lazy' or codon_consumption == 'parameterised' or codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                    if len(total_options) > 1:
                        remainders.append(Ch[3])
                        possible_choices.append(len(total_options))
                        if codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                            PR_used_idx.append(idx_NT)       
                
                if Ch[2] > 1:
                    if idx_branch == 0:
                        depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                    else:
                        depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                if Ch[1] == 'terminal':
                    idx_branch += 1
                
                #remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)]
                remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)]
            
            #Generate the genome
            genome = []
            if codon_consumption == 'eager' or codon_consumption == 'lazy' or codon_consumption == 'parameterised' or codon_consumption == 'cosmo_eager' or codon_consumption == 'cosmo_eager_depth':
            	for j in range(len(remainders)):
            		codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[j])) * possible_choices[j]) + remainders[j]
            		genome.append(codon)
            elif codon_consumption == 'leap':
            	for j in range(len(remainders)):
            		codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[j])) * possible_choices[j]) + remainders[j]
            		for k in range(tile_size):
            			if k == tile_idx[PR_used_idx[j]]:
            				genome.append(codon)
            			else:
            				genome.append(random.randint(0, codon_size))
            elif codon_consumption == 'leap2' or codon_consumption == 'leap3':
                #Firstly we need to know how many tiles we will have
                tile_map = [[False]*tile_size]
                n_tiles = 0
                order_map_inside_tile = 1 #The first one to be used will receive the order 1
                for k in range(len(remainders)):
                    for l in range(tile_size):
                        if l == tile_idx[PR_used_idx[k]]: 
                            if tile_map[n_tiles][l]: #If we already used this position, we open the next tile
                                order_map_inside_tile = 1 #To check how position is being mapped firstly inside the tile
                                n_tiles += 1
                                tile_map.append([False]*tile_size)
                                tile_map[n_tiles][l] = order_map_inside_tile
                                order_map_inside_tile += 1
                            else: #If not, we keep in the same tile
                                tile_map[n_tiles][l] = order_map_inside_tile
                                order_map_inside_tile += 1
                #Now we know how the tiles are distributed, so we can map
                positions_used_each_tile = []
                for k in range(len(tile_map)):
                    positions = 0
                    for l in range(tile_size):
                        if tile_map[k][l]:
                            positions += 1
                    positions_used_each_tile.append(positions)    
                    
                id_mapping = 0
                
                for k in range(len(tile_map)):
                    for l in range(tile_size):
                        if tile_map[k][l]:
                            if codon_consumption == 'leap2':
                                codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[id_mapping+tile_map[k][l]-1])) * possible_choices[id_mapping+tile_map[k][l]-1]) + remainders[id_mapping+tile_map[k][l]-1]
                            elif codon_consumption == 'leap3':
                                codon = remainders[id_mapping+tile_map[k][l]-1]#possible_choices[id_mapping+tile_map[k][l]-1]
                            genome.append(codon)
                            #id_mapping += 1
                        else:
                            if codon_consumption == 'leap2':
                                genome.append(random.randint(0, codon_size))
                            elif codon_consumption == 'leap3':
                                genome.append(random.randint(0, tile_n_rules[l]))
                    id_mapping += positions_used_each_tile[k]
            else:
            	raise ValueError("Unknown mapper")

            #Include a tail with 50% of the genome's size
            if codon_consumption == 'eager' or codon_consumption == 'lazy' or codon_consumption == 'parameterised' or codon_consumption == 'cosmo_eager' or codon_consumption == 'cosmo_eager_depth':
                size_tail = max(int(0.5*len(genome)), 1) #Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.
            elif codon_consumption == 'leap':
                raise ValueError("This mapping process was not implemented for this initialisation method")
            elif codon_consumption == 'leap2':
                n_tiles_tail = max(int(0.5*n_tiles), 1)
                size_tail = n_tiles_tail * tile_size
            elif codon_consumption == 'leap3':
                raise ValueError("This mapping process was not implemented for this initialisation method")
            
            for j in range(size_tail):
                genome.append(random.randint(0,codon_size))
                
            #Initialise the individual and include in the population
            ind = ind_class(genome, bnf_grammar, max_init_depth, codon_consumption)
            
            #Check if the individual was mapped correctly
            if remainders != ind.structure or phenotype != ind.phenotype or max(depths) != ind.depth:
                raise Exception('error in the mapping')

            population.append(ind)    
    
        if genome_representation == 'list':
            return population
        elif genome_representation == 'numpy':
            for ind in population:
                ind.genome = np.array(ind.genome)
            return population
        else:
            raise ValueError("Unkonwn genome representation")

def sensible_initialisation_multichromosomal(ind_class, pop_size, bnf_grammar, min_init_depth, 
                            max_init_depth, codon_size, codon_consumption,
                            genome_representation):
        """
        Only for codon_consumption = 'lazy' and genome_representation = 'list'
        """
        
        if genome_representation != 'list':
            raise ValueError("Unkonwn genome representation")
        if codon_consumption != 'multichromosomalGE':
            raise ValueError("Unkonwn codon consumption")
            
        n_rules_check = 0
        idx_mapped_PRs = []        
        for n_rules in bnf_grammar.n_rules:
            if n_rules > 1:
                #Filling up the idx of the genome considering the PRs with at least two choices
                idx_mapped_PRs.append(n_rules_check)
                n_rules_check += 1
            else:
                idx_mapped_PRs.append(None)

        #Calculate the number of individuals to be generated with each method
        is_odd = pop_size % 2
        n_grow = int(pop_size/2)
        
        n_sets_grow = max_init_depth - min_init_depth + 1
        set_size = int(n_grow/n_sets_grow)
        remaining = n_grow % n_sets_grow
        
        n_full = n_grow + is_odd + remaining #if pop_size is odd, generate an extra ind with "full"
        
        population = []
        #Generate inds using "Grow"
        for i in range(n_sets_grow):
            max_init_depth_ = min_init_depth + i
            for j in range(set_size):
                if j == 2:
                    pass
                genome = [[] for _ in range(n_rules_check)]
                remainders = [] #it will register the choices
                possible_choices = [] #it will register the respective possible choices
    
                phenotype = bnf_grammar.start_rule
                remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)] #
                depths = [1]*len(remaining_NTs) #it keeps the depth of each branch
                idx_branch = 0 #index of the current branch being grown
                while len(remaining_NTs) != 0:
                    idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[0])
                    total_options = [PR for PR in bnf_grammar.production_rules[idx_NT]]
                    actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if PR[5] + depths[idx_branch] <= max_init_depth_]
                    Ch = random.choice(actual_options)
                    phenotype = phenotype.replace(remaining_NTs[0], Ch[0], 1)
                    depths[idx_branch] += 1
                    if len(total_options) > 1:
                        current_remainder = Ch[3]
                        remainders.append(current_remainder)
                        current_possible_choices = len(total_options)
                        possible_choices.append(current_possible_choices)
                        
                        #Generate the genome
                        idx_genome = idx_mapped_PRs[idx_NT] #idx of the genome which refers to the PR being used                        
                        codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / current_possible_choices)) * current_possible_choices) + current_remainder
                        genome[idx_genome].append(codon)
                        
                            
                    if Ch[2] > 1:
                        if idx_branch == 0:
                            depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                        else:
                            depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                    if Ch[1] == 'terminal':
                        idx_branch += 1
                    
                    remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)]
                    
                #Include a tail with 50% of the genome's size
                for k in range(n_rules_check): #number of genomes
                    #size_tail = max(int(0.5*len(genome[k])), 1) #Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.
                    size_tail = max(int(0.5*len(genome[k])), 1) #For MCGE, tail must have at least two codons, considering the cases where there is no codons in a genome, which would disturb de crossover point
                    for l in range(size_tail):
                        genome[k].append(random.randint(0,codon_size))
                    
                #Initialise the individual and include in the population
                ind = ind_class(genome, bnf_grammar, max_init_depth_, codon_consumption)
                
                #Check if the individual was mapped correctly
                if remainders != ind.structure or phenotype != ind.phenotype or max(depths) != ind.depth:
                    raise Exception('error in the mapping')

                population.append(ind)    

            
        for i in range(n_full):
            genome = [[] for _ in range(n_rules_check)]
            remainders = [] #it will register the choices
            possible_choices = [] #it will register the respective possible choices

            phenotype = bnf_grammar.start_rule
            remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)] #
            depths = [1]*len(remaining_NTs) #it keeps the depth of each branch
            idx_branch = 0 #index of the current branch being grown

            while len(remaining_NTs) != 0:
                idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[0])
                total_options = [PR for PR in bnf_grammar.production_rules[idx_NT]]
                actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if PR[5] + depths[idx_branch] <= max_init_depth]
                recursive_options = [PR for PR in actual_options if PR[4]]
                if len(recursive_options) > 0:
                    Ch = random.choice(recursive_options)
                else:
                    Ch = random.choice(actual_options)
                phenotype = phenotype.replace(remaining_NTs[0], Ch[0], 1)
                depths[idx_branch] += 1
                if len(total_options) > 1:
                    current_remainder = Ch[3]
                    remainders.append(current_remainder)
                    current_possible_choices = len(total_options)
                    possible_choices.append(current_possible_choices)
                    
                    #Generate the genome
                    idx_genome = idx_mapped_PRs[idx_NT] #idx of the genome which refers to the PR being used                        
                    codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / current_possible_choices)) * current_possible_choices) + current_remainder
                    genome[idx_genome].append(codon)
                
                if Ch[2] > 1:
                    if idx_branch == 0:
                        depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                    else:
                        depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                if Ch[1] == 'terminal':
                    idx_branch += 1
                
                remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)]

            #Include a tail with 50% of the genome's size
            #size_tail = max(int(0.5*len(genome)), 1) #Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.
            #size_tail = max(int(0.5*len(genome)), 2) #For MCGE, tail must have at least two codons, considering the cases where there is no codons in a genome, which would disturb de crossover point
            
            #Include a tail with 50% of the genome's size
            for j in range(n_rules_check): #number of genomes
                size_tail = max(int(0.5*len(genome[j])), 1) #Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.
                for k in range(size_tail):
                    genome[j].append(random.randint(0,codon_size))
                
            #Initialise the individual and include in the population
            ind = ind_class(genome, bnf_grammar, max_init_depth, codon_consumption)
            
            #Check if the individual was mapped correctly
            if remainders != ind.structure or phenotype != ind.phenotype or max(depths) != ind.depth:
                raise Exception('error in the mapping')

            population.append(ind)    
    
        return population
    
def PI_Grow(ind_class, pop_size, bnf_grammar, min_init_depth, max_init_depth, 
            codon_size, codon_consumption,
            genome_representation):
    
    count = 0
    
    #Calculate the number of individuals to be generated with each depth
    n_sets = max_init_depth - min_init_depth + 1 
    set_size = int(pop_size/n_sets)
    remaining = pop_size % n_sets #the size of the last set, to be initialised with a random max depth between min_init_depth and max_init_depth
    n_sets += 1 #including the last set, which will have random init depth
    
    #TODO check if it is possible to generate inds with max_init_depth and min_init_depth
    
    if codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
        tile_size = 0
        tile_idx = [] #Index of each grammar.production_rules in the tile
        tile_n_rules = [] #Number of choices (PRs) for each position of the tile
        for i in range(len(bnf_grammar.production_rules)):
            if len(bnf_grammar.production_rules[i]) == 1: #The PR has a single option
                tile_idx.append(False)
            else:
                tile_idx.append(tile_size)
                tile_n_rules.append(len(bnf_grammar.production_rules[i]))
                tile_size += 1     

    population = []
    for i in range(n_sets):
        if i == n_sets - 1:
            max_init_depth_ = random.randint(min_init_depth, max_init_depth + 1)
            set_size = remaining
        else:
            max_init_depth_ = min_init_depth + i

        for j in range(set_size):
            
            if count >= 7111:
                pass
            
            remainders = [] #it will register the choices
            possible_choices = [] #it will register the respective possible choices
            if codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                PR_used_idx = [] #it will register the respective index of the PRs being used
    
            phenotype = bnf_grammar.start_rule
            #remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)]
            remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)]
            list_phenotype = remaining_NTs #it keeps in each position a terminal if the respective branch was terminated or a <NT> otherwise
            depths = [1]*len(remaining_NTs) #it keeps the depth of each branch
            branches = [False]*len(remaining_NTs) #False if the respective branch was terminated. True, otherwise
            n_expansions = [0]*len(remaining_NTs) #Number of expansions used in each branch
            while len(remaining_NTs) != 0:
                choose_ = False
                if max(depths) < max_init_depth_:
                    #Count the number of NT with recursive options remaining
                    NT_with_recursive_options = 0
                    l = 0
                    for k in range(len(branches)):
                        if not branches[k]:
                            idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[l])
                            actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if PR[5] + depths[k] <= max_init_depth_]
                            recursive_options = [PR for PR in actual_options if PR[4]]
                            if len(recursive_options) > 0:
                                NT_with_recursive_options += 1
                                idx_recursive = idx_NT
                                idx_branch = k
                                PI_index = l
                            if NT_with_recursive_options == 2:
                                break
                            l += 1
                    if NT_with_recursive_options == 1: #if there is just one NT with recursive options remaining, choose between them
                        total_options = [PR for PR in bnf_grammar.production_rules[idx_recursive]]
                        recursive_options = [PR for PR in bnf_grammar.production_rules[idx_recursive] if PR[5] + depths[idx_branch] <= max_init_depth_ and PR[4]]
                        Ch = random.choice(recursive_options)
                        n_similar_NTs = remaining_NTs[:PI_index+1].count(remaining_NTs[PI_index])
                    
                        phenotype = replace_nth(phenotype, remaining_NTs[PI_index], Ch[0], n_similar_NTs)
                        
                        #new_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>", Ch[0])]
                        new_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>", Ch[0])]
                        list_phenotype = list_phenotype[0:idx_branch] + new_NTs + list_phenotype[idx_branch+1:]
                        
                        if remainders == []:
                            if codon_consumption == 'eager':
                                remainders.append(Ch[3])
                                possible_choices.append(len(total_options))
                            elif codon_consumption == 'lazy' or codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                                if len(total_options) > 1:
                                    remainders.append(Ch[3])
                                    possible_choices.append(len(total_options))
                                    if codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                                        #PR_used_idx.append(idx_NT) 
                                        PR_used_idx.append(idx_recursive) 
                        else:
                            if codon_consumption == 'eager':
                                remainder_position = sum(n_expansions[:idx_branch+1])
                                remainders = remainders[:remainder_position] + [Ch[3]] + remainders[remainder_position:]
                                possible_choices = possible_choices[:remainder_position] + [len(total_options)] + possible_choices[remainder_position:]
                            elif codon_consumption == 'lazy' or codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                                if len(total_options) > 1:
                                    remainder_position = sum(n_expansions[:idx_branch+1])
                                    remainders = remainders[:remainder_position] + [Ch[3]] + remainders[remainder_position:]
                                    possible_choices = possible_choices[:remainder_position] + [len(total_options)] + possible_choices[remainder_position:]    
                                    if codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                                        #PR_used_idx.append(idx_NT)
                                        #PR_used_idx = PR_used_idx[:remainder_position] + [idx_NT] + PR_used_idx[remainder_position:]
                                        PR_used_idx = PR_used_idx[:remainder_position] + [idx_recursive] + PR_used_idx[remainder_position:]
                        depths[idx_branch] += 1
                    #    n_expansions[idx_branch] += 1
                        
                        #We don't expand the branch for lazy and leap approaches, if there is a single option in the choices
                        if codon_consumption == 'eager':
                            n_expansions[idx_branch] += 1
                        elif codon_consumption == 'lazy' or codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                            if len(total_options) > 1:
                                n_expansions[idx_branch] += 1

                        if Ch[2] > 1:
                            if idx_branch == 0:
                                depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                            else:
                                depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                        if Ch[1] == 'terminal':
                            branches[idx_branch] = True
                        else:
                            branches = branches[0:idx_branch] + [False,]*Ch[2] + branches[idx_branch+1:]
                            n_expansions = n_expansions[0:idx_branch+1] + [0,]*(Ch[2]-1) + n_expansions[idx_branch+1:]
                            
                        #remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)]
                        remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)]
                    
                    else: #choose within any other branch
                        choose_ = True
                else:
                    choose_ = True
                            
                if choose_: #at least one branch has reached the max depth
                    PI_index = random.choice(range(len(remaining_NTs))) #index of the current branch being grown
                    count_ = 0
                    for k in range(len(branches)):
                        if not branches[k]:
                            if count_ == PI_index:
                                idx_branch = k
                                break
                            count_ += 1
                    
                    idx_NT = bnf_grammar.non_terminals.index(remaining_NTs[PI_index])
                    total_options = [PR for PR in bnf_grammar.production_rules[idx_NT]]
                    actual_options = [PR for PR in bnf_grammar.production_rules[idx_NT] if PR[5] + depths[idx_branch] <= max_init_depth_]
    
                    Ch = random.choice(actual_options)
                    n_similar_NTs = remaining_NTs[:PI_index+1].count(remaining_NTs[PI_index])
                    phenotype = replace_nth(phenotype, remaining_NTs[PI_index], Ch[0], n_similar_NTs)

                    #new_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>", Ch[0])]
                    new_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>", Ch[0])]
                    list_phenotype = list_phenotype[0:idx_branch] + new_NTs + list_phenotype[idx_branch+1:]
                    
                    if remainders == []:
                        if codon_consumption == 'eager':
                            remainders.append(Ch[3])
                            possible_choices.append(len(total_options))
                        elif codon_consumption == 'lazy' or codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                            if len(total_options) > 1:
                                remainders.append(Ch[3])
                                possible_choices.append(len(total_options))
                                if codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                                    PR_used_idx.append(idx_NT)    
                    else:
                        if codon_consumption == 'eager':
                            remainder_position = sum(n_expansions[:idx_branch+1])
                            remainders = remainders[:remainder_position] + [Ch[3]] + remainders[remainder_position:]
                            possible_choices = possible_choices[:remainder_position] + [len(total_options)] + possible_choices[remainder_position:]
                        elif codon_consumption == 'lazy' or codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                            if len(total_options) > 1:
                                remainder_position = sum(n_expansions[:idx_branch+1])
                                remainders = remainders[:remainder_position] + [Ch[3]] + remainders[remainder_position:]
                                possible_choices = possible_choices[:remainder_position] + [len(total_options)] + possible_choices[remainder_position:]
                                if codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                                    #PR_used_idx.append(idx_NT)
                                    PR_used_idx = PR_used_idx[:remainder_position] + [idx_NT] + PR_used_idx[remainder_position:]
                    depths[idx_branch] += 1
                    
                    #We don't expand the branch for lazy and leap approaches, if there is a single option in the choices
                    if codon_consumption == 'eager':
                        n_expansions[idx_branch] += 1
                    elif codon_consumption == 'lazy' or codon_consumption == 'leap' or codon_consumption == 'leap2' or codon_consumption == 'leap3':
                        if len(total_options) > 1:
                            n_expansions[idx_branch] += 1
                            
                    if Ch[2] > 1:
                        if idx_branch == 0:
                            depths = [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                        else:
                            depths = depths[0:idx_branch] + [depths[idx_branch],]*Ch[2] + depths[idx_branch+1:]
                    if Ch[1] == 'terminal':
                        branches[idx_branch] = True
                    else:
                        branches = branches[0:idx_branch] + [False,]*Ch[2] + branches[idx_branch+1:]
                        n_expansions = n_expansions[0:idx_branch+1] + [0,]*(Ch[2]-1) + n_expansions[idx_branch+1:]
                        
                    #remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<(\w+)\>",phenotype)]
                    remaining_NTs = ['<' + term + '>' for term in re.findall(r"\<([\(\)\w,-.]+)\>",phenotype)]
            
            #Generate the genome
            genome = []
            if codon_consumption == 'eager' or codon_consumption == 'lazy':
                for k in range(len(remainders)):
                    codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[k])) * possible_choices[k]) + remainders[k]
                    genome.append(codon)
            elif codon_consumption == 'leap':
                raise ValueError("This mapping process was not implemented for this initialisation method")
            elif codon_consumption == 'leap2' or codon_consumption == 'leap3':
                #Firstly we need to know how many tiles we will have
                tile_map = [[False]*tile_size]
                n_tiles = 0
                order_map_inside_tile = 1 #The first one to be used will receive the order 1
                for k in range(len(remainders)):
                    for l in range(tile_size):
                        if l == tile_idx[PR_used_idx[k]]: 
                            if tile_map[n_tiles][l]: #If we already used this position, we open the next tile
                                order_map_inside_tile = 1 #To check how position is being mapped firstly inside the tile
                                n_tiles += 1
                                tile_map.append([False]*tile_size)
                                tile_map[n_tiles][l] = order_map_inside_tile
                                order_map_inside_tile += 1
                            else: #If not, we keep in the same tile
                                tile_map[n_tiles][l] = order_map_inside_tile
                                order_map_inside_tile += 1
                #Now we know how the tiles are distributed, so we can map
                positions_used_each_tile = []
                for k in range(len(tile_map)):
                    positions = 0
                    for l in range(tile_size):
                        if tile_map[k][l]:
                            positions += 1
                    positions_used_each_tile.append(positions)    
                    
                id_mapping = 0
                
                for k in range(len(tile_map)):
                    for l in range(tile_size):
                        if tile_map[k][l]:
                            if codon_consumption == 'leap2':
                                codon = (random.randint(0,1e10) % math.floor(((codon_size + 1) / possible_choices[id_mapping+tile_map[k][l]-1])) * possible_choices[id_mapping+tile_map[k][l]-1]) + remainders[id_mapping+tile_map[k][l]-1]
                            elif codon_consumption == 'leap3':
                                codon = remainders[id_mapping+tile_map[k][l]-1]#possible_choices[id_mapping+tile_map[k][l]-1]                                    
                            genome.append(codon)
                        else:
                            if codon_consumption == 'leap2':
                                genome.append(random.randint(0, codon_size))
                            elif codon_consumption == 'leap3':
                                genome.append(random.randint(0, tile_n_rules[l]))
                    id_mapping += positions_used_each_tile[k]
            else:
                raise ValueError("Unknown mapper")
            
            #Include a tail with 50% of the genome's size
            if codon_consumption == 'eager' or codon_consumption == 'lazy':
                size_tail = max(int(0.5*len(genome)), 1) #Tail must have at least one codon. Otherwise, in the lazy approach, when we have the last PR with just a single option, the mapping procces will not terminate.
            elif codon_consumption == 'leap':
                raise ValueError("This mapping process was not implemented for this initialisation method")
            elif codon_consumption == 'leap2':
                n_tiles_tail = max(int(0.5*n_tiles), 1)
                size_tail = n_tiles_tail * tile_size
            elif codon_consumption == 'leap3':
                raise ValueError("This mapping process was not implemented for this initialisation method")
            
            for j in range(size_tail):
                genome.append(random.randint(0,codon_size))
                
            #Initialise the individual and include in the population
            ind = ind_class(genome, bnf_grammar, max_init_depth_, codon_consumption)
            
#            print(phenotype)
#            print(ind.phenotype)
#            print(remainders)
#            print(ind.structure)
#            print(ind.invalid)
#            print(genome)
#            print(ind.genome)
#            print(possible_choices)
#            print(count)
            
            count += 1
            


        
            #Check if the individual was mapped correctly
            if remainders != ind.structure or phenotype != ind.phenotype or max(depths) != ind.depth:
                raise Exception('error in the mapping')
                
            population.append(ind)   
            
    if genome_representation == 'list':
        return population
    elif genome_representation == 'numpy':
        for ind in population:
            ind.genome = np.array(ind.genome)
        return population
    else:
        raise ValueError("Unkonwn genome representation")

    return population

def crossover_onepoint(parent0, parent1, bnf_grammar, max_depth, codon_consumption, 
                       invalidate_max_depth,
                       genome_representation='list', max_genome_length=None):
    """
    
    """
    if max_genome_length:
        raise ValueError("max_genome_length not implemented in this onepoint")
    
    if parent0.invalid: #used_codons = 0
        possible_crossover_codons0 = len(parent0.genome)
    else:
        possible_crossover_codons0 = min(len(parent0.genome), parent0.used_codons) #in case of wrapping, used_codons can be greater than genome's length
    if parent1.invalid:
        possible_crossover_codons1 = len(parent1.genome)
    else:
        possible_crossover_codons1 = min(len(parent1.genome), parent1.used_codons)
#        print()
    
    parent0_genome = parent0.genome.copy()
    parent1_genome = parent1.genome.copy()
    continue_ = True
#    a = 0
    while continue_:
        #Set points for crossover within the effective part of the genomes
        point0 = random.randint(1, possible_crossover_codons0)
        point1 = random.randint(1, possible_crossover_codons1)
        
        if genome_representation == 'list':
            #Operate crossover
            new_genome0 = parent0_genome[0:point0] + parent1_genome[point1:]
            new_genome1 = parent1_genome[0:point1] + parent0_genome[point0:]
        else:
            raise ValueError("Only 'list' representation is implemented")
        
        new_ind0 = reMap(parent0, new_genome0, bnf_grammar, max_depth, codon_consumption)
        new_ind1 = reMap(parent1, new_genome1, bnf_grammar, max_depth, codon_consumption)
        
        if invalidate_max_depth: # In the mapping, if a ind surpasses max depth, it is invalid, and we won't redo crossover
            continue_ = False
        else: # We check if a ind surpasses max depth, and if so we will redo crossover
            continue_ = new_ind0.depth > max_depth or new_ind1.depth > max_depth

        

        
#        if not check_:
#            print()
#            print("checking")
#            print("parent0:")
#            print("length = ", len(parent0_genome))
#            print("used codons = ", parent0.used_codons)
#            print("invalid = ", parent0.invalid)
#            print("cut point = ", point0)
#            print("parent1:")
#            print("length = ", len(parent1_genome))
#            print("used codons = ", parent1.used_codons)
#            print("invalid = ", parent1.invalid)
#            print("cut point = ", point1)
#            check = True
            
        #if len(new_genome0) == 1 or len(new_genome1) == 1:
            #print(continue_)
#            if continue_:
#                print()
#                print("parent0:")
#                print("length = ", len(parent0_genome))
#                print("used codons = ", parent0.used_codons)
#                print("invalid = ", parent0.invalid)
#                print("cut point = ", point0)
#                print("parent1:")
#                print("length = ", len(parent1_genome))
#                print("used codons = ", parent1.used_codons)
#                print("invalid = ", parent1.invalid)
#                print("cut point = ", point1)
#                check_ = False
                
#        if not continue_:
#            if len(new_genome0) == 1 or len(new_genome1) == 1:
#                print()
#                print("stopped")
#                print("parent0:")
#                print("length = ", len(parent0_genome))
#                print("used codons = ", parent0.used_codons)
#                print("invalid = ", parent0.invalid)
#                print("cut point = ", point0)
#                print("parent1:")
#                print("length = ", len(parent1_genome))
#                print("used codons = ", parent1.used_codons)
#                print("invalid = ", parent1.invalid)
#                print("cut point = ", point1)
                      
        
    del new_ind0.fitness.values, new_ind1.fitness.values
    return new_ind0, new_ind1   
    
def crossover_onepoint2(parent0, parent1, bnf_grammar, max_depth, codon_consumption, 
                       genome_representation, max_genome_length):
    """
    
    """
    if parent0.invalid: #used_codons = 0
        possible_crossover_codons0 = len(parent0.genome)
    else:
        possible_crossover_codons0 = min(len(parent0.genome), parent0.used_codons) #in case of wrapping, used_codons can be greater than genome's length
    if parent1.invalid:
        possible_crossover_codons1 = len(parent1.genome)
    else:
        possible_crossover_codons1 = min(len(parent1.genome), parent1.used_codons)
#        print()
    
    continue_ = True
    check_ = True
#    a = 0
    while continue_:
        #Set points for crossover within the effective part of the genomes
        point0 = random.randint(1, possible_crossover_codons0)
        point1 = random.randint(1, possible_crossover_codons1)
        
        if genome_representation == 'numpy':
            #TODO This operations is not working in case of wrapping
            len0 = point0 + (len(parent1.genome) - point1)
            len1 = point1 + (len(parent0.genome) - point0)
            new_genome0 = np.zeros([len0], dtype=int)
            new_genome1 = np.zeros([len1], dtype=int)
  #          print("length", len(parent0.genome), len(parent1.genome))
  #          print("possible crossover codons", possible_crossover_codons0, possible_crossover_codons1)
  #          print("crossover points", point0, point1)
  #          print("new lengths", len0, len1)
            #Operate crossover
            new_genome0[0:point0] = parent0.genome[0:point0]
            new_genome0[point0:] = parent1.genome[point1:]
            new_genome1[0:point1] = parent1.genome[0:point1]
            new_genome1[point1:] = parent0.genome[point0:]
        elif genome_representation == 'list':
            #Operate crossover
            new_genome0 = parent0.genome[0:point0] + parent1.genome[point1:]
            new_genome1 = parent1.genome[0:point1] + parent0.genome[point0:]
        else:
            raise ValueError("Unknown genome representation")
        
        new_ind0 = reMap(parent0, new_genome0, bnf_grammar, max_depth, codon_consumption)
        new_ind1 = reMap(parent1, new_genome1, bnf_grammar, max_depth, codon_consumption)
        
        continue_ = new_ind0.depth > max_depth or new_ind1.depth > max_depth
        
        if not check_:
            print()
            print("checking")
            print("parent0:")
            print("length = ", len(parent0.genome))
            print("used codons = ", parent0.used_codons)
            print("invalid = ", parent0.invalid)
            print("cut point = ", point0)
            print("parent1:")
            print("length = ", len(parent1.genome))
            print("used codons = ", parent1.used_codons)
            print("invalid = ", parent1.invalid)
            print("cut point = ", point1)
            check = True
            
        if len(new_genome0) == 1 or len(new_genome1) == 1:
            if continue_:
                print()
                print("parent0:")
                print("length = ", len(parent0.genome))
                print("used codons = ", parent0.used_codons)
                print("invalid = ", parent0.invalid)
                print("cut point = ", point0)
                print("parent1:")
                print("length = ", len(parent1.genome))
                print("used codons = ", parent1.used_codons)
                print("invalid = ", parent1.invalid)
                print("cut point = ", point1)
                check_ = False
                
        if not continue_:
            if len(new_genome0) == 1 or len(new_genome1) == 1:
                print()
                print("stopped")
                print("parent0:")
                print("length = ", len(parent0.genome))
                print("used codons = ", parent0.used_codons)
                print("invalid = ", parent0.invalid)
                print("cut point = ", point0)
                print("parent1:")
                print("length = ", len(parent1.genome))
                print("used codons = ", parent1.used_codons)
                print("invalid = ", parent1.invalid)
                print("cut point = ", point1)
                      


        
 #       if continue_:
  #          a += 1
   #         print(a)
    if max_genome_length:
        if new_ind0.depth > max_depth or len(new_ind0.genome) > max_genome_length:
            return0 = parent0
        else:
            return0 = new_ind0
        if new_ind1.depth > max_depth or len(new_ind1.genome) > max_genome_length:
            return1 = parent1
        else:
            return1 = new_ind1
    else:
        if new_ind0.depth > max_depth:
            #print("error")
            return0 = parent0
        else:
            return0 = new_ind0
        if new_ind1.depth > max_depth:
            #print("error")
            return1 = parent1
        else:
            return1 = new_ind1
        
    return return0, return1   


def crossover_onepoint_multiGE_One(parent0, parent1, bnf_grammar, max_depth, codon_consumption, 
                       genome_representation, max_genome_length):
    """
    
    """
    
    if len(parent0.genome) != len(parent1.genome):
        raise ValueError("Wrong codon consumption")
    elif genome_representation != 'list':
        raise ValueError("Multi GE is implemented only using list as genome representation")
    else:
        n_genomes = len(parent0.genome)
    
    #Identify effective genomes (which are actually being used in the mapping process)
    effective_genomes0 = []
    effective_genomes1 = []
    for i in range(n_genomes):
        if parent0.used_codons[i] != 0:
            effective_genomes0.append(i)
        if parent1.used_codons[i] != 0:
            effective_genomes1.append(i)
    effective_genomes = list(set(effective_genomes0).intersection(effective_genomes1))
    
    #Define which genome will be used for crossover
    #We use the same genome for both parents
    if effective_genomes: #if both parents have at least one effective genome in common
        crossover_genome = random.choice(effective_genomes) 
        possible_crossover_codons0 = parent0.used_codons[crossover_genome]
        possible_crossover_codons1 = parent1.used_codons[crossover_genome]
    else: #Otherwise (it also includes the case in which at least one ind is invalid)
        crossover_genome = random.randint(0, n_genomes-1)
        if parent0.used_codons[crossover_genome] != 0: #this ind is valid and this genome is used
            possible_crossover_codons0 = parent0.used_codons[crossover_genome]
        else: #this ind is invalid or this genome is not used
            possible_crossover_codons0 = len(parent0.genome[crossover_genome])
        if parent1.used_codons[crossover_genome] != 0: 
            possible_crossover_codons1 = parent1.used_codons[crossover_genome]
        else:
            possible_crossover_codons1 = len(parent1.genome[crossover_genome])
    
    #if parent0.used_codons[crossover_genome] == 0: #parent0 is invalid or this genome was not used in the mapping
#    if parent0.invalid:
#        possible_crossover_codons0 = len(parent0.genome[crossover_genome])
#    else:
#        possible_crossover_codons0 = parent0.used_codons[crossover_genome]
    #if parent1.used_codons[crossover_genome] == 0: #parent1 is invalid or this genome was not used in the mapping
#    if parent1.invalid:
#        possible_crossover_codons1 = len(parent1.genome[crossover_genome])
#    else:
#        possible_crossover_codons1 = parent1.used_codons[crossover_genome]

    continue_ = True
    while continue_:
        new_genome0 = parent0.genome.copy()
        new_genome1 = parent1.genome.copy()
        
        #Set points for crossover within the effective part of the genomes
        try:
            point0 = random.randint(1, possible_crossover_codons0)
        except ValueError:
            pass
        point1 = random.randint(1, possible_crossover_codons1)
        
        #Operate crossover
        new_genome0[crossover_genome] = parent0.genome[crossover_genome][0:point0] + parent1.genome[crossover_genome][point1:]
        new_genome1[crossover_genome] = parent1.genome[crossover_genome][0:point1] + parent0.genome[crossover_genome][point0:]
        
        new_ind0 = reMap(parent0, new_genome0, bnf_grammar, max_depth, codon_consumption)
        new_ind1 = reMap(parent1, new_genome1, bnf_grammar, max_depth, codon_consumption)
        
        if codon_consumption == 'multichromosomalGE':
            continue_ = new_ind0.depth > max_depth or new_ind1.depth > max_depth
        elif codon_consumption == 'multiGE':
            continue_ = max(new_ind0.depth) > max_depth or max(new_ind1.depth) > max_depth
        else:
            raise ValueError("Codon consumption should be multichromosomalGE or multiGE")
    
    if max_genome_length:
        raise ValueError("Multi GE is implemented with max_genome_length = None")
        
    return new_ind0, new_ind1   

def crossover_onepoint_multiGE(parent0, parent1, bnf_grammar, max_depth, codon_consumption, 
                       genome_representation, max_genome_length):
    """
    
    """
    
    if len(parent0.genome) != len(parent1.genome):
        raise ValueError("Wrong codon consumption")
    elif genome_representation != 'list':
        raise ValueError("Multi GE is implemented only using list as genome representation")
    else:
        n_genomes = len(parent0.genome)
        
    #Define which genome will be used for crossover
    #We use the same genome for both parents
    #crossover_genome = random.randint(0, n_genomes-1)
    new_genome0 = parent0.genome.copy()
    new_genome1 = parent1.genome.copy()
    for i in range(n_genomes):
        if parent0.invalid: #used_codons = 0
            possible_crossover_codons0 = len(parent0.genome[i])
        else:
            possible_crossover_codons0 = min(len(parent0.genome[i]), parent0.used_codons[i]) #in case of wrapping, used_codons can be greater than genome's length
        if parent1.invalid:
            possible_crossover_codons1 = len(parent1.genome[i])
        else:
            possible_crossover_codons1 = min(len(parent1.genome[i]), parent1.used_codons[i])

        #Set points for crossover within the effective part of the genomes
        point0 = random.randint(1, possible_crossover_codons0)
        point1 = random.randint(1, possible_crossover_codons1)
        
        #Operate crossover
        new_genome0[i] = parent0.genome[i][0:point0] + parent1.genome[i][point1:]
        new_genome1[i] = parent1.genome[i][0:point1] + parent0.genome[i][point0:]
        
        new_ind0 = reMap(parent0, new_genome0, bnf_grammar, max_depth, codon_consumption)
        new_ind1 = reMap(parent1, new_genome1, bnf_grammar, max_depth, codon_consumption)
        
    if max_genome_length:
        raise ValueError("Multi GE is implemented with max_genome_length = None")
        
    return new_ind0, new_ind1   

#def selRandom(individuals, k):
#    return [random.choice(individuals) for i in range(k)]

#def selTournament(individuals, k, tournsize, fit_attr="fitness"):
#    chosen = []
#    for i in range(k):
#        aspirants = selRandom(individuals, tournsize)
#        valid_aspirants =  [ind for ind in aspirants if not ind.invalid]
#        if len(valid_aspirants):
#            chosen.append(max(valid_aspirants, key=attrgetter(fit_attr)))
#        else:
#            chosen.append(random.choice(aspirants))
#    return chosen

def selLexicase(individuals, k):
    """Returns an individual that does the best on the fitness cases when
    considered one at a time in random order.
    http://faculty.hampshire.edu/lspector/pubs/lexicase-IEEE-TEC.pdf
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.
    """
    selected_individuals = []
    valid_individuals = individuals.copy()#[i for i in individuals if not i.invalid]
    l_samples = np.shape(valid_individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if ind.fitness.values[0] == 0]
    
    cases = list(range(0,l_samples))
    #fit_weights = valid_individuals[0].fitness.weights
    candidates = valid_individuals
    
    if len(inds_fitness_zero) > 0:
        for i in range(k):
            selected_individuals.append(random.choice(inds_fitness_zero))
        return selected_individuals

    for i in range(k):
        #cases = list(range(len(valid_individuals[0].fitness.values)))
        random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            candidates_update = [i for i in candidates if i.fitness_each_sample[cases[0]] == True]
            
            if len(candidates_update) == 0:
                #no candidate correctly predicted the case
                pass
            else:
                candidates = candidates_update    
            del cases[0]                    

        #If there is only one candidate remaining, it will be selected
        #If there are more than one, the choice will be made randomly
        selected_individuals.append(random.choice(candidates))
        
        cases = list(range(0,l_samples))
        candidates = valid_individuals

    return selected_individuals

def selLexicaseFilter(individuals, k):
    """
   

    """
    selected_individuals = []
    #valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if ind.fitness.values[0] == 0]
    if len(inds_fitness_zero) > 0:
        for i in range(k):
            selected_individuals.append(random.choice(inds_fitness_zero))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        random.shuffle(cases)
        while len(cases) > 0 and len(pool) > 1:
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals

def selLexicaseFilterCount(individuals, k):
    """
   

    """
    selected_individuals = []
    #valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if ind.fitness.values[0] == 0]
    if len(inds_fitness_zero) > 0:
        for i in range(k):
            selected_individuals.append(random.choice(inds_fitness_zero))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals


def selLexi2_nodes(individuals, k):
    """
   

    """
    selected_individuals = []
   # valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if all(item == 1 for item in ind.fitness_each_sample)] #all checks if every fitness sample = 1
    if len(inds_fitness_zero) > 0:
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, inds_fitness_zero))
        candidates = [ind for ind in inds_fitness_zero if ind.nodes == best_val_for_nodes]
        for i in range(k):
            selected_individuals.append(random.choice(candidates))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, cands))
        cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        random.shuffle(cases)
        while len(cases) > 0 and len(pool) > 1:
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals

def selLexi2_rmseCount(individuals, k):
    """
   

    """
    selected_individuals = []
   # valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if all(item == 1 for item in ind.fitness_each_sample)] #all checks if every fitness sample = 1
    if len(inds_fitness_zero) > 0:
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, inds_fitness_zero))
        candidates = [ind for ind in inds_fitness_zero if ind.nodes == best_val_for_nodes]
        for i in range(k):
            selected_individuals.append(random.choice(candidates))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        f = min
        best_val_for_rmse = f(map(lambda x: x.rmse, cands))
        cands = [ind for ind in cands if ind.rmse == best_val_for_rmse]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals

def selLexi2_nodesCount(individuals, k):
    """
   

    """
    selected_individuals = []
   # valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if all(item == 1 for item in ind.fitness_each_sample)] #all checks if every fitness sample = 1
    if len(inds_fitness_zero) > 0:
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, inds_fitness_zero))
        candidates = [ind for ind in inds_fitness_zero if ind.nodes == best_val_for_nodes]
        for i in range(k):
            selected_individuals.append(random.choice(candidates))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, cands))
        cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals

def selLexi2_nodesCountTies(individuals, k):
    """
    same as selLexi2_nodesCount, but also registers the number of ties in the selected individual in the attribute 'ties'

    """
    selected_individuals = []
   # valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if all(item == 1 for item in ind.fitness_each_sample)] #all checks if every fitness sample = 1
    if len(inds_fitness_zero) > 0:
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, inds_fitness_zero))
        candidates = [ind for ind in inds_fitness_zero if ind.nodes == best_val_for_nodes]
        for i in range(k):
            selected_individuals.append(random.choice(candidates))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        for ind in cands:
            ind.ties = len(cands)
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, cands))
        cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals
   
def selEpsilonLexi2_nodesCountTies(individuals, k):
    """
    same as selEpsilonLexi2_nodesCount, but also registers the number of ties in the selected individual in the attribute 'ties'
   
    """
    selected_individuals = []
   # valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if all(item == 1 for item in ind.fitness_each_sample)] #all checks if every fitness sample = 1
    if len(inds_fitness_zero) > 0:
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, inds_fitness_zero))
        candidates = [ind for ind in inds_fitness_zero if ind.nodes == best_val_for_nodes]
        for i in range(k):
            selected_individuals.append(random.choice(candidates))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
    min_ = np.nanmin(fitness_cases_matrix, axis=0)
    #min_ = np.min(np.where(fitness_cases_matrix != 0, fitness_cases_matrix, np.inf), axis=0)

    #mad = robust.mad(fitness_cases_matrix, axis=0, c=1.0)
    #mad = np.std(fitness_cases_matrix, axis=0)
    #try:
    mad = median_abs_deviation(fitness_cases_matrix, axis=0)
    #except (MemoryError):
    #    pass
    
    for i in range(len(candidates)):
        for j in range(l_samples):
            #if fitness_cases_matrix[i][j] >= min_[j] and fitness_cases_matrix[i][j] <= min_[j] + mad[j]:
            if fitness_cases_matrix[i][j] <= min_[j] + mad[j]:
                fitness_cases_matrix[i][j] = 1
                candidates[i].fitness_each_sample[j] = 1
            else:
                fitness_cases_matrix[i][j] = 0
                candidates[i].fitness_each_sample[j] = 0
                
    error_vectors = list(fitness_cases_matrix)

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        for ind in cands:
            ind.ties = len(cands)
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, cands))
        cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
     #   print(sum(pool[0].fitness_each_sample)/l_samples, pool[0].mce)
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals
     
def selEpsilonLexi2_nodesCount(individuals, k):
    """
        
    """
    selected_individuals = []
   # valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if all(item == 1 for item in ind.fitness_each_sample)] #all checks if every fitness sample = 1
    if len(inds_fitness_zero) > 0:
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, inds_fitness_zero))
        candidates = [ind for ind in inds_fitness_zero if ind.nodes == best_val_for_nodes]
        for i in range(k):
            selected_individuals.append(random.choice(candidates))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
    min_ = np.nanmin(fitness_cases_matrix, axis=0)
    #min_ = np.min(np.where(fitness_cases_matrix != 0, fitness_cases_matrix, np.inf), axis=0)

    #mad = robust.mad(fitness_cases_matrix, axis=0, c=1.0)
    #mad = np.std(fitness_cases_matrix, axis=0)
    #try:
    mad = 1.5*median_abs_deviation(fitness_cases_matrix, axis=0)
    #except (MemoryError):
    #    pass
    
    for i in range(len(candidates)):
        for j in range(l_samples):
            #if fitness_cases_matrix[i][j] >= min_[j] and fitness_cases_matrix[i][j] <= min_[j] + mad[j]:
            if fitness_cases_matrix[i][j] <= min_[j] + mad[j]:
                fitness_cases_matrix[i][j] = 1
                candidates[i].fitness_each_sample[j] = 1
            else:
                fitness_cases_matrix[i][j] = 0
                candidates[i].fitness_each_sample[j] = 0
                
    error_vectors = list(fitness_cases_matrix)

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, cands))
        cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
     #   print(sum(pool[0].fitness_each_sample)/l_samples, pool[0].mce)
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals

def selBatchLexicase(individuals, k, batch_size=20):
    """
        
    """
    selected_individuals = []
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors
	
    n_batches = math.ceil(l_samples / batch_size)
    
    for i in range(len(candidates)):
        candidates[i].fitness_each_batch = [0] * n_batches

    for _ in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            candidate = random.choice(list_)
            candidate.fitness_each_batch = [0] * n_batches
            pool.append(candidate) 
        random.shuffle(cases)
        batch_ = 0
        while batch_ < n_batches - 1 and len(pool) > 1:
            #Build batch of batch_size cases
            for _ in range(batch_size):
                for i in range(len(pool)):
                    pool[i].fitness_each_batch[batch_] += pool[i].fitness_each_sample[cases[0]]
                del cases[0]
            f = max
            best_val_for_batch = f(map(lambda x: x.fitness_each_batch[batch_], pool))
            pool = [ind for ind in pool if ind.fitness_each_batch[batch_] == best_val_for_batch]
            batch_ += 1
        if batch_ == n_batches - 1 and len(pool) > 1:
            #Build batch with the remaining cases
            for case in cases:
                for i in range(len(pool)):
                    pool[i].fitness_each_batch[batch_] += pool[i].fitness_each_sample[case]
            f = max
            best_val_for_batch = f(map(lambda x: x.fitness_each_batch[batch_], pool))
            pool = [ind for ind in pool if ind.fitness_each_batch[batch_] == best_val_for_batch]
            batch_ += 1
        
        #Despite filtering the individuals initially, we can have more than one remaining in the pool after checking the batches, because inds with different behaviours can have the same batch fitness
        if len(pool) == 1:
            selected_individual = pool[0]
        else:
#            print("error")
            selected_individual = random.choice(pool)
        selected_individual.n_cases = batch_
        selected_individuals.append(selected_individual)
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals

def selBatchEpsilonLexi2_nodesCount(individuals, k, batch_size=20):
    """
        
    """
    selected_individuals = []
    error_vectors = [ind.fitness_each_sample for ind in individuals]
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
   
    pop_size = len(individuals)
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    n_batches = math.ceil(l_samples / batch_size)
    
    cases = list(range(0,l_samples))
    random.shuffle(cases)
    fitness_batches_matrix = np.zeros([pop_size, n_batches], dtype=float) # inds (rows) x samples (cols)
    #partitions
    for i in range(n_batches-1):
        for _ in range(batch_size):
            fitness_batches_matrix[:,i] += fitness_cases_matrix[:,cases[0]]
            del cases[0]
    for case in cases:
        fitness_batches_matrix[:,n_batches-1] += fitness_cases_matrix[:,case]

    min_ = np.nanmin(fitness_batches_matrix, axis=0)
    mad = median_abs_deviation(fitness_batches_matrix, axis=0)

    candidates = individuals
    for i in range(len(candidates)):
        candidates[i].fitness_each_batch = [0] * n_batches
        for j in range(n_batches):
            if fitness_batches_matrix[i][j] <= min_[j] + mad[j]:
                fitness_batches_matrix[i][j] = 1
                candidates[i].fitness_each_batch[j] = 1
            else:
                fitness_batches_matrix[i][j] = 0
                candidates[i].fitness_each_batch[j] = 0
            
    error_vectors = list(fitness_batches_matrix)

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_batch == unique_error_vectors[i]]
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, cands))
        cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    batches = list(range(0,n_batches))
    
    for _ in range(k):
        random.shuffle(batches)
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        
        count_ = 0
        while len(batches) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_batch[batches[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_batch[batches[0]] == best_val_for_case]
            del batches[0]
            
        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate
        
        batches = list(range(0,n_batches))

    return selected_individuals
        
def selBatchEpsilonLexi2_nodesCountOld(individuals, k, batch_size=2):
    """
    different batches for select each individual    
    """
    selected_individuals = []
    error_vectors = [ind.fitness_each_sample for ind in individuals]
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
   # valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    pop_size = len(individuals)
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    n_batches = math.ceil(l_samples / batch_size)
    
    candidates = individuals
    
    for _ in range(k):
        cases = list(range(0,l_samples))
        random.shuffle(cases)
        fitness_batches_matrix = np.zeros([pop_size, n_batches], dtype=float) # inds (rows) x samples (cols)
        #partitions
        for i in range(n_batches-1):
            for _ in range(batch_size):
                fitness_batches_matrix[:,i] += fitness_cases_matrix[:,cases[0]]
                del cases[0]
        for case in cases:
            fitness_batches_matrix[:,n_batches-1] += fitness_cases_matrix[:,case]

        min_ = np.nanmin(fitness_batches_matrix, axis=0)
        mad = median_abs_deviation(fitness_batches_matrix, axis=0)

        for i in range(len(candidates)):
            candidates[i].fitness_each_batch = [0] * n_batches
            for j in range(n_batches):
                if fitness_batches_matrix[i][j] <= min_[j] + mad[j]:
                    fitness_batches_matrix[i][j] = 1
                    candidates[i].fitness_each_batch[j] = 1
                else:
                    fitness_batches_matrix[i][j] = 0
                    candidates[i].fitness_each_batch[j] = 0
                
        error_vectors = list(fitness_batches_matrix)

        unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
        unique_error_vectors = [list(i) for i in unique_error_vectors]
        
        candidates_prefiltered_set = []
        for i in range(len(unique_error_vectors)):
            cands = [ind for ind in candidates if ind.fitness_each_batch == unique_error_vectors[i]]
            f = min
            best_val_for_nodes = f(map(lambda x: x.nodes, cands))
            cands = [ind for ind in cands if ind.nodes == best_val_for_nodes]
            candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        
        count_ = 0
        while count_ < n_batches and len(pool) > 1:
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_batch[count_], pool))
            pool = [ind for ind in pool if ind.fitness_each_batch[count_] == best_val_for_case]
            count_ += 1

        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate

    return selected_individuals

def selEpsilonLexicaseCount(individuals, k):
    """
        
    """
    selected_individuals = []
   # valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if all(item == 1 for item in ind.fitness_each_sample)] #all checks if every fitness sample = 1
    if len(inds_fitness_zero) > 0:
        f = min
        best_val_for_nodes = f(map(lambda x: x.nodes, inds_fitness_zero))
        candidates = [ind for ind in inds_fitness_zero if ind.nodes == best_val_for_nodes]
        for i in range(k):
            selected_individuals.append(random.choice(candidates))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]
    
    fitness_cases_matrix = np.array(error_vectors) # inds (rows) x samples (cols)
    min_ = np.nanmin(fitness_cases_matrix, axis=0)
    #mad = robust.mad(fitness_cases_matrix, axis=0, c=1.0)
    mad = median_abs_deviation(fitness_cases_matrix, axis=0)
    
    for i in range(len(candidates)):
        for j in range(l_samples):
            if fitness_cases_matrix[i][j] <= min_[j] + mad[j]:
                fitness_cases_matrix[i][j] = 1
                candidates[i].fitness_each_sample[j] = 1
            else:
                fitness_cases_matrix[i][j] = 0
                candidates[i].fitness_each_sample[j] = 0
                
    error_vectors = list(fitness_cases_matrix)

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered_set = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        candidates_prefiltered_set.append(cands) #list of lists, each one with the inds with the same error vectors and same number of nodes

    for i in range(k):
        #fill the pool only with candidates with unique error vectors
        pool = []
        for list_ in candidates_prefiltered_set:
            pool.append(random.choice(list_)) 
        random.shuffle(cases)
        count_ = 0
        while len(cases) > 0 and len(pool) > 1:
            count_ += 1
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], pool))
            pool = [ind for ind in pool if ind.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        pool[0].n_cases = count_
        selected_individuals.append(pool[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases

    return selected_individuals

def selLexicaseFilterDepth(individuals, k):
    """
   to be fixed as nodes

    """
    selected_individuals = []
    #valid_individuals = individuals#.copy()#[i for i in individuals if not i.invalid]
    l_samples = np.shape(individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if ind.fitness.values[0] == 0]
    if len(inds_fitness_zero) > 0:
        f = min
        best_val_for_depth = f(map(lambda x: x.depth, inds_fitness_zero))
        candidates = [i for i in inds_fitness_zero if i.depth == best_val_for_depth]
        for i in range(k):
            selected_individuals.append(random.choice(candidates))
        return selected_individuals
    
    cases = list(range(0,l_samples))
    candidates = individuals
    
    error_vectors = [ind.fitness_each_sample for ind in candidates]

    unique_error_vectors = list(set([tuple(i) for i in error_vectors]))
    unique_error_vectors = [list(i) for i in unique_error_vectors]
    
    candidates_prefiltered = []
    for i in range(len(unique_error_vectors)):
        cands = [ind for ind in candidates if ind.fitness_each_sample == unique_error_vectors[i]]
        f = min
        best_val_for_depth = f(map(lambda x: x.depth, cands))
        cands = [i for i in cands if i.depth == best_val_for_depth]
        candidates_prefiltered.append(random.choice(cands))

    candidates = candidates_prefiltered.copy()
    for i in range(k):
        random.shuffle(cases)
        while len(cases) > 0 and len(pool) > 1:
            f = max
            best_val_for_case = f(map(lambda x: x.fitness_each_sample[cases[0]], candidates))
            candidates = [i for i in candidates if i.fitness_each_sample[cases[0]] == best_val_for_case]
            del cases[0]                    

        selected_individuals.append(candidates[0]) #Select the remaining candidate
        cases = list(range(0,l_samples)) #Recreate the list of cases
        candidates = candidates_prefiltered.copy() #Restart the pool of candidates

    return selected_individuals

def mutation_one_codon(ind, mut_probability, codon_size, bnf_grammar, 
                                     max_depth, codon_consumption,
                                     invalidate_max_depth,
                                     max_genome_length): #TODO include code for this one
    """

    """
    if ind.invalid: #used_codons = 0
        possible_mutation_codons = len(ind.genome)
    else:
        possible_mutation_codons = min(len(ind.genome), ind.used_codons) #in case of wrapping, used_codons can be greater than genome's length
    continue_ = True
    
    genome = ind.genome.copy()
    while continue_:
        genome_mutated = genome.copy()
        codon_to_mutate = random.randint(0, possible_mutation_codons-1)
        genome_mutated[codon_to_mutate] = random.randint(0, codon_size)
        new_ind = reMap(ind, genome_mutated, bnf_grammar, max_depth, codon_consumption)
        
        if invalidate_max_depth: # In the mapping, if ind surpasses max depth, it is invalid, and we won't redo mutation
            continue_ = False
        else: # We check if ind surpasses max depth, and if so we will redo mutation
            continue_ = new_ind.depth > max_depth
            #print("repeat: ", continue_)
        
    del new_ind.fitness.values
    #print("finished")
    return new_ind,

def mutation_int_flip_per_codon(ind, mut_probability, codon_size, bnf_grammar, max_depth, 
                                codon_consumption, invalidate_max_depth,
                                max_genome_length):
    """

    """
    # Operation mutation within the effective part of the genome
    if ind.invalid: #used_codons = 0
        possible_mutation_codons = len(ind.genome)
    else:
        possible_mutation_codons = min(len(ind.genome), ind.used_codons) #in case of wrapping, used_codons can be greater than genome's length
    continue_ = True
    
    genome = copy.deepcopy(ind.genome)
    mutated_ = False

    while continue_:
        for i in range(possible_mutation_codons):
            if random.random() < mut_probability:
                genome[i] = random.randint(0, codon_size)
                mutated_ = True
               # break
    
        new_ind = reMap(ind, genome, bnf_grammar, max_depth, codon_consumption)
        
        if invalidate_max_depth: # In the mapping, if ind surpasses max depth, it is invalid, and we won't redo mutation
            continue_ = False
        else: # We check if ind surpasses max depth, and if so we will redo mutation
            continue_ = new_ind.depth > max_depth
        
#    if max_genome_length:
#        if new_ind.depth > max_depth or len(new_ind.genome) > max_genome_length:
#            return ind,
#        else:
#            return new_ind,
#    else:
        #if new_ind.depth > max_depth:
        #    return ind,
        #else:
    if mutated_:
        del new_ind.fitness.values
    return new_ind,
        
def mutation_int_flip_per_codon_multiGE_One(ind, mut_probability, codon_size, bnf_grammar, max_depth, 
                                codon_consumption, max_genome_length):
    """

    """
    n_genomes = len(ind.genome)
    
    #Identify effective genomes (which are actually being used in the mapping process)
    effective_genomes = []
    if not ind.invalid:
        for i in range(n_genomes):
            if ind.used_codons[i] != 0:
                effective_genomes.append(i)
    
    #Define which genome will be used for mutation
    # Operation mutation within the effective part of the genome
    if ind.invalid:
        mutation_genome = random.randint(0, n_genomes-1)
        possible_mutation_codons = len(ind.genome[mutation_genome])
    else:
        mutation_genome = random.choice(effective_genomes)
        possible_mutation_codons = ind.used_codons[mutation_genome]
        
    continue_ = True
    
    genome = copy.deepcopy(ind.genome)

    while continue_:
        for i in range(possible_mutation_codons):
            if random.random() < mut_probability:
                genome[mutation_genome][i] = random.randint(0, codon_size)
    
        new_ind = reMap(ind, genome, bnf_grammar, max_depth, codon_consumption)
        
        if codon_consumption == 'multichromosomalGE':
            continue_ = new_ind.depth > max_depth
        elif codon_consumption == 'multiGE':
            continue_ = max(new_ind.depth) > max_depth
        else:
            raise ValueError("Codon consumption should be multichromosomalGE or multiGE")
        
    if max_genome_length:
        raise ValueError("Multi GE is implemented with max_genome_length = None")
    
    return new_ind,

def mutation_int_flip_per_codon_multiGE(ind, mut_probability, codon_size, bnf_grammar, max_depth, 
                                codon_consumption, max_genome_length):
    """

    """
    genome = copy.deepcopy(ind.genome)
    for i in range(len(genome)):
        # Operation mutation within the effective part of the genome
        if ind.invalid: #used_codons = 0
            possible_mutation_codons = len(genome[i])
        else:
            possible_mutation_codons = min(len(genome[i]), ind.used_codons[i]) #in case of wrapping, used_codons can be greater than genome's length

        for j in range(possible_mutation_codons):
            if random.random() < mut_probability:
                genome[i][j] = random.randint(0, codon_size)
    
        new_ind = reMap(ind, genome, bnf_grammar, max_depth, codon_consumption)
            
    if max_genome_length:
        raise ValueError("Multi GE is implemented with max_genome_length = None")
    
    return new_ind,

def reMap(ind, genome, bnf_grammar, max_tree_depth, codon_consumption):
    #TODO refazer todo o reMap para nao copiar o ind
    #
    #ind = Individual(genome, bnf_grammar, max_tree_depth, codon_consumption)
    #ind = Individual(genome, bnf_grammar, max_tree_depth, codon_consumption)
    ind.genome = genome
    if codon_consumption == 'lazy':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_lazy(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'eager':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_eager(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'leap':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure, ind.tile_size, ind.effective_positions = mapper_leap(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'leap2' or codon_consumption == 'leap3':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure, ind.tile_size, ind.effective_positions = mapper_leap2(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'parameterised':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_parameterised(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'multiGE':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_multi(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'multichromosomalGE':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure = mapper_multichromosomal(genome, bnf_grammar, max_tree_depth)         
    elif codon_consumption == 'cosmo_eager':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure, ind.cosmo = mapper_cosmo(genome, bnf_grammar, max_tree_depth)
    elif codon_consumption == 'cosmo_eager_depth':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure, ind.cosmo = mapper_cosmo_ext(genome, bnf_grammar, max_tree_depth)    
    elif codon_consumption == 'cosmo_total':
        ind.phenotype, ind.nodes, ind.depth, \
        ind.used_codons, ind.invalid, ind.n_wraps, \
        ind.structure, ind.cosmo = mapper_cosmo_total(genome, bnf_grammar, max_tree_depth)     
    else:
        raise ValueError("Unknown mapper")
        
    return ind

def replace_nth(string, substring, new_substring, nth):
    find = string.find(substring)
    i = find != -1
    while find != -1 and i != nth:
        find = string.find(substring, find + 1)
        i += 1
    if i == nth:
        return string[:find] + new_substring + string[find+len(substring):]
    return string

def selLexicase(individuals, k):
    """
    """
    selected_individuals = []
    valid_individuals = [i for i in individuals if not i.invalid]
    l_samples = np.shape(valid_individuals[0].fitness_each_sample)[0]
    
    cases = list(range(0,l_samples))
    candidates = valid_individuals
    
    for i in range(k):
        random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            candidates_update = [i for i in candidates if i.fitness_each_sample[cases[0]] == True]
            
            if len(candidates_update) == 0:
                #no candidate correctly predicted the case
                pass
            else:
                candidates = candidates_update    
            del cases[0]                    

        #If there is only one candidate remaining, it will be selected
        #If there are more than one, the choice will be made randomly
        selected_individuals.append(random.choice(candidates))
        
        cases = list(range(0,l_samples))
        candidates = valid_individuals

    return selected_individuals

def selLexicaseCount(individuals, k):
    """Same as Lexicase Selection, but counting attempts of filtering and
    updating respective attributes on ind.
    
    If some ind has fitness equal to zero, do not enter in the loop.
    Instead, select randomly within the inds with fitness equal to zero.
    """
    selected_individuals = []
    valid_individuals = [i for i in individuals if not i.invalid]
    l_samples = np.shape(valid_individuals[0].fitness_each_sample)[0]
    
    inds_fitness_zero = [ind for ind in individuals if ind.fitness.values[0] == 0]
    
    #For analysing Lexicase selection
    samples_attempted = [0]*l_samples
    samples_used = [0]*l_samples
    samples_unsuccessful1 = [0]*l_samples
    samples_unsuccessful2 = [0]*l_samples
    inds_to_choose = [0]*k
    times_chosen = [0]*4
    
    cases = list(range(0,l_samples))
    #fit_weights = valid_individuals[0].fitness.weights
    candidates = valid_individuals
    
    if len(inds_fitness_zero) > 0:
        for i in range(k):
            selected_individuals.append(random.choice(inds_fitness_zero))
            inds_to_choose[i] = len(inds_fitness_zero)
            if len(inds_fitness_zero) == 1:
                times_chosen[0] += 1 #The choise was made by error
            else:
                times_chosen[3] += 1 #The choise was made by randomly
        samples_attempted = [x+k for x in samples_attempted]
        samples_used = [x+1 for x in samples_used]
        samples_unsuccessful1 = [x+k-1 for x in samples_unsuccessful1]
        
        return selected_individuals, samples_attempted, samples_used, samples_unsuccessful1, samples_unsuccessful2, inds_to_choose, times_chosen

    for i in range(k):
        #cases = list(range(len(valid_individuals[0].fitness.values)))
        random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            print(cases[0])
            print(candidates[0].fitness_each_sample[cases[0]])
            print(type(True))
            #f = min if fit_weights[cases[0]] < 0 else max
            candidates_update = [i for i in candidates if i.fitness_each_sample[cases[0]] == True]
            
            samples_attempted[cases[0]] += 1
            if (len(candidates_update) < len(candidates)) and (len(candidates_update) > 0):
                samples_used[cases[0]] += 1
            if (len(candidates_update) == len(candidates)):
                samples_unsuccessful1[cases[0]] += 1
            if len(candidates_update) == 0:
                samples_unsuccessful2[cases[0]] += 1
            
            if len(candidates_update) == 0:
                #no candidate correctly predicted the case
                pass
            else:
                candidates = candidates_update    
            del cases[0]                    

            #best_val_for_case = f(map(lambda x: x.fitness.values[cases[0]], candidates))

            #candidates = list(filter(lambda x: x.fitness.values[cases[0]] == best_val_for_case, candidates))
            #cases.pop(0)

        #If there is only one candidate remaining, it will be selected
        if len(candidates) == 1:
            selected_individuals.append(candidates[0])
            inds_to_choose[i] = 1
            times_chosen[0] += 1 #The choise was made by fitness
        else: #If there are more than one, the choice will be made randomly
            selected_individuals.append(random.choice(candidates))
            inds_to_choose[i] = len(candidates)
            times_chosen[3] += 1 #The choise was made by randomly
        
        cases = list(range(0,l_samples))
        candidates = valid_individuals

    return selected_individuals, samples_attempted, samples_used, samples_unsuccessful1, samples_unsuccessful2, inds_to_choose, times_chosen



#GRAMMAR_FILE = 'TwoBoxes.bnf' #heartDisease.bnf'    
#GRAMMAR_FILE = 'parity4.bnf'    

#structure = [[0, 0, 0, 0], [0, 2, 0, 0], [0, 3, 1, 2]]
#genome = [[2, 4, 3, 6, 3, 6], [2, 6, 3, 6, 3, 6], [2, 7, 3, 7, 3, 8]]
#genome = [[2, 3, 3, 6], [4, 2, 6, 3, 6, 3, 6], [3, 4, 7, 3, 8]] #Out[14]: 'and_(x[0],x[1])'
#genome = [random.randint(0,255) for _ in range(25)]

#genome=[222, 93, 202, 24, 151, 151, 40, 198, 78, 249, 81, 135, 133, 139, 183, 121, 139, 58, 131, 97, 215, 10, 39, 98, 255]
#print(genome)
#genome = [216, 201, 25, 66, 8, 214, 174, 253, 121, 55, 242, 74, 6, 114, 197]
#genome = [2, 4, 3, 3, 3, 4] #for leap (parity3: 'and_(nor_(x[2],or_(x[1],x[2])),x[0])')
#genome = [15, 200, 18, 200, 19, 8, 16, 200, 29, 10, 39, 8, 49, 9] #equivalent to above for leap2


# 200, 200, 4, 200, 200, 200, 
# 200, 200, 200, 24, 200, 200, 
# 200, 200, 200, 200, 200, 35, 
# 200, 20, 200, 200, 200, 200, 
# 200, 22, 200, 200, 200, 200, 
# 200, 23, 200, 200, 200, 200, 
# 200, 25, 200, 200, 200, 200, 
# 200, 200, 200, 24, 200, 200, 
# 200, 200, 200, 200, 200, 12]

#genome = [10, 20, 4, 24, 200, 35, #leap2
# 200, 22, 200, 200, 200, 200, 
# 200, 23, 200, 200, 200, 200, 
# 200, 25, 200, 24, 200, 12]


#codon_consumption = 'multichromosomalGE'


#struc = []
#for i in range(len(genome)):
#    if i % 2 == 0:
#        struc.append(genome[i] % 5)
#    else:
#        struc.append(genome[i] % 3)

#from os import path

#GRAMMAR_FILE = 'Vladislavleva4.bnf'
#BNF_GRAMMAR = Grammar(path.join("grammars", GRAMMAR_FILE))

#codon_consumption = 'cosmo_eager_depth'
#ind = Individual(genome, BNF_GRAMMAR, 10, codon_consumption)
#print(ind.phenotype, ind.cosmo, ind.invalid)