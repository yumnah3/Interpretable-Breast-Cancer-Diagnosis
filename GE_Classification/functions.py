#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:31:49 2021

@author: allan
"""

import random
import numpy as np
import math

def WTA(a, b):
    #print(a,b)
    return np.where(b > a, 1, 0)
    #if a > b:
    #    return 0
    #else:
    #    return 1
    
def sigmoid(arr):
    """
    Calculate the sigmoid of a given input x, which could be an array.
    
    Arguments:
    x -- A numeric value.
    
    Returns:
    s -- The sigmoid of x.
    """
    if np.isscalar(arr):
        arr = np.array([arr])
    return 1 / (1 + np.exp(-arr))


def int_(string):
    if string == '0' or string =='00': #the number is 0
        return 0
    else:
        while string[0] == '0' and len(string) >= 2: #the number starts with 0
            string = string[1:]
        if string == '0': #the number was filled only with 0's (more than 2)
            return 0
        else:
            return int(string)

def int_to_char(int_):
    try:
        return chr(int_)
    except ValueError:
        return '0'

def mod(a, b):
    return a % b

def get_char(string, idx):
    length = len(string)
    if idx < 0 or idx >= length:
        return ''
    else:
        return string[idx]

def char_to_int(char):
    if len(char) == 1:
        return ord(char)
    else:
        return 0
        
def WA(a, b, x):
    x = float(x)
    return x*a+(1-x)*b

def OWA(a, b, x):
    x = float(x)
    return x*np.maximum(a, b)+(1-x)*np.minimum(a, b)

def minimum(a, b):
    return np.minimum(a, b)

def maximum(a, b):
    return np.maximum(a, b)
    
def dilator(b):
    return b**0.5

def concentrator(b):
    return b**2

def complement(b):
    return 1 - b

def power(a, b):
   return np.power(a, b)

def exp(a):
    result = np.exp(a)
    #if result == float("inf"):
    #    raise OverflowError
    #else:
    return result

def neg(a):
    return -a

def pdiv(a, b):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(b == 0, np.ones_like(a), a / b)
    except ZeroDivisionError:
        # In this case we are trying to divide two constants, one of which is 0
        # Return a constant.
        return 1.0
    
def sin(n):
    return np.sin(n)

def cos(n):
    return np.cos(n)

def tanh(n):
    return np.tanh(n)


def add(a, b):
    return np.add(a,b)

def sub(a, b):
    return np.subtract(a,b)

def mul(a, b):
    return np.multiply(a,b)

def psqrt(a):
    return np.sqrt(abs(a))

def max_(a,b):
    return np.maximum(a, b)

def min_(a,b):
    return np.minimum(a, b)

def plog(a):
    return np.log(1.0 + np.abs(a))

def log(a):
    return np.log(a)

def not_(a):
    return np.logical_not(a)

def and_(a, b):
    return np.logical_and(a,b)

def or_(a, b):
    return np.logical_or(a,b)

def nand_(a, b):
    return np.logical_not(np.logical_and(a,b))

def nor_(a, b):
    return np.logical_not(np.logical_or(a,b))

def xor_ (a,b):
    return or_(and_(a, not_(b)), and_(not_(a), b))

def greater_than_or_equal(a, b):
    return a >= b

def less_than_or_equal(a, b):
    return a <= b

def if_(i, o0, o1):
    """If _ than _ else _"""
    return np.where(i, o0, o1)

def progn(a, b):
    if type(a) is not tuple:
        exec(a)
    if type(b) is not tuple:
        output_b = exec(b)
    else:
        output_b = b
    return output_b
        
def left(lawnmower):
    #global lawnmower
    if lawnmower[2] == 'E':
        lawnmower[2] = 'N'
    elif lawnmower[2] == 'W':
        lawnmower[2] = 'S'
    elif lawnmower[2] == 'N':
        lawnmower[2] = 'W'
    elif lawnmower[2] == 'S':
        lawnmower[2] = 'E'
    return (0,0)

def v8a(a, b):
    return (add_toroidal_8(a[0], b[0]), add_toroidal_8(a[1], b[1]))
    
def frog(a, lawnmower, square64):
    #global lawnmower, square64
    #Horizontal moving (operations in the position i of the square64)    
    if lawnmower[2] == 'E': #If it is facing east, we add the values, because i increases when going to the right
        lawnmower[0] = add_toroidal_8(lawnmower[0], a[0])
    elif lawnmower[2] == 'W': #If it is facing west, we subtract the values, because i decreases when going to the left
        lawnmower[0] = sub_toroidal_8(lawnmower[0], a[0])
    
    #Vertical moving (operations in the position j of the square64)    
    if lawnmower[2] == 'N': #If it is facing north, we subtract the values, because j decreases when going up
        lawnmower[1] = sub_toroidal_8(lawnmower[1], a[1])
    elif lawnmower[2] == 'S': #If it is facing south, we add the values, because j increases when going down
        lawnmower[1] = add_toroidal_8(lawnmower[1], a[1])
    
    #Mow the grass in the final position
    square64[lawnmower[0], lawnmower[1]] = 0
    
    return a
    
def mow(lawnmower, square64):
    #global lawnmower, square64
    if lawnmower[2] == 'E': #If it is facing east, move to the right
        lawnmower[0] = add_toroidal_8(lawnmower[0], 1)
    elif lawnmower[2] == 'W': #If it is facing west, move to the left
        lawnmower[0] = sub_toroidal_8(lawnmower[0], 1)
    elif lawnmower[2] == 'N': #If it is facing north, move up
        
        lawnmower[1] = sub_toroidal_8(lawnmower[1], 1)
    elif lawnmower[2] == 'S': #If it is facing south, move down
        lawnmower[1] = add_toroidal_8(lawnmower[1], 1)
    
    #Mow the grass in the final position
    square64[lawnmower[0], lawnmower[1]] = 0
    
    return (0,0)
    
def rv8():
    return (random.randint(0,7), random.randint(0,7))
    
def add_toroidal_8(a, b):
    add = a + b
    if add >= 8:
        add -= 8
    return add

def sub_toroidal_8(a, b):
    sub = a - b
    if sub < 0:
        sub += 8
    return sub

def v12a(a, b):
    return (add_toroidal_12(a[0], b[0]), add_toroidal_12(a[1], b[1]))
    
def frog12(a, lawnmower, square):
    #global lawnmower, square64
    #Horizontal moving (operations in the position i of the square64)    
    if lawnmower[2] == 'E': #If it is facing east, we add the values, because i increases when going to the right
        lawnmower[0] = add_toroidal_12(lawnmower[0], a[0])
    elif lawnmower[2] == 'W': #If it is facing west, we subtract the values, because i decreases when going to the left
        lawnmower[0] = sub_toroidal_12(lawnmower[0], a[0])
    
    #Vertical moving (operations in the position j of the square64)    
    if lawnmower[2] == 'N': #If it is facing north, we subtract the values, because j decreases when going up
        lawnmower[1] = sub_toroidal_12(lawnmower[1], a[1])
    elif lawnmower[2] == 'S': #If it is facing south, we add the values, because j increases when going down
        lawnmower[1] = add_toroidal_12(lawnmower[1], a[1])
    
    #Mow the grass in the final position
    square[lawnmower[0], lawnmower[1]] = 0
    
    return a
    
def mow12(lawnmower, square):
    #global lawnmower, square64
    if lawnmower[2] == 'E': #If it is facing east, move to the right
        lawnmower[0] = add_toroidal_12(lawnmower[0], 1)
    elif lawnmower[2] == 'W': #If it is facing west, move to the left
        lawnmower[0] = sub_toroidal_12(lawnmower[0], 1)
    elif lawnmower[2] == 'N': #If it is facing north, move up
        
        lawnmower[1] = sub_toroidal_12(lawnmower[1], 1)
    elif lawnmower[2] == 'S': #If it is facing south, move down
        lawnmower[1] = add_toroidal_12(lawnmower[1], 1)
    
    #Mow the grass in the final position
    square[lawnmower[0], lawnmower[1]] = 0
    
    return (0,0)
    
def rv12():
    return (random.randint(0,11), random.randint(0,11))
    
def add_toroidal_12(a, b):
    add = a + b
    if add >= 12:
        add -= 12
    return add

def sub_toroidal_12(a, b):
    sub = a - b
    if sub < 0:
        sub += 12
    return sub

def v14a(a, b):
    return (add_toroidal_14(a[0], b[0]), add_toroidal_14(a[1], b[1]))
    
def frog14(a, lawnmower, square):
    #global lawnmower, square64
    #Horizontal moving (operations in the position i of the square64)    
    if lawnmower[2] == 'E': #If it is facing east, we add the values, because i increases when going to the right
        lawnmower[0] = add_toroidal_14(lawnmower[0], a[0])
    elif lawnmower[2] == 'W': #If it is facing west, we subtract the values, because i decreases when going to the left
        lawnmower[0] = sub_toroidal_14(lawnmower[0], a[0])
    
    #Vertical moving (operations in the position j of the square64)    
    if lawnmower[2] == 'N': #If it is facing north, we subtract the values, because j decreases when going up
        lawnmower[1] = sub_toroidal_14(lawnmower[1], a[1])
    elif lawnmower[2] == 'S': #If it is facing south, we add the values, because j increases when going down
        lawnmower[1] = add_toroidal_14(lawnmower[1], a[1])
    
    #Mow the grass in the final position
    square[lawnmower[0], lawnmower[1]] = 0
    
    return a

def mow14(lawnmower, square):
    #global lawnmower, square64
    if lawnmower[2] == 'E': #If it is facing east, move to the right
        lawnmower[0] = add_toroidal_14(lawnmower[0], 1)
    elif lawnmower[2] == 'W': #If it is facing west, move to the left
        lawnmower[0] = sub_toroidal_14(lawnmower[0], 1)
    elif lawnmower[2] == 'N': #If it is facing north, move up
        
        lawnmower[1] = sub_toroidal_14(lawnmower[1], 1)
    elif lawnmower[2] == 'S': #If it is facing south, move down
        lawnmower[1] = add_toroidal_14(lawnmower[1], 1)
    
    #Mow the grass in the final position
    square[lawnmower[0], lawnmower[1]] = 0
    
    return (0,0)
    
def rv14():
    return (random.randint(0,13), random.randint(0,13))
    
def add_toroidal_14(a, b):
    add = a + b
    if add >= 14:
        add -= 14
    return add

def sub_toroidal_14(a, b):
    sub = a - b
    if sub < 0:
        sub += 14
    return sub