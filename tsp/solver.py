#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
from copy import copy
import numpy as np


Point = namedtuple("Point", ['x', 'y','index'])

def parse_input(input_data):
    lines = input_data.split('\n')
    nodeCount = int(lines[0])
    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1]),i-1))
    return nodeCount,points

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def get_objective(nodeCount,points,solution):
    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])
    return obj

# I do the computation of distances once and for all
def get_distance_dictionary(nodeCount, points):
    # returns dictionary d. d[node_index] gives a dictionary with next_node as key and distance as value.

    # fromkeys ==> all keys will point to the same object so can't be a mutable object like a dictionary...
    # distance_dictionary = dict.fromkeys(range(nodeCount),dict())
    # Use dict comprehension instead
    distance_dictionary = {n: {} for n in range(nodeCount)}

    for p in points:
        for next_p in points[p.index + 1:]:
            d = length(p, next_p)
            distance_dictionary[p.index][next_p.index] = d
            distance_dictionary[next_p.index][p.index] = d

    # once all distances calculated. Just sort
    for p in points:
        sorted_distances = sorted(distance_dictionary[p.index].items(), key=lambda x: x[1])
        distance_dictionary[p.index] = dict(sorted_distances)
    return distance_dictionary


## Dictionary seems to be memory inefficient (don't know why !) Let's try with an array
def get_distance_matrix(nodeCount, points):
    # returns dictionary d. d[node_index] gives a dictionary with next_node as key and distance as value.

    # fromkeys ==> all keys will point to the same object so can't be a mutable object like a dictionary...
    # distance_dictionary = dict.fromkeys(range(nodeCount),dict())
    # Use dict comprehension instead
    distance_matrix = np.zeros((nodeCount, nodeCount))

    for p in points:
        for next_p in points[p.index + 1:]:
            d = length(p, next_p)
            distance_matrix[p.index, next_p.index] = d
            distance_matrix[next_p.index, p.index] = d

    return distance_matrix



#def get_closest_unvisited_node(current_node_index,path,distance_dictionary):
#    closest_nodes = [*distance_dictionary[current_node_index]]
#    i = 0
#    candidate = closest_nodes[0]
#    while i<len(closest_nodes):
#        candidate = closest_nodes[i]
#        if candidate not in path :
#            break
#        else:
#            i+=1
#    return candidate

def get_closest_unvisited_node(current_node_index,path,distance_matrix):
    closest_nodes = np.argsort(distance_matrix[current_node_index])[1:]
    i = 0
    candidate = closest_nodes[0]
    while i<len(closest_nodes):
        candidate = closest_nodes[i]
        if candidate not in path :
            break
        else:
            i+=1
    return candidate


def greedy_solution(nodeCount, distance_dictionary):
    path = [0]
    current_node_index = 0
    while len(path) < nodeCount:
        next_node_index = get_closest_unvisited_node(current_node_index, path, distance_dictionary)
        path.append(next_node_index)
        current_node_index = next_node_index
    return path

def cost_change(distance_dictionary, n1, n2, n3, n4):
    return distance_dictionary[n1][n3] + distance_dictionary[n2][n4] - distance_dictionary[n1][n2] - distance_dictionary[n3][n4]


def two_opt(route, distance_dictionary,nodeCount,points):
    best = copy(route)
    improved = True
    while improved:
        improved = False
        old_value = get_objective(nodeCount,points,best)
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                if cost_change(distance_dictionary, best[i - 1], best[i], best[j - 1], best[j]) < 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
                    if improved and nodeCount>30000:
                        improved = False
        new_value = get_objective(nodeCount,points,best)
        improvement = 1-new_value/old_value
        print('Improvement is {:.2f}'.format(improvement))
        if nodeCount>1000 and improvement<0.01:
            print('Improvement was less than 1%. Getting out')
            break
        #best = route
    return best

##############3-opt
def reverse_segment_if_better(tour, i, j, k,distance_dictionary):
    """If reversing tour[i:j] would make the tour shorter, then do it."""
    # Given tour [...A-B...C-D...E-F...]
    A, B, C, D, E, F = tour[i-1], tour[i], tour[j-1], tour[j], tour[k-1], tour[k % len(tour)]
    d0 = distance_dictionary[A, B] + distance_dictionary[C, D] + distance_dictionary[E, F]
    d1 = distance_dictionary[A, C] + distance_dictionary[B, D] + distance_dictionary[E, F]
    d2 = distance_dictionary[A, B] + distance_dictionary[C, E] + distance_dictionary[D, F]
    d3 = distance_dictionary[A, D] + distance_dictionary[E, B] + distance_dictionary[C, F]
    d4 = distance_dictionary[F, B] + distance_dictionary[C, D] + distance_dictionary[E, A]

    if d0 > d1:
        tour[i:j] = reversed(tour[i:j])
        return -d0 + d1
    elif d0 > d2:
        tour[j:k] = reversed(tour[j:k])
        return -d0 + d2
    elif d0 > d4:
        tour[i:k] = reversed(tour[i:k])
        return -d0 + d4
    elif d0 > d3:
        tmp = tour[j:k] + tour[i:j]
        tour[i:k] = tmp
        return -d0 + d3
    return 0

def three_opt(tour_og,distance_dictionary):
    """Iterative improvement based on 3 exchange."""
    tour = copy(tour_og)
    if len(tour)>600:
        return tour
    else:
        while True:
            delta = 0
            for (a, b, c) in all_segments(len(tour)):
                delta += reverse_segment_if_better(tour, a, b, c,distance_dictionary)
            if delta >= 0:
                break
        return tour

def all_segments(n: int):
    """Generate all segments combinations"""
    return ((i, j, k)
        for i in range(n)
        for j in range(i + 2, n)
        for k in range(j + 2, n + (i > 0)))


def solve_it(input_data):
    nodeCount,points = parse_input(input_data)
    print(nodeCount)
    #distance_dictionary = get_distance_dictionary(nodeCount, points)
    distance_matrix = get_distance_matrix(nodeCount, points)
    solution_greedy = greedy_solution(nodeCount, distance_matrix)
    obj_greedy = get_objective(nodeCount,points,solution_greedy)
    solution_two_opt = two_opt(solution_greedy, distance_matrix,nodeCount,points)
    obj_two_opt = get_objective(nodeCount,points,solution_two_opt)
    solution_three_opt = three_opt(solution_two_opt, distance_matrix)
    obj_three_opt = get_objective(nodeCount,points,solution_three_opt)

    solution = solution_three_opt
    obj = obj_three_opt
    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

