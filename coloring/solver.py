#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

def parse_input(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    return node_count, edge_count, edges

def naive(node_count,degrees,nodes_sorted_by_degree,adjacency_dic):
    # Color each node with a new color that is available.
    # 0 is colored with the color 0 to avoid symmetries.
    coloring = {nodes_sorted_by_degree[0]:0}
    for i in range(1,node_count):
        node = nodes_sorted_by_degree[i]
        colors_used =  {*coloring.values()}
        # get the neighbors
        neighbors = adjacency_dic[node]
        colors_of_neighbors = set()
        for neighbor in neighbors:
            if neighbor in coloring:
                colors_of_neighbors.add(coloring[neighbor])
        available_colors_among_the_ones_already_used = set(colors_used)-set(colors_of_neighbors)
        if len(available_colors_among_the_ones_already_used)!=0:
            color_to_assign = min(available_colors_among_the_ones_already_used)
        else:
            color_to_assign = max(colors_used)+1
        coloring[node] = color_to_assign
    color_count = len(np.unique([*coloring.values()]))
    coloring_sorted = {k:coloring[k] for k in sorted([*coloring.keys()])}
    solution = [*coloring_sorted.values()]
    return color_count,solution


def solve_dsatur(node_count, degrees, nodes_sorted_by_degree, adjacency_dic):
    color_count = solve_dsatur
    # 0 is colored with the color 0 to avoid symmetries.
    coloring = {nodes_sorted_by_degree[0]: 0}
    # Beginning is the same as naive. Ok all good.
    uncolored_nodes = list(nodes_sorted_by_degree[1:])

    def get_colored(list_of_nodes, coloring):
        all_colored = [*coloring.keys()]
        return [el for el in list_of_nodes if el in all_colored]

    def choose_next_node(uncolored_nodes, coloring):
        degrees_of_saturation = {k: len(get_colored(adjacency_dic[k], coloring)) for k in uncolored_nodes}
        max_degrees_of_saturation = max(degrees_of_saturation.values())
        candidates = {k: degrees[k] for k in degrees_of_saturation if
                      degrees_of_saturation[k] == max_degrees_of_saturation}
        max_degree_in_candidates = max(candidates.values())
        final_candidates = [k for k in candidates if candidates[k] == max_degree_in_candidates]
        return final_candidates[0]

    while len(uncolored_nodes) > 0:
        node = choose_next_node(uncolored_nodes, coloring)
        colors_used = {*coloring.values()}
        # get the neighbors
        neighbors = adjacency_dic[node]
        colors_of_neighbors = set()
        for neighbor in neighbors:
            if neighbor in coloring:
                colors_of_neighbors.add(coloring[neighbor])
        available_colors_among_the_ones_already_used = set(colors_used) - set(colors_of_neighbors)
        if len(available_colors_among_the_ones_already_used) != 0:
            color_to_assign = min(available_colors_among_the_ones_already_used)
        else:
            color_to_assign = max(colors_used) + 1
        coloring[node] = color_to_assign
        uncolored_nodes.remove(node)
    color_count = len(np.unique([*coloring.values()]))
    coloring_sorted = {k: coloring[k] for k in sorted([*coloring.keys()])}
    solution = [*coloring_sorted.values()]
    return color_count, solution

# Backtracking code
def check_constraint(considered_node,adjacency_dic,candidate_color,coloring):
    # For a considered node, will get all the neighbors, get the colors they are colored in and check if there
    # is a conflict with the candidate color
    neighbors = adjacency_dic[considered_node]
    colors_of_neighbors = set()
    for neighbor in neighbors:
        if neighbor in coloring:
            colors_of_neighbors.add(coloring[neighbor])
    return False if candidate_color in colors_of_neighbors else True


def explore_tree(considered_node_index, nodes_sorted_by_degree, coloring, adjacency_dic, max_colors=100):
    # Base Case : if all vertices are colored
    node_count = len(nodes_sorted_by_degree)
    if np.all([n in coloring for n in range(node_count)]):
        return coloring
    # Else look at the next node
    considered_node = nodes_sorted_by_degree[considered_node_index]
    # print(considered_node_index)
    # print(nodes_sorted_by_degree)
    # If base case was not activated, iterate through the colors, try to color if possible
    # print(coloring)
    for candidate_color in range(max_colors):
        coloring_is_valid = check_constraint(considered_node, adjacency_dic, candidate_color, coloring)
        if coloring_is_valid:
            # Color with the color that is valide
            coloring[considered_node] = candidate_color
            # Check further down the tree
            if explore_tree(considered_node_index + 1, nodes_sorted_by_degree, coloring, adjacency_dic, max_colors):
                return coloring
            # Un-color (this is backtracking)
            del coloring[considered_node]
    # This would mean that all colors were considered
    return False

def solve_backtracking(node_count,degrees,nodes_sorted_by_degree,adjacency_dic,upper_bound):
    coloring = {nodes_sorted_by_degree[0]:0}
    coloring = explore_tree(1,nodes_sorted_by_degree,coloring,adjacency_dic,upper_bound-1)
    if coloring:
        coloring_sorted = {k: coloring[k] for k in sorted([*coloring.keys()])}
        solution = [*coloring_sorted.values()]
        return solution
    else :
        return False

def solve_it(input_data):
    node_count, edge_count, edges = parse_input(input_data)
    print(node_count)
    print(edge_count)
    adjacency_dic = {k: set() for k in range(node_count)}
    for edge in edges:
        node_1 = edge[0]
        node_2 = edge[1]
        adjacency_dic[node_1].add(node_2)
        adjacency_dic[node_2].add(node_1)
    degrees = [len(adjacency_dic[k]) for k in adjacency_dic]
    nodes_sorted_by_degree = np.argsort(degrees)[::-1]
    if node_count<=500:
        color_count_dsatur, solution_dsatur = solve_dsatur(node_count, degrees, nodes_sorted_by_degree, adjacency_dic)
        color_count_naive_sorted, solution_naive_sorted = naive(node_count, degrees, nodes_sorted_by_degree, adjacency_dic)

        if color_count_naive_sorted>color_count_dsatur:
            print('DSAtur was better.')
            color_count = color_count_dsatur
            solution = solution_dsatur
        else:
            print('Naive was better.')
            color_count = color_count_naive_sorted
            solution = solution_naive_sorted
    else :
        print('Too many nodes {} went with naive.'.format(node_count))
        color_count, solution = naive(node_count, degrees, nodes_sorted_by_degree, adjacency_dic)
    best_color_count_found = color_count
    print('Best color count found is {}'.format(color_count))
    optimal = 0
    if node_count <=70:

        solution_bt = True
        best_solution_so_far = solution
        count_iterations = 0
        while solution_bt:
            optimal = 1
            print('Backtracking will be successful with {} colors. Taking one more off.'.format(best_color_count_found))
            solution_bt = solve_backtracking(node_count, degrees, nodes_sorted_by_degree,
                                          adjacency_dic, best_color_count_found)
            if solution_bt:
                best_color_count_found = best_color_count_found-1
                best_solution_so_far = solution_bt
            count_iterations += 1
            if count_iterations >= 2:
                print('Already went twice through bt. Stopping.')
                solution_bt = False
        if solution_bt:
            optimal = 0
        solution = best_solution_so_far

    print('Backtracking found a solution with {} colors !'.format(best_color_count_found))

    # prepare the solution in the specified output format
    output_data = str(best_color_count_found) + ' ' + str(optimal) + '\n'
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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

