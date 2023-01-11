#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
import time
Item = namedtuple("Item", ['index', 'value', 'weight'])


def parse_input(input_data):
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i - 1, int(parts[0]), int(parts[1])))

    return items, capacity, item_count


def greedy(items, capacity):
    items = np.array(items)
    items_ranked_by_value_by_weight = items[np.argsort([-i[1] / i[2] for i in items])]
    i = 0
    free_space = capacity
    estimate = 0
    selected = []
    while free_space != 0 and i < len(items):
        item = items_ranked_by_value_by_weight[i]
        i += 1
        value = item[1]
        weight = item[2]
        index = item[0]
        if weight < free_space:
            estimate += value
            free_space -= weight
            selected.append(index)
        else:
            continue

    return estimate, selected

def dp(n, capacity, dp_array, items):
    wt = [i[2] for i in items]
    val = [i[1] for i in items]

    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                dp_array[i][w] = 0
            elif wt[i - 1] <= w:
                dp_array[i][w] = max(val[i - 1]
                                     + dp_array[i - 1][w - wt[i - 1]],
                                     dp_array[i - 1][w])
            else:
                dp_array[i][w] = dp_array[i - 1][w]

    return dp_array[n][capacity]

def back_track(dp_array,remainder,result_back_tracking,items):
    weight_last = 0
    while remainder!=0:
        last = np.where(dp_array[:,-1-weight_last]==remainder)[0][0]
        value_last = items[last-1][1]
        weight_last += items[last-1][2]
        remainder = remainder - value_last
        result_back_tracking.append(last-1)


def solve_it(input_data, cutoff=300, capacity_override=None):
    items, capacity, item_count = parse_input(input_data)
    if capacity_override: capacity = capacity_override
    optimal = 1
    # if  <

    # DP solution
    # Try DP solution if takes less than 10min
    dp_array = np.zeros((item_count + 1, 30000 + 1))
    # Test run
    start_time = time.time()
    dp(item_count, 30000, dp_array, items)
    run_time = (time.time() - start_time)
    estimated_runtime = run_time * capacity / 30000
    if estimated_runtime > cutoff:
        print('Would take too long to run dp solution. Resorting to greedy')
        greedy_solution, selected = greedy(items, capacity)
        optimal = 0
        solution = greedy_solution
    else:

        dp_array = np.zeros((item_count + 1, capacity + 1))
        dp_solution = dp(item_count, capacity, dp_array, items)

        result_back_tracking = []
        back_track(dp_array, dp_solution, result_back_tracking,items)
        solution = dp_solution
        selected = result_back_tracking

    to_return = np.zeros(item_count)
    to_return[selected] = 1
    to_return = to_return.astype(int)

    output_data = str(int(solution)) + ' ' + str(optimal) + '\n'
    output_data += ' '.join(map(str, to_return))
    return output_data

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

