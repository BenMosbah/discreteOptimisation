#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
from itertools import product
import numpy as np
from pyscipopt import Model, quicksum, multidict

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def parse_input(input_data):
    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    print('Number of facilities :',facility_count)

    facilities = []
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.append(Facility(i - 1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3]))))

    customers = []
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.append(Customer(i - 1 - facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))
    return parts, facility_count, customer_count, facilities, customers

def flp(customer_count,facility_count,demand,capacity,setup_cost,shipping_cost):
    model = Model("flp")
    #y is which facilites are open
    #x is assignment
    x,y = {},{}
    # Adding Decision Variables
    for j in range(facility_count):
        # Add variable y_j to model : is facility j open or not
        y[j] = model.addVar(vtype="B", name="y(%s)"%j)
        for i in range(customer_count):
            # Add variable x_i,j to model : is client i serviced by facility j?
            x[i,j] = model.addVar(vtype="B", name="x(%s,%s)"%(i,j))
    # Adding constraints
    # Demand constraint : Every client is serviced.
    for i in range(customer_count):
        model.addCons(quicksum(x[i,j] for j in range(facility_count)) == 1, "Demand(%s)"%i)
    # Capacity constraint : For every factory. Sum of demand of customers is smaller than capacity.
    for j in range(facility_count):
        model.addCons(quicksum(x[i,j]*demand[i] for i in range(customer_count)) <= capacity[j]*y[j], "Capacity(%s)"%i)
    # Each client is serviced by exactly one facility
    #for (i,j) in x:
    #    model.addCons(x[i,j] == demand[i]*y[j], "Strong(%s,%s)"%(i,j))
    model.setObjective(
        quicksum(setup_cost[j]*y[j] for j in range(facility_count)) +
        quicksum(shipping_cost[i,j]*x[i,j] for i in range(customer_count) for j in range(facility_count)),
        "minimize")

    model.data = x,y
    return model

def get_assignment_pyscipopt(edges,customer_count):
    assignment = ['inv']*customer_count
    for customer,facility in edges:
        assignment[customer] = facility
    return assignment

def calculate_obj_from_assignment(setup_cost,facilities_open,assignment,customer_count,shipping_cost):
    sc = np.dot(facilities_open,setup_cost)
    ac = 0
    for customer in range(customer_count):
        ac += shipping_cost[(customer,assignment[customer])]
    return sc + ac

def solve_it(input_data):
    # Reading input
    parts, facility_count, customer_count, facilities, customers = parse_input(input_data)
    if facility_count<2000:
        cartesian_prod = list(product(range(customer_count), range(facility_count)))
        shipping_cost = {(c, f): length(customers[c].location, facilities[f].location) for c, f in cartesian_prod}
        # setup cost is the list of setup costs
        setup_cost = [f.setup_cost for f in facilities]
        # demand is the list of demands by customers
        demand = [c.demand for c in customers]
        # capacity is the list of capacities by facility
        capacity = [c.capacity for c in facilities]
        # Using pyscipopt
        model = flp(customer_count, facility_count, demand, capacity, setup_cost, shipping_cost)
        # setting a time limit of 20min
        model.setParam('limits/time', 20*60)
        model.optimize()
        EPS = 1.e-6
        x = model.data[0]
        y = model.data[1]
        edges = [(i, j) for (i, j) in x if model.getVal(x[i, j]) > EPS]
        facilities_chosen = [j for j in y if model.getVal(y[j]) > EPS]
        assignment = get_assignment_pyscipopt(edges, customer_count)
        facilities_open = [1 if i in facilities_chosen else 0 for i in range(facility_count)]

        # calculate the cost of the solution
        obj = calculate_obj_from_assignment(setup_cost,facilities_open,assignment,customer_count,shipping_cost)
        solution = assignment
        optimal = 1 if model.getStatus() == 'optimal' else 0

        # prepare the solution in the specified output format
        output_data = '%.2f' % obj + ' ' + str(optimal) + '\n'
        output_data += ' '.join(map(str, solution))

        return output_data
    else:
        #Method crashes for 2000 facilities so let's just return None
        return


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

