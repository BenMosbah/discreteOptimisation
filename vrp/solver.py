#!/usr/bin/python
# -*- coding: utf-8 -*-
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
from collections import namedtuple
import numpy as np

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)


def parse_data(input_data):
    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    customers = []
    for i in range(1, customer_count + 1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i - 1, int(parts[0]), float(parts[1]), float(parts[2])))

    # the depot is always the first customer in the input
    depot = customers[0]
    print('Customer count :',customer_count)
    print('Vehicle count :',vehicle_count)
    return customer_count, vehicle_count, vehicle_capacity, customers, depot

def get_distance_matrix(customer_count,customers):
    distance_matrix = np.zeros((customer_count,customer_count))
    for i in range(customer_count):
        for j in range(i,customer_count):
            d = length(customers[i],customers[j])
            distance_matrix[i,j] = d
            distance_matrix[j,i] = d
    return distance_matrix

def create_data_model(customer_count,vehicle_count,vehicle_capacity,customers):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = get_distance_matrix(customer_count,customers)
    data['num_vehicles'] = vehicle_count
    data['demands'] = [c.demand for c in customers]
    data['vehicle_capacities'] = [vehicle_capacity]*vehicle_count
    data['depot'] = 0
    return data

def get_obj_and_routes(solution,routing,manager,vehicle_count,customers):
    obj = solution.ObjectiveValue()
    vehicle_tours = [[] for i in range(vehicle_count)]
    for vehicle_id in range(vehicle_count):
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            index = solution.Value(routing.NextVar(index))
            vehicle_tours[vehicle_id].append(customers[node_index])
        vehicle_tours[vehicle_id].append(customers[manager.IndexToNode(index)])
    return obj,vehicle_tours
def google_or(customer_count,vehicle_count,vehicle_capacity,customers):
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model(customer_count,vehicle_count,vehicle_capacity,customers)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(1)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    #if solution:
    #    print_solution(data, manager, routing, solution)
    return routing,solution,manager
def solve_it(input_data):
    customer_count, vehicle_count, vehicle_capacity, customers, depot = parse_data(input_data)


    # Using Google's OR solver !
    routing, solution, manager = google_or(customer_count, vehicle_count, vehicle_capacity, customers)
    obj, vehicle_tours = get_obj_and_routes(solution, routing, manager, vehicle_count,customers)
    #print(obj)
    # checks that the number of customers served is correct
    #assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1

    # calculate the cost of the solution; for each vehicle the length of the route
    obj = 0
    for v in range(0, vehicle_count):
        vehicle_tour = vehicle_tours[v]
        if len(vehicle_tour) > 0:
            obj += length(depot,vehicle_tour[0])
            for i in range(0, len(vehicle_tour)-1):
                obj += length(vehicle_tour[i],vehicle_tour[i+1])
            obj += length(vehicle_tour[-1],depot)

    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(vehicle_count):
        outputData +=  ' '.join([str(customer.index) for customer in vehicle_tours[v]]) + '\n'

    return outputData


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

