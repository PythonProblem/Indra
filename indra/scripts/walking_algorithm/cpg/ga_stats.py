#!/usr/bin/python3

from deap import base, creator, tools
from cpg import CPG

import vrep
from vrepConst import simx_opmode_blocking, simx_opmode_oneshot_wait

import rospy
from indra.msg import robot_feedback, robot_input
from std_msgs.msg import Float64

from math import pi
import random
import pickle
import time
import traceback
from copy import deepcopy
import matplotlib.pyplot as plt

# Seed to reproduce results
random.seed(31)

# Global variables
robot_position = []
robot_fallen = False
start_subscribing = False
default_angles = [
    90, 0, 0, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 0, 0, 90, 90
]
min_angles = [0, 0, 0, 45, 0, 0, 45, 45, 45, 45, 0, 0, 45, 0, 0, 0, 0]
max_angles = [
    180, 180, 180, 180, 180, 90, 180, 180, 180, 180, 90, 180, 180, 180, 180,
    180, 180
]

last_gen = 1

# Constants
crossover_probability = 0.9

mutation_probability = 0.4
mutation_mu = 0.5
mutation_sigma = 0.01
mutation_alpha = 0.8

population_size = 500
max_generations = 50
tournament_size = 5

probability_change = 0.3 / max_generations


def generate_individual():
    # Create an individual to intialisze the population.
    # Individual is a list with the parameters required to create a CPG network.

    # Generate inter-oscillator weights
    weights = [random.random() * 2 - 1 for i in range(8)]

    # Generate initial bias
    shoulder_bias = (random.random() * (pi / 2))
    hip_bias = (random.random() * pi - (pi / 2))
    knee_bias = (random.random() * -1 * (pi / 2))

    #  Generate the different neuron parameters
    neuron_parameters = []
    for _ in range(5):
        # Generate t_r,t_a,a,b for each type of oscillator
        # z = t_r/t_a
        # a = z to 5
        # b = a-1 to 5
        t_r = random.random() * 10 + 0.02
        t_a = random.random() * 10 + 0.02
        z = t_r / t_a
        a = random.random() * (100 - z) + z
        b = random.random() * (101 - a) + (a - 1)
        neuron_parameters.append([t_r, t_a, b, a])

    inputs = [random.random() * 10 for i in range(9)]

    return [
        weights, shoulder_bias, hip_bias, knee_bias, neuron_parameters, inputs
    ]


def crossover_values(x, y, alpha):
    if (type(x) != list):
        x = [x]
        y = [y]

    c1 = [alpha * i + (1 - alpha) * j for i, j in zip(x, y)]
    c2 = [alpha * j + (1 - alpha) * i for i, j in zip(x, y)]

    if (len(c1) == 1):
        c1 = c1[0]
        c2 = c2[0]

    return c1, c2


def crossover(individual1, individual2):
    crossover_alpha = random.random()

    weights1, shoulder_bias1, hip_bias1, knee_bias1, neuron_parameters1, inputs1 = individual1
    weights2, shoulder_bias2, hip_bias2, knee_bias2, neuron_parameters2, inputs2 = individual2

    weights1, weights2 = crossover_values(weights1, weights2, crossover_alpha)
    shoulder_bias1, shoulder_bias2 = crossover_values(shoulder_bias1,
                                                      shoulder_bias2,
                                                      crossover_alpha)
    hip_bias1, hip_bias2 = crossover_values(hip_bias1, hip_bias2,
                                            crossover_alpha)
    knee_bias1, knee_bias2 = crossover_values(knee_bias1, knee_bias2,
                                              crossover_alpha)
    inputs1, inputs2 = crossover_values(inputs1, inputs2, crossover_alpha)

    for i in range(len(neuron_parameters1)):
        neuron_parameters1[i], neuron_parameters2[i] = crossover_values(
            neuron_parameters1[i], neuron_parameters2[i], crossover_alpha)

    temp1 = [
        weights1, shoulder_bias1, hip_bias1, knee_bias1, neuron_parameters1,
        inputs1
    ]
    temp2 = [
        weights2, shoulder_bias2, hip_bias2, knee_bias2, neuron_parameters2,
        inputs2
    ]

    for i in range(len(individual1)):
        individual1[i] = temp1[i]
        individual2[i] = temp2[i]

    return individual1, individual2


def mutate(individual):
    global mutation_mu, mutation_sigma, mutation_alpha

    weights, shoulder_bias, hip_bias, knee_bias, neuron_parameters, inputs = individual
    biases = [shoulder_bias, hip_bias, knee_bias]

    tools.mutGaussian(weights, mutation_mu, mutation_sigma, mutation_alpha)
    tools.mutGaussian(biases, mutation_mu, mutation_sigma, mutation_alpha)
    tools.mutGaussian(inputs, mutation_mu, mutation_sigma, mutation_alpha)

    for i in range(len(neuron_parameters)):
        tools.mutGaussian(neuron_parameters[i], mutation_mu, mutation_sigma,
                          mutation_alpha)

    shoulder_bias, hip_bias, knee_bias = biases

    temp = [
        weights, shoulder_bias, hip_bias, knee_bias, neuron_parameters, inputs
    ]

    for i in range(len(individual)):
        individual[i] = temp[i]

    return individual


def evaluate(individual):
    # Set the predefined parameters
    dt = 0.05  # Time step (in seconds)
    oscillator_number = 9  # Number of oscillators

    # Unpack variables
    weights, neuron_parameters, inputs = individual
    # Create the oscillator parameters
    oscillator_parameters = [
        [
            neuron_parameters[0][:3], neuron_parameters[0][:3],
            [neuron_parameters[0][3], neuron_parameters[0][3]], 0, "pacemaker"
        ],
        [
            neuron_parameters[1][:3], neuron_parameters[1][:3],
            [neuron_parameters[1][3], neuron_parameters[1][3]], 0,
            "right_hip_roll"
        ],
        [
            neuron_parameters[1][:3], neuron_parameters[1][:3],
            [neuron_parameters[1][3], neuron_parameters[1][3]], 0,
            "left_hip_roll"
        ],
        [
            neuron_parameters[2][:3], neuron_parameters[2][:3],
            [neuron_parameters[2][3], neuron_parameters[2][3]], 0,
            "right_shoulder_pitch"
        ],
        [
            neuron_parameters[2][:3], neuron_parameters[2][:3],
            [neuron_parameters[2][3], neuron_parameters[2][3]], 0,
            "left_shoulder_pitch"
        ],
        [
            neuron_parameters[3][:3], neuron_parameters[3][:3],
            [neuron_parameters[3][3], neuron_parameters[3][3]], 0,
            "right_hip_pitch"
        ],
        [
            neuron_parameters[3][:3], neuron_parameters[3][:3],
            [neuron_parameters[3][3], neuron_parameters[3][3]], 0,
            "left_hip_pitch"
        ],
        [
            neuron_parameters[4][:3], neuron_parameters[4][:3],
            [neuron_parameters[4][3], neuron_parameters[4][3]], 0,
            "right_knee_pitch"
        ],
        [
            neuron_parameters[4][:3], neuron_parameters[4][:3],
            [neuron_parameters[4][3], neuron_parameters[4][3]], 0,
            "left_knee_pitch"
        ]
    ]

    # Create the weight matrix
    weight_matrix = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [weights[0], 0, 0, 0, 0, 0, 0, 0, 0],
        [-weights[0], 0, 0, 0, 0, 0, 0, 0, 0],
        [weights[1], 0, 0, 0, 0, 0, 0, 0, 0],
        [-weights[1], 0, 0, 0, 0, 0, 0, 0, 0],
        [-weights[2], 0, 0, 0, 0, 0, 0, 0, 0],
        [weights[2], 0, 0, 0, 0, 0, 0, 0, 0],
        [-weights[3], 0, 0, 0, 0, 0, 0, 0, 0],
        [weights[3], 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    # Create CPG network
    network = CPG(oscillator_number, oscillator_parameters, weight_matrix, dt)
    feedback = [0] * oscillator_number

    angles = []
    start_time = time.time()
    # Evaluate until robot falls or times out
    for i in range(1000):
        network(inputs, feedback)
        output = network.output()
        angles.append(output[0:1])
    for i in network.get_parameters()[1]:
        print(i)
        print()
    plt.plot(angles)
    plt.show()


def send_angles(publisher, rate, output, steps):
    global default_angles, min_angles, max_angles

    output_map = [12, 3, 13, 2, 11, 4, 10, 5]

    for step in range(1, steps + 1):
        target_angles = deepcopy(default_angles)
        for i in range(8):
            target_angles[output_map[i]] += int(
                (output[i + 1] * step * 180) / (steps * pi))
            if (target_angles[output_map[i]] < min_angles[output_map[i]]):
                target_angles[output_map[i]] = min_angles[output_map[i]]
            if (target_angles[output_map[i]] > max_angles[output_map[i]]):
                target_angles[output_map[i]] = max_angles[output_map[i]]

        target_angles[6] = min(270 - target_angles[5] - target_angles[4], 180)
        target_angles[9] = min(270 - target_angles[10] - target_angles[11],
                               180)
        target_angles[7] = target_angles[3]
        target_angles[8] = target_angles[12]
        publisher.publish(target_angles)
        rate.sleep()


def calculate_fitness():
    global robot_position

    if len(robot_position) < 10:
        return -1

    start_x, start_y = robot_position[0][:2]
    end_x = robot_position[-1][0]

    distance_walked = end_x - start_x
    average_deviation = sum([pos[1] - start_y
                             for pos in robot_position]) / len(robot_position)
    average_hip_height = sum([pos[2]
                              for pos in robot_position]) / len(robot_position)
    return (distance_walked - abs(average_deviation) + average_hip_height) * 10


def robot_feedback_callback(message):
    global robot_fallen, robot_position
    position = message.hip_position

    if (start_subscribing):
        robot_position.append(position)
        if (position[2] < 0.13):
            robot_fallen = True


if __name__ == "__main__":

    # Program path to store each generation backup
    path = '/home/vishwas/Projects/RoboCup/ros_workspace/src/indra/scripts/walking_algorithm/cpg'

    # Create classes and functions for GA
    creator.create("Fitness", base.Fitness, weights=(1.0, ))
    creator.create("Individual", list, fitness=creator.Fitness)
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    # Initialize population
    f = open(path + "/ga_backup/gen_1", "rb")
    population = pickle.load(f)
    print(len(population))
    fit = max(population, key=lambda x: x.fitness.values[0])
    print(fit.fitness.values[0])
    print(fit)
    # evaluate(fit)
    # print(fit[2])
    # w1 = []
    # w2 = []
    # tr = []
    # ta = []
    # hip = []
    # knee = []
    # for i in population:
    #     w1.append(i[0][0])
    #     w2.append(i[0][1])
    #     tr.append(i[-2][0][0])
    #     ta.append(i[-2][0][1])
    #     hip.append(i[2])
    #     knee.append(i[3])
    # plt.scatter(hip, knee)
    # plt.show()
    # print(fit[0])