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

# Constants
last_gen = None

crossover_probability = 0.9

mutation_probability = 0.4
mutation_mu = 0
mutation_sigma = 0.1
mutation_alpha = 0.8

population_size = 500
max_generations = 50
tournament_size = 5

probability_change = 0.3 / max_generations


def generate_individual():
    # Create an individual to intialisze the population.
    # Individual is a list with the parameters required to create a CPG network.

    # Generate inter-oscillator weights
    weights = [random.random() for i in range(4)]

    #  Generate the different neuron parameters
    neuron_parameters = []
    for _ in range(5):
        # Generate t_r,t_a,a,b for each type of oscillator
        # z = t_r/t_a
        # a = z to 10
        # b = a-1 to 10
        t_r = random.random() * 5 + 0.02
        t_a = random.random() * 5 + 0.02
        z = t_r / t_a
        a = random.random() * (10 - z) + z
        b = random.random() * (11 - a) + (a - 1)
        neuron_parameters.append([t_r, t_a, b, a])

    pace_input = 1
    hip_roll_input = random.random()
    hip_pitch_input = random.random() * 3
    shoulder_pitch_input = random.random() * 3
    knee_pitch_input = random.random() * 3

    inputs = [
        pace_input, hip_roll_input, hip_roll_input, shoulder_pitch_input,
        shoulder_pitch_input, hip_pitch_input, hip_pitch_input,
        knee_pitch_input, knee_pitch_input
    ]

    return [weights, neuron_parameters, inputs]


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

    weights1, neuron_parameters1, inputs1 = individual1
    weights2, neuron_parameters2, inputs2 = individual2

    weights1, weights2 = crossover_values(weights1, weights2, crossover_alpha)
    inputs1, inputs2 = crossover_values(inputs1, inputs2, crossover_alpha)

    for i in range(len(neuron_parameters1)):
        neuron_parameters1[i], neuron_parameters2[i] = crossover_values(
            neuron_parameters1[i], neuron_parameters2[i], crossover_alpha)

    temp1 = [weights1, neuron_parameters1, inputs1]
    temp2 = [weights2, neuron_parameters2, inputs2]

    for i in range(len(individual1)):
        individual1[i] = temp1[i]
        individual2[i] = temp2[i]

    return individual1, individual2


def mutate(individual):
    global mutation_mu, mutation_sigma, mutation_alpha

    weights, neuron_parameters, inputs = individual

    tools.mutGaussian(weights, mutation_mu, mutation_sigma, mutation_alpha)
    tools.mutGaussian(inputs, mutation_mu, mutation_sigma, mutation_alpha)

    for i in range(len(weights)):
        if (weights[i] > 1):
            weights[i] = 1
        if (weights[i] < -1):
            weights[i] = -1

    for i in range(len(neuron_parameters)):
        tools.mutGaussian(neuron_parameters[i], mutation_mu, mutation_sigma,
                          mutation_alpha)

    temp = [weights, neuron_parameters, inputs]

    for i in range(len(individual)):
        individual[i] = temp[i]

    return individual


def evaluate(individual, sim, scene, rate, pub):
    # Evaluate a given individual by generating the CPG network and running the simulations
    global robot_position, robot_fallen, start_subscribing

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
        [weights[1], 0, 0, 0, 0, 0, 0, 0, 0],
        [weights[2], 0, 0, 0, 0, 0, 0, 0, 0],
        [weights[3], 0, 0, 0, 0, 0, 0, 0, 0],
        [0, weights[4], 0, 0, 0, 0, 0, 0, 0],
        [0, 0, weights[5], 0, 0, 0, 0, 0, 0],
        [0, weights[6], 0, 0, 0, 0, 0, 0, 0],
        [0, 0, weights[7], 0, 0, 0, 0, 0, 0],
    ]

    # Create CPG network
    network = CPG(oscillator_number, oscillator_parameters, weight_matrix, dt)
    feedback = [0] * oscillator_number

    # Clear robot position array
    robot_position = []
    robot_fallen = False

    # Start the simulation and wait for the robot to settle
    vrep.simxStartSimulation(sim, simx_opmode_oneshot_wait)
    time.sleep(1)

    # Set initial pose slowly
    output = network.output()
    send_angles(pub, rate, output, 50)
    time.sleep(2)
    start_subscribing = True
    start_time = time.time()
    x = []

    # Evaluate until robot falls or times out
    while (time.time() - start_time <= 20):
        network(inputs, feedback)
        output = network.output()
        x.append(output)
        send_angles(pub, rate, output, 1)
    duration = time.time() - start_time

    start_subscribing = False
    vrep.simxStopSimulation(sim, simx_opmode_oneshot_wait)
    time.sleep(1)
    fitness = calculate_fitness(duration)
    print(fitness)
    plt.plot(x)
    plt.legend([str(i) for i in range(len(x))])
    plt.show()
    return (fitness, )


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


def calculate_fitness(duration):
    global robot_position, robot_fallen

    fit = 0

    if len(robot_position) < 10:
        return -1

    start_x, start_y = robot_position[0][:2]
    end_x = robot_position[-1][0]

    distance_walked = end_x - start_x
    average_deviation = sum([pos[1] - start_y
                             for pos in robot_position]) / len(robot_position)
    average_hip_height = sum([pos[2]
                              for pos in robot_position]) / len(robot_position)
    # return (distance_walked - abs(average_deviation) + average_hip_height) * 10
    fit = distance_walked
    if (robot_fallen):
        fit -= 0.33
    return fit


def robot_feedback_callback(message):
    global robot_fallen, robot_position
    position = message.hip_position

    if (start_subscribing):
        robot_position.append(position)
        if (position[2] < 0.13):
            robot_fallen = True


if __name__ == "__main__":
    vrep.simxFinish(-1)

    # Program path to store each generation backup
    path = '/home/vishwas/Projects/RoboCup/ros_workspace/src/indra/scripts/walking_algorithm/cpg'

    # Connect to simulation
    scene_path = '/home/vishwas/Projects/RoboCup/vrep_related/scenes/indra.ttt'
    sim = vrep.simxStart('127.0.0.1', 19997, True, False, 5000, 5)
    scene = vrep.simxLoadScene(sim, scene_path, 1, simx_opmode_blocking)

    # Setup ROS for communication
    rospy.init_node('ga', anonymous=False)
    rate = rospy.Rate(20)
    publisher = rospy.Publisher('input', robot_input, queue_size=10)
    subscriber = rospy.Subscriber("feedback",
                                  robot_feedback,
                                  robot_feedback_callback,
                                  queue_size=10)

    # Create classes and functions for GA
    creator.create("Fitness", base.Fitness, weights=(1.0, ))
    creator.create("Individual", list, fitness=creator.Fitness)
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate",
                     evaluate,
                     sim=sim,
                     scene=scene,
                     rate=rate,
                     pub=publisher)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    # Initialize population
    if (last_gen is None):
        start_gen = 0
        population = toolbox.population(n=population_size)
    else:
        start_gen = last_gen
        with open(path + "/ga_backup/gen_" + str(last_gen), "rb") as backup:
            population = pickle.load(backup)
        mutation_probability -= start_gen * probability_change

    # Store the best individual ever found
    best_individual = None

    def test_inidividual():
        weights = [1, -1, 1, -1, -1, -1, 1, 1]

        neuron_parameters = []
        for _ in range(5):
            # Generate t_r,t_a,a,b for each type of oscillator
            # z = t_r/t_a
            # a = z to 5
            # b = a-1 to 5
            t_r = 1
            t_a = 2
            a = 2.5
            b = 3.5
            neuron_parameters.append([t_r, t_a, b, a])

        inputs = [1, 0.45, 0.45, 1, 1, 1, 1, 1, 1]

        return [weights, neuron_parameters, inputs]

    individual = test_inidividual()
    toolbox.evaluate(individual)
    vrep.simxStopSimulation(sim, simx_opmode_oneshot_wait)
    exit()

    # Run GA "max_generation" times
    try:
        for gen in range(start_gen, max_generations):

            print("Generation", gen)
            # Evaluate all individuals in the current generation
            for individual in population:
                if (not individual.fitness.valid):
                    individual.fitness.values = toolbox.evaluate(individual)

            print("Updating best")
            # Update best individual
            best = max(population, key=lambda x: x.fitness.values[0])
            if (best_individual is None or best.fitness.values[0] >
                    best_individual.fitness.values[0]):
                best_individual = toolbox.clone(best)

            print("Logging")
            # Log the entire generation
            with open(path + "/ga_backup/gen_" + str(gen), "wb") as backup:
                pickle.dump(population, backup)

            print("Selection")
            # Select well performing individuals
            selected_individuals = toolbox.select(population, population_size)
            population = [
                toolbox.clone(individual)
                for individual in selected_individuals
            ]

            print("Crossover")
            # Crossover selected individuals to create the next generation
            for i in range(0, population_size - 1, 2):
                if (random.random() < crossover_probability):
                    population[i], population[i + 1] = crossover(
                        population[i], population[i + 1])
                    del population[i].fitness.values
                    del population[i + 1].fitness.values

            print("Mutation")
            # Mutate random individuals
            for i in range(population_size):
                if (random.random() < mutation_probability):
                    population[i] = mutate(population[i])
                    if (population[i].fitness.valid):
                        del population[i].fitness.values

            print("Best fitness", best_individual.fitness.values[0])

            # Change probabilities
            mutation_probability -= probability_change

        print(best_individual)

    except Exception as e:
        traceback.print_exc()

    finally:
        # Stop simulation if running
        vrep.simxStopSimulation(sim, simx_opmode_oneshot_wait)

        # Store latest generation
        with open(path + "/ga_backup/crash", "wb") as backup:
            pickle.dump(population, backup)
            pickle.dump(best_individual, backup)
