#******************************************************************************************************#
# Class Name        : CPG                                                                              #
# Class Description : CPG(Central Pattern Generator) network containing multiple matsuoka oscillators. #
#                     Provides oscillatory output for every oscillator, which is used as angles.       #
#                                                                                                      #
# Authors           : Vishwas N S, Karthik K Bhat                                                      #
#******************************************************************************************************#

# Import required libraries
from matsuoka import Oscillator
import numpy as np
import time
import matplotlib.pyplot as plt
from copy import deepcopy


# Class definition of an CPG network
class CPG:

    # Initialize the CPG network with the number of oscillators requires.
    def __init__(self, number, oscillator_parameters, weight_matrix, dt):
        self.oscillator_number = number  # Number of oscillators
        self.weight_matrix = deepcopy(
            weight_matrix)  # Inter oscillator weights
        self.dt = dt  # Time step
        self.oscillator_objects = [
            Oscillator(*oscillator_parameters[i], self.dt)
            for i in range(self.oscillator_number)
        ]  # List of oscillator objects
        self.oscillator_outputs = [0 for _ in range(self.oscillator_number)
                                   ]  # List of oscillator outputs

    # Retrieve parameters of the all oscillators.
    # Returns a list
    def get_parameters(self):
        oscillator_parameters = [
            osc.get_parameters() for osc in self.oscillator_objects
        ]
        return [
            self.oscillator_number, oscillator_parameters, self.weight_matrix,
            self.dt
        ]

    # Change the netowork parameters during runtime.
    def set_parameters(self, oscillator_parameters, weight_matrix):
        for osc in range(self.oscillator_number):
            self.oscillator_objects[osc].set_parameters(
                oscillator_parameters[osc])
        self.weight_matrix = deepcopy(weight_matrix)

    # Reset the output of required oscillators.
    def reset_oscillators(self, oscillator_list):
        for n, choice in enumerate(oscillator_list):
            if (choice):
                self.oscillator_objects[n].reset_oscillator()
                self.oscillator_outputs[n] = 0

    # Return the output of the oscillators without calculating the output for the next time step.
    # Returns a list.
    def output(self):
        self.oscillator_outputs = [
            osc.output() for osc in self.oscillator_objects
        ]
        return self.oscillator_outputs

    # Calculate the output of the oscillators for the next time step.
    def __call__(self, tonic_inputs, feedbacks):

        # Iterate through all the oscillators and find the outputs.
        for osc, weights, tonic_in, feed in zip(self.oscillator_objects,
                                                self.weight_matrix,
                                                tonic_inputs, feedbacks):
            # print(osc.name, weights, tonic_in, feed)
            osc(np.array(self.oscillator_outputs), np.array(weights), tonic_in,
                feed)

        return self.output()


# Create a CPG network with dummy parameters to test the program.
if __name__ == "__main__":

    oscillator_number = 3
    oscillator_1 = [[0.5, 1, 5], [0.5, 1, 5], [3, 3], 0, 1, "osc_1"]
    oscillator_2 = [[0.5, 1, 5], [0.5, 1, 5], [3, 3], 0, 1, "osc_2"]
    oscillator_3 = [[0.5, 1, 5], [0.5, 1, 5], [3, 3], 0, 1, "osc_3"]
    weight_matrix = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    dt = 0.05

    network = CPG(oscillator_number,
                  [oscillator_1, oscillator_2, oscillator_3], weight_matrix,
                  dt)
    x = []
    y = []

    main = [
        oscillator_number,
    ]

    for i in range(100):
        x.append(i * dt)
        y.append(network([3, 3, 3], [0, 0, 0]))

    plt.plot(x, y)
    plt.show()
