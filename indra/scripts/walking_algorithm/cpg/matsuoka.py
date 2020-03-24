#*****************************************************************************************#
# Class Name        : Oscillator                                                          #
# Class Description : Model of a Matsuoka Oscillator.                                     #
#                     Contains two different neurons.                                     #
#                     Difference in the output of neurons is given as output.             #
#                                                                                         #
# Authors           : Vishwas N S, Karthik K Bhat                                         #
#*****************************************************************************************#

# Import required libraries
import numpy as np
from neuron import Neuron
import matplotlib.pyplot as plt


# Class definition of an Oscillator
class Oscillator:

    # Initialise the oscillator with two neurons, inter-neuron weights, output bias and a name.
    # The two neurons (flexor and extensor) work against each other to provide an oscillatory output.
    # Output of the oscillator is the difference between the neurons.
    # A bias is added to the output to generate oscillations around a different value.
    def __init__(self, extensor_parameters, flexor_parameters, weight_vector,
                 output_bias, output_gain, name, dt):

        # Initialise the extensor and the flexor neurons.
        self.extensor = Neuron(*extensor_parameters, dt)
        self.flexor = Neuron(*flexor_parameters, dt)

        # Save internal parameters of the oscillator
        self.weight_vector = weight_vector  # Contains the inter-neuron weights
        self.dt = dt  # Time step (usually 0.1)
        self.output_bias = output_bias  # Bias added to oscillator output
        self.output_gain = output_gain  # Gain multiplied to oscillator output
        self.name = name  # Name for identification

    # Retrieve values of the oscillators's internal parameters.
    # Returns a list
    def get_parameters(self):
        extensor_parameters = self.extensor.get_parameters()
        flexor_parameters = self.flexor.get_parameters()

        return [
            extensor_parameters, flexor_parameters, self.weight_vector,
            self.output_bias, self.output_gain, self.name
        ]

    # Change the oscillators's internal parameters during runtime.
    def set_parameters(self, extensor_parameters, flexor_parameters,
                       weight_vector, output_bias, name):
        self.extensor.set_parameters(*extensor_parameters)
        self.flexor.set_parameters(*flexor_parameters)
        self.weight_vector = weight_vector
        self.output_bias = output_bias
        self.output_gain = output_gain
        self.name = name

    # Reset the output of the oscillator.
    def reset_oscillator(self):
        self.extensor.reset_neuron()
        self.flexor.reset_neuron()

    # Return the output of the oscillator without calculating the output for the next time step.
    # Returns a floating point value
    def output(self):
        flexor_out = self.flexor.output()
        extensor_out = self.extensor.output()
        return (extensor_out - flexor_out)*self.output_gain + self.output_bias

    # Calculate the output of the oscillator for the next time step by calculating the neuron's output.
    def __call__(self, oscillator_inputs, oscillator_weights, tonic_input,
                 feedback):

        flexor_out = self.flexor.output()

        # Variables to hold sensory input to the two neurons.
        extensor_in = flexor_out * self.weight_vector[1]
        flexor_in = 0

        # Calculated weighted inputs
        wieghted_oscillator_inputs = oscillator_inputs * oscillator_weights

        # For each input from other oscillators, send it's absolute value to the flexor neuron if its negative.
        # Else send it to the extensor neuron.
        for oscillator_in in wieghted_oscillator_inputs:
            if (oscillator_in < 0):
                flexor_in += -1 * oscillator_in
            else:
                extensor_in += oscillator_in

        # Similar routing for the feedback signal
        if (feedback < 0):
            flexor_in += -1 * feedback
        else:
            extensor_in += feedback

        # Calculate the extensor neuron's output
        self.extensor(tonic_input, extensor_in)

        # Add the extensor neuron's output to the flexor neurons input.
        extensor_out = self.extensor.output()
        flexor_in += extensor_out * self.weight_vector[0]

        # Calculate the flexor neuron's output
        self.flexor(tonic_input, flexor_in)

        # Return the oscillator output.
        return self.output()


# Test the program by creating a single oscillator by providing random parameters
if __name__ == "__main__":

    # # Neuron parameters
    # tau_r = 1
    # tau_a = 2
    # a = 2.5
    # b = 3.5

    # extensor_parameters = [tau_r, tau_a, b]
    # flexor_parameters = [tau_r, tau_a, b]

    # # Oscillator parameters
    # weight_vector = [a, a]  # Inter-neuron weights
    # output_bias = 0
    # name = "test_oscillator"
    dt = 0.05

    l = [[1.096101336290272, 2.099824043144844, 6.47900555675202],
         [1.096101336290272, 2.099824043144844, 6.47900555675202],
         [5.49422099409134, 5.49422099409134], 0, 3, 'pacemaker', 0.05]

    # Initialise a oscillator object
    # osc = Oscillator(extensor_parameters, flexor_parameters, weight_vector,
    #                  output_bias, name, dt)

    osc = Oscillator(*l)
    x = []
    y = []

    # Calculate output of the oscillator for 10,000 steps with constant external input and zero feedback.
    for i in range(1000):
        x.append(i * dt)
        y.append(osc(np.array([]), np.array([]), 3, 0))
    plt.plot(x, y)
    plt.show()
