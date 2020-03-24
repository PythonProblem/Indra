#*****************************************************************************************#
# Class Name        : Neuron                                                              #
# Class Description : Model of a single neuron used to build Matsuoka Oscillators.        #
#                     Contains two differential equations.                                #
#                     Solves the equations at each time step to find the neuron's output. #
#                     Basic building block of CPG (Central Patter Generator).             #
#                                                                                         #
# Authors           : Vishwas N S, Karthik K Bhat                                         #
#*****************************************************************************************#

# Import required libraries
import numpy as np


# Class definition of a Neuron
class Neuron:

    # Initialise the neuron with all the internal parameters.
    # Provide initial state of the neuron (u and f) if required.
    # Actual variable names from paper_1 have been used.
    # Refer to paper_1 in references section for the equations.

    def __init__(self, tau_r, tau_a, b, dt, u=0, f=0):
        self.tau_r = tau_r  # Rise time
        self.tau_a = tau_a  # Adaptation time
        self.b = b  # Inhibition weight
        self.dt = dt  # Time step (usually 0.1)
        self.u = u  # Membrane Potential
        self.f = f  # Adaptation
        self.y = max(u, 0)  # Neuron Output

    # Retrieve values of the neuron's internal parameters.
    # Returns a list
    def get_parameters(self):
        return [self.tau_r, self.tau_a, self.b]

    # Change the neuron's internal parameters during runtime.
    def set_parameters(self, tau_r, tau_a, b):
        self.tau_r = tau_r
        self.tau_a = tau_a
        self.b = b

    # Reset the output of the neuron.
    def reset_neuron(self):
        self.u = 0
        self.f = 0

    # Return the output of the neuron without calculating the output for the next time step.
    # Returns a non-negative floating point value
    def output(self):
        self.y = max(self.u, 0)
        return self.y

    # Calculate the value of du/dt from the first differential equation for a given value of u.
    # Returns a floating point value.
    def dudt(self, u, total_input):
        return (-u + total_input - self.b * self.f) / self.tau_r

    # Calculate the value of df/dt from the second differential equation for a given value of f.
    # Returns a floating point value.
    def dfdt(self, f):
        return (-f + self.y) / self.tau_a

    # Calculate the output of the neuron for the next time step using the 4th Order Runge-Kutta method
    def __call__(self, tonic_input, sensory_input):

        # Tonic input is a constant non-negative value.
        # Sensory input is a combination of feedback input and inputs from other oscillators.
        total_input = tonic_input - sensory_input

        # Calculate Runge-Kutta coefficients to find change in u.
        u_k1 = self.dt * self.dudt(self.u, total_input)
        u_k2 = self.dt * self.dudt(self.u + 0.5 * u_k1, total_input)
        u_k3 = self.dt * self.dudt(self.u + 0.5 * u_k2, total_input)
        u_k4 = self.dt * self.dudt(self.u + u_k3, total_input)

        # Calculate Runge-Kutta coefficients to find change in f.
        f_k1 = self.dt * self.dfdt(self.f)
        f_k2 = self.dt * self.dfdt(self.f + 0.5 * f_k1)
        f_k3 = self.dt * self.dfdt(self.f + 0.5 * f_k2)
        f_k4 = self.dt * self.dfdt(self.f + f_k3)

        # Update u and f
        self.u += (u_k1 + 2 * u_k2 + 2 * u_k3 + u_k4) / 6
        self.f += (f_k1 + 2 * f_k2 + 2 * f_k3 + f_k4) / 6

        # Return the neuron's output
        return self.output()


# Test the program by creating a single neuron by providing random parameters
if __name__ == "__main__":

    # Create neuron
    n = Neuron(0.1, 0.2, 1, 0.01)

    # Calculate output of the neuron for 10,000 time steps with constant external input and zero feedback.
    for i in range(10000):

        # Call the neuron to calculate the value of next step
        out = n(0, 2, 0)
        print(out)
