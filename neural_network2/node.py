import math
import random


class Node(object):
    def __init__(self, inputs = 0):
        self.number_of_inputs = inputs
        self.weights = []
        self.inputs = []
        self.bias = 0
        self.output = 0
        self.z_output = 0
        self.set_weights_and_bias()
    def set_weights_and_bias(self, weights = [], bias = None):
        """" If weights are given, such as on an update
        from backpropogation then the nodes weights for
        each input will be set. Otherwise the weights will
        be initialized to a random value. The Amount of
        weights initialized is determined by the number
        of inputs the node receives."""

        #the weights are given
        if len(weights) > 0:
            self.weights = weights
        #no weights given
        else:
            i = 0
            while (i < self.number_of_inputs):
                self.weights.append(random.uniform(0, 1))
                i += 1
        #set bias
        if bias is not None:
            self.bias = bias
        else:
            if self.number_of_inputs > 0:
                self.bias = random.uniform(0, 1)

    #This function will multiply the nodes input with its respective
    #weight  and sum that for all inputs. Then it will add the bias
    #Finally it will call sigmoid to squash its output.
    def activation_sum(self, inputs = []):
        if self.number_of_inputs == 0:
            self.inputs = inputs
            self.output = inputs[0]
        else:
            i = 0
            self.inputs = inputs
            while (i < self.number_of_inputs):
                self.output += self.weights[i] * self.inputs[i]
                i += 1
            self.output += self.bias
        self.z_output = self.output
        self.activation_2_sigmoid_squash()

    #the purpose of this function is to squash thde output to a
    #value between 0...1
    def activation_2_sigmoid_squash(self):
        self.output = 1 / (1 + math.exp(-self.output))