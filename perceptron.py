import random

class Perceptron:
    def __init__(self,no_of_inputs,activation_function,learning_rate=0.1):

        # init weights
        weights = []
        for _ in range(no_of_inputs):
            weights.append(random.random())
        self.weights = weights

        #init a bias
        self.bias = 0

        # init activation function 
        self.activation_function = activation_function

        # init learning_rate
        self.learning_rate = learning_rate

    def get_attributes(self):
        return {
            "weights" : self.weights,
            "bias" : self.bias,
            "activation_function" : self.activation_function,
            "learning_rate" : self.learning_rate
        }

    def calulate_weighted_sum(self,inputs):
        weighted_sum = 0
        for index in range(len(inputs)):
            weighted_sum += inputs[index] * self.weights[index]

        weighted_sum += self.bias
        return weighted_sum
    

    def train(self,inputs,y_actual):
        weighted_sum =  self.calulate_weighted_sum(inputs)
        output = self.activation_function(weighted_sum)

        # update weights 
        error = y_actual-output
        for index in range(len(inputs)):
            change_in_weight = error*self.learning_rate*inputs[index]
            self.weights[index] += change_in_weight

        # update bias
        self.bias += error*self.learning_rate
        return (output,self.get_attributes())

    def predict(self,inputs):
        print("Predicting for ",inputs)
        weighted_sum =  self.calulate_weighted_sum(inputs)
        output = self.activation_function(weighted_sum)
        return (output,self.get_attributes())


