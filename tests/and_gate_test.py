from perceptron import Perceptron

def step(x):
    return 1 if x >= 0 else 0


perceptron = Perceptron(2,step)
perceptron.train([0,0],0)
perceptron.train([1,1],1)
perceptron.train([0,1],0)
perceptron.train([1,0],0)

print(perceptron.predict([1,0]))