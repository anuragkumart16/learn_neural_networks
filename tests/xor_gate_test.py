from perceptron import Perceptron

def step(x):
    return 1 if x >= 0 else 0


perceptron = Perceptron(2,step)

# train multiple epoches
no_of_epoches = 20

# epoch
data = [
    {"inputs" : [0,0],"output" : 0},
    {"inputs" : [0,1],"output" : 1},
    {"inputs" : [1,0],"output" : 1},
    {"inputs" : [1,1],"output" : 0},
]

for epoch in range(no_of_epoches):
    for i in range(4):
        result = perceptron.train(
            data[i]["inputs"],
            data[i]["output"]
        )
        print(result)
        
print()
print("Training Complete!!")
print()
print(perceptron.predict([1,0]))

# XOR detection requires equation with more order, which in some sense means to use multiple perceptrons.