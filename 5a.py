
import numpy as np
from functools import reduce
def perceptron(weight, bias, x):
    model = np.add(np.dot(x, weight), bias)#wx+b
    print('model: ',(model))
    logit = 1/(1+np.exp(-model))
    print('Type:',(logit))
    return np.round(logit)
def compute(logictype, weightdict, dataset):
    weights = np.array([ weightdict[logictype][w] for w in weightdict[logictype].keys()])
    output = np.array([ perceptron(weights, weightdict['bias'][logictype], val) for val in dataset])
    print(logictype)
    return logictype, output

def main():
    logic = {
        'logic_and' : {
            'w0': -0.1,
            'w1': 0.2,
            'w2': 0.2
        },
        
        'logic_nand': {
            'w0': 0.6,
            'w1': -0.8,
            'w2': -0.8
        },
        
        'bias': {
            'logic_and': -0.2,
            'logic_nand': 0.3,
        }
    }
    dataset = np.array([
        [1,0,0],
        [1,0,1],
        [1,1,0],
        [1,1,1]
    ])

    logic_and = compute('logic_and', logic, dataset)
    logic_nand = compute('logic_nand', logic, dataset)
    print(logic_nand)
    def template(dataset, name, data):
        print("Logic Function: ",name[6:].upper())
        print("X0	X1	X2	Y")
        toPrint = ["{1}	{2}	{3}	{0}".format(output, *datas) for datas, output in zip(dataset, data)]
        for i in toPrint:
            print(i)

    gates = [logic_and, logic_nand]
    print("yaha pe maine print karaya",*logic_and[1])
    for i in gates:
        template(dataset, *i)
main()
