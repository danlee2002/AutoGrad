import sys
sys.path.append('../')
from autograd.neuralnetwork import nn, MLP

def main():
    model =nn([MLP(2,5,10,2)], loss = nn.crossentropy, lr = 0.05)
    #sample program to learn xor logic gate
    xs = [[1.0,0.0],[0.0,1.0],[0.0,0.0],[1.0,1.0]]
    ys = [[1,0],[1,0],[0,1],[0,1]]
    for i in range(1000):
        y_pred = model.forward(xs)
        model.backward(y_pred, ys)

if __name__ == '__main__':
    main()
