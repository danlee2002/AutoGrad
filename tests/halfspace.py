import sys
sys.path.append('../')
from autograd import engine
from autograd import neuralnetwork as neural

def main():
  model =neural.nn([neural.MLP(2,4,10,2)], loss = neural.nn.crossentropy, lr = 0.2)
  #sample program to learn a halfspace
  xs = [[-1,-2],[-1,-1],[1,3],[1,1],[2,1],[-3,-1],[-2,-2]]
  ys = [[1,0],[1,0],[0,1],[0,1],[1,0],[1,0]]
  for i in range(400):
    y_pred = model.forward(xs)
    model.backward(y_pred, ys)
if __name__ == '__main__':
  main()