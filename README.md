# MNIST-NNmathsANDnumpy

resources & blogs used:
1. [blog 1](https://www.kdnuggets.com/2019/11/build-artificial-neural-network-scratch-part-1.html)
2. [blog 2](https://towardsdatascience.com/building-a-deep-neural-network-from-scratch-using-numpy-4f28a1df157a)
3. [blog 3](https://www.samsonzhang.com/2020/11/24/understanding-the-math-behind-neural-networks-by-building-one-from-scratch-no-tf-keras-just-numpy)

## Forward propagation

Z[1]=W[1]X+b[1]

A[1]=gReLU(Z[1]))

Z[2]=W[2]A[1]+b[2]

A[2]=gsoftmax(Z[2])

## Backward propagation

dZ[2]=A[2]−Y

dW[2]=1mdZ[2]A[1]T

dB[2]=1mΣdZ[2]

dZ[1]=W[2]TdZ[2].∗g[1]′(z[1])

dW[1]=1mdZ[1]A[0]T

dB[1]=1mΣdZ[1]]


## Parameter updates

W[2]:=W[2]−αdW[2] 

b[2]:=b[2]−αdb[2]

W[1]:=W[1]−αdW[1]

b[1]:=b[1]−αdb[1]
