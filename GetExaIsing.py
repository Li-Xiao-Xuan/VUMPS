## The exact solution of Ising model ##
import torch
import numpy as np
from args import args
from math import sin, cos, sqrt, pi

# finite chain length
def GetFiniteExaIsing(L=500, dtype=torch.float64):
    # Open boundary
    # H = J\sum_{i} S_{i}^zS_{i+1}^{z} + hx\sum_{i} S_{i}^{x}
    J, h = args.J, args.hz
    H11 = 0.5 * h * torch.eye(L, dtype=dtype) + 0.125 * J * torch.diag(torch.diag(torch.eye(L - 1, dtype=dtype)), 1) + 0.125 * J * torch.diag(torch.diag(torch.eye(L - 1, dtype=dtype)), -1)
    H22 = -H11
    H12 = 0.125 * J * torch.diag(torch.diag(torch.eye(L - 1, dtype=dtype)), 1) - 0.125 * J * torch.diag(torch.diag(torch.eye(L - 1, dtype=dtype)), -1)
    H21 = -H12

    H1 = torch.cat((H11, H12), 1)
    H2 = torch.cat((H21, H22), 1)
    H  = torch.cat((H1,  H2) , 0)
    H  = 0.5 * (H + H.t())

    spe, _ = torch.linalg.eig(H)
    spe, _ = torch.sort(torch.real(spe), descending=False)

    exaE = sum(spe[:L]) / L

    return exaE

# infinite chain length
def GetInitFiniteExaE(L=5000):

    J, mu, hz = args.J, args.mu, args.hz
    ek = 0
    for n in np.linspace(-L/2 + 1, L / 2, L):
        k = 2 * n * pi / L
        ek = ek + J * sqrt((cos(k) - hz/J) ** 2 + (mu * sin(k)) ** 2)

    return -abs(ek / L)


# def func(x):
#     print("x=", x)  # 用于展示quad()函数对func的多次调用
#     return math.cos(2 * math.pi * x) * math.exp(-x) + 1.2
#
# fArea, err = integrate.quad(func, 0.7, 4)
# print("Integral area:", fArea)
#
# for k in range():