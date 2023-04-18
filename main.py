import torch
torch.manual_seed(15)
from args import args
from kronecker_product import kronecker_product as kron
from OneStepVar import OneStepVar
from OneStepVarGr import OneStepVarGr
from GetExaIsing import GetFiniteExaIsing, GetInitFiniteExaE

dtype = torch.complex128
J, mu, hz, D, d, MaxIter, VarMethod = args.J, args.mu, args.hz, args.D, args.d, args.MaxIter, args.VarMethod

args.OutResults = 'VarMethod=' + args.VarMethod + ',J=' + str(J) + ',mu=' + str(mu) + ',hz=' + str(hz) + ',D=' + str(D) + \
                  ',MaxIter=' + str(MaxIter) + ',lr=' + str(args.lr) + ',max_iter=' + str(args.max_iter) + '.txt'

# GetExaSolution
args.InitExaE = GetInitFiniteExaE()

s0 = torch.tensor([[1, 0], [0, 1]], dtype=dtype)
sx = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
sy = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
sz = torch.tensor([[1, 0], [0, -1]], dtype=dtype)

Jx = 0.5 * J * (1 + mu)
Jy = 0.5 * J * (1 - mu)
Ham = -Jx * kron(sx, sx) - Jy * kron(sy, sy) - hz * (kron(sz, s0) + kron(s0, sz)) / 2
Ham = Ham.contiguous().view(d, d, d, d)

# Init Tensor
UL = torch.rand(D * d, D * d, dtype=dtype)
T = torch.matrix_exp(UL - UL.conj().t())
T = T[:, :D]

for Iter in range(MaxIter):
    if VarMethod == 'Normal':
        T = OneStepVar(Iter, T, Ham, dtype)
    elif VarMethod == 'Grassmann':
        T = OneStepVarGr(Iter, T, Ham, dtype)
    else:
        raise ValueError("=========== No Suit VarMethod ===========")