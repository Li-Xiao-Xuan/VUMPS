import io
import torch
from args import args
from TorNcon import ncon
from GetVec import GetRightVec, GetLeftVec, GetVec

def OneStepVar(Iter, T, Ham, dtype):


    TReal = torch.nn.Parameter(T.real, requires_grad=True)
    TImag = torch.nn.Parameter(T.imag, requires_grad=True)
    optimizer = torch.optim.LBFGS([TReal, TImag], lr=args.lr, max_iter=args.max_iter)

    def LossW(TReal, TImag, Ham):
        T = TReal + 1j * TImag
        T = T.contiguous().view(args.D, args.d, args.D).permute(0, 2, 1)
        TMTMat = ncon([T.conj(), T.conj(), Ham, T, T], ([-1, 1, 3], [1, -3, 4], [3, 4, 5, 6], [-2, 2, 5], [2, -4, 6]))
        TMTMat = TMTMat.contiguous().view(args.D * args.D, args.D * args.D)

        TTMat = ncon([T.conj(), T], ([-1, -3, 1], [-2, -4, 1]))
        TTMat = TTMat.contiguous().view(args.D * args.D, args.D * args.D)

        # Iterative method
        # RightVec, MaxRightVal, RightError = GetRightVec(TTMat, dtype)
        # LeftVec, MaxLeftVal, LeftError = GetLeftVec(TTMat, dtype)

        # Diagonalization
        RightError, LeftError = 0, 0
        LeftVec, RightVec, MaxLeftVal, MaxRightVal = GetVec(TTMat)

        VarE = LeftVec @ TMTMat @ RightVec / (LeftVec @ RightVec * MaxLeftVal * MaxRightVal)
        VarE = VarE.real

        return VarE, LeftError, RightError

    def closure():
        optimizer.zero_grad()
        VarE, LeftError, RightError = LossW(TReal, TImag, Ham)
        VarE.backward()
        Grad = [TReal.grad.detach().norm().item(), TImag.grad.detach().norm().item()]

        with io.open(args.OutResults, 'a', buffering=1, newline='\n') as file:
            message = (9 * '{:.12f}  ').format(args.mu, args.hz, args.D, Iter, VarE, args.InitExaE, (VarE - args.InitExaE) / abs(args.InitExaE), Grad[0], Grad[1])
            file.write(message + u'\n')
        file.close()
        print('Iter: %d, Phy: %.16f, diff: %.16f, Grad: %.16f, %.16f' % (Iter, VarE, (VarE - args.InitExaE) / abs(args.InitExaE), Grad[0], Grad[1]))

        return VarE

    VarE = optimizer.step(closure)
    T = TReal + 1j * TImag

    return T