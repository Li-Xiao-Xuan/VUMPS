import torch
from args import args

def GetRightVec(TranMat, dtype):

    # Get Right Vector
    RightVec = torch.rand(TranMat.shape[1], dtype=dtype)
    RightVec = RightVec / RightVec.norm()

    LastRightVec = RightVec
    for ix in range(1000):
        RightVec = TranMat @ RightVec
        RightVec = RightVec / RightVec.norm()
        RightError = torch.dist(RightVec, LastRightVec)
        if RightError < args.Threshold:
            break
        else:
            LastRightVec = RightVec
    MaxRightVal = (TranMat @ RightVec)[0] / RightVec[0]

    return RightVec, MaxRightVal, RightError

def GetLeftVec(TranMat, dtype):

    # Get Left Vector
    LeftVec = torch.rand(TranMat.shape[0], dtype=dtype)
    LeftVec = LeftVec / LeftVec.norm()
    LastLeftVec = LeftVec
    for ix in range(1000):
        LeftVec = LeftVec @ TranMat
        LeftVec = LeftVec / LeftVec.norm()
        LeftError = torch.dist(LeftVec, LastLeftVec)
        if LeftError < args.Threshold:
            break
        else:
            LastLeftVec = LeftVec
    MaxLeftVal = (LeftVec @ TranMat)[0] / LeftVec[0]

    return LeftVec, MaxLeftVal, LeftError

def GetVec(TranMat):

    # Get Right Vec
    RightVal, RightU = torch.linalg.eig(TranMat)
    RightE, RightInd = torch.sort(RightVal.abs(), descending=True)
    RightVec = RightU[:, RightInd[0]]

    # Get Left Vec
    LeftVal, LeftU = torch.linalg.eig(TranMat.conj().t())
    LeftE, LeftInd = torch.sort(LeftVal.abs(), descending=True)
    LeftVec = LeftU.conj().t()[LeftInd[0], :]

    return LeftVec, RightVec, LeftE[LeftInd[0]], RightE[RightInd[0]]

if __name__ == '__main__':
    TranMat = torch.rand()