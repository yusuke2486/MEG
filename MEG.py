import torch
import numpy as np
import numpy.random as rand
from numpy import ndarray
from torch import Tensor
from torch.nn import Module
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, dense_to_sparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Cora dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0].to(device)

print(data)

class GCNLayer(Module):
    def __init__(self, fdim: int, outfdim: int, dropout: float = .25):
        super().__init__()
        self.W = nn.Linear(fdim, outfdim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.sig = nn.ReLU()

    def forward(self, DADX: Tensor) -> Tensor:
        return self.sig(self.dropout(self.W(DADX.float())))

class GCN(Module):
    def __init__(self, fdim: int, layers: int = None, device = device):
        super().__init__()
        if layers is None:
            layers = fdim - 1
        self.layers = nn.ModuleList(
            [GCNLayer(max(1, fdim - l), max(1, fdim - (l + 1))).to(device) for l in range(layers)]
        )
    
    def forward(self, A: Tensor, X: Tensor):
        Dc = torch.diag(torch.rsqrt(torch.sum(A, axis=0)))
        DAD = torch.matmul(torch.matmul(Dc, A), Dc)
        Xs = [X]
        for l in self.layers:
            Xs.append(l(torch.matmul(DAD, Xs[-1]).float()))
        return Xs[-1], Xs

def AdjMat(S: Tensor):
    U = torch.triu(S, diagonal=1)
    L = torch.tril(S, diagonal=1)
    U[U == 0] = L[U == 0]
    U = torch.triu(U, diagonal=1)
    return U + U.T + torch.eye(S.shape[0], device=device)

class AdjGen(Module):
    def __init__(self, ndim: int, fdim: int, numheads: int, dropout: float = .2, device = device):
        super().__init__()
        hiddim = int(np.ceil(fdim / numheads)) * numheads
        self.attn = nn.MultiheadAttention(
            hiddim, numheads, dropout, bias=False, kdim=fdim, vdim=ndim).to(device)
        self.dense = nn.Linear(hiddim, ndim)
        self.sig = nn.Sigmoid()
    
    def forward(self, A: Tensor, X: Tensor):
        N = X.shape[0]
        fil: Tensor = (torch.ones(N, device=device) - torch.eye(N, device=device)).to(bool)
        pred = []
        for i in range(N):
            output, _ = self.attn(
                X[i].unsqueeze(0), X[fil[i]], A[fil[i]])
            pred.append(self.dense(output.squeeze().float()))
        return torch.mul(A.clone(), self.sig(torch.stack(pred).float()))

def RandomCrossSplit(num: int, setcount: int, S: Tensor):
    sets = []
    vs = S.shape[0]
    num = min(num, vs)
    for _ in range(setcount):
        slct = rand.choice(vs, num, False)
        sets.append((S[slct, :][:, slct], slct))
    return sets

def RestoreCrossSplited(SList: list[tuple[Tensor, ndarray]], BaseS: Tensor):
    BS = BaseS.clone()
    BSRate = torch.zeros(BS.shape, device=device)
    BSMask = torch.zeros(BS.shape, device=device)
    for S, slcd in SList:
        for i, slc in enumerate(slcd.tolist()):
            BSRate[slc, slcd] = BSRate[slc, slcd] + (S[i, :] * 2.0 - 1.0)
            BSMask[slc, slcd] = 1
    BSRate = torch.clamp(BSRate, 0, 1)
    return torch.mul(BSRate, BSMask) + torch.mul(BS, 1.0 - BSMask)

def randomOneHotTensor(shape):
    d = rand.rand(*shape)
    d[d > 0.5] = 1.0
    d[d <= 0.5] = 0.0
    return torch.tensor(d, device=device).float()

def easySample(P: Tensor, th: Tensor):
    Pc: Tensor = P.clone()
    Pc[P < th] = 0
    Pc[P >= th] = 1
    return Pc

def Train(B, Features, Models: list[list[Module]], Optims: list[list[torch.optim.Optimizer]]):
    Datas = len(B)
    RetLosses = []
    AdjLosses = []
    Adjs = []
    for bi in range(Datas):
        b = B[bi]
        models = Models[bi]
        V, slct = b
        Fs = Features[slct]
        for model in models:
            model.eval()
        ret_post, _ = models[1](V, Fs)
        for model in models:
            model.train()
        adj: Tensor = models[0](V, Fs)
        adjclone = AdjMat(easySample(adj, torch.tensor(rand.random(adj.shape), device=device))).detach()
        Adjs.append((adjclone, slct))
        ret, _ = models[1](adjclone, Fs)
        loss_fn = nn.MSELoss()
        ret_optimizer = Optims[bi][1]
        adj_optimizer = Optims[bi][0]
        ret_optimizer.zero_grad()
        retloss = loss_fn(ret, ret_post)
        adj_optimizer.zero_grad()
        adjloss: Tensor = torch.sum(adj) * torch.clip(retloss.detach() + 1e-5, -10., 10.).float()
        AdjLosses.append(adjloss.detach())
        adjloss.backward()
        RetLosses.append(retloss.detach())
        retloss.backward()
        ret_optimizer.step()
        adj_optimizer.step()
        del adjloss
        del retloss
    return RetLosses, AdjLosses, Adjs

fs = 1440
N = 2708
SampleSets = 5
Samples = 52
convL = 10
Models: list[list[Module]] = [
    [AdjGen(Samples, fs, 5).to(device), GCN(fs, convL).to(device)] for _ in range(SampleSets)
]
Optims: list[list[torch.optim.Optimizer]] = [
    [torch.optim.Adam(Models[i][j].parameters(), lr=0.0001) for j in range(len(Models[i]))] for i in range(SampleSets)
]

def Trains(Episodes: int, trainEpochs: int = 20):
    for ep in range(Episodes):
        A = AdjMat(randomOneHotTensor((N, N)))
        sumA = torch.sum(A).item()
        Features = randomOneHotTensor((N, fs))
        print("A = ", A)
        print("Features = ", Features)
        B = RandomCrossSplit(Samples, SampleSets, A)
        for epc in range(trainEpochs):
            retlosses, adjlosses, adjs = Train(B, Features, Models, Optims)
            Result = RestoreCrossSplited(adjs, A)
            Result = AdjMat(Result)
            print("Result =", Result)
            print(f"Reduced to {(torch.sum(Result).item() / sumA) * 100.0:3.3f}%")
            print("GCN Loss Mean:", torch.mean(torch.stack(retlosses)).item())
            print("AdjGen Loss Mean:", torch.mean(torch.stack(adjlosses)).item())

Trains(100)
