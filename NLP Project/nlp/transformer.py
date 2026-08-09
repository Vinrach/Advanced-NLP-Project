import torch
from torch import nn
from torch import optim

class AttentionHead(nn.Module):
    def __init__(self, input_len: int, d: int):
        super().__init__()

        self.d = d
        self.w_q = nn.Sequential(
            nn.Linear(input_len,d,bias=False))
        self.w_k = nn.Sequential(
            nn.Linear(input_len,d,bias=False))
        self.w_v = nn.Sequential(
            nn.Linear(input_len,d,bias=False))
        
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        scores = torch.matmul(q, k.t()) / torch.sqrt(torch.tensor(self.d, dtype=torch.float32))
        scores = nn.Softmax(dim=0)(scores)

        z = torch.matmul(scores, v)
        return z
    

class MultiHeadAttention(nn.Module):
    def __init__(self, input_len: int, d: int, num_heads: int):
        super().__init__()
        self.heads = []
        for _ in range(0,num_heads):
            self.heads.append(AttentionHead(input_len,d).to(device))

        self.w_0 = nn.Sequential(
            nn.Linear(d*num_heads,input_len,bias=False))


    def forward(self,x: torch.Tensor) -> torch.Tensor:
        z = self.heads[0](x)
        print(z)
        for i in range(1,len(self.heads)):
            z = torch.cat((z,self.heads[i](x)),1)

        return self.w_0(z)


class Encoder(nn.Module):
    def __init__(self): 
        super().__init__()
            self.part1 = nn.Sequential(
            nn.Linear(,20,bias=False))




#Initilaize device
device = {
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
}

print(f"Using device {device}")  

device = device.pop()
mha = MultiHeadAttention(4,3,2).to(device)

print(mha)

x = torch.zeros(2,4).to(device)
z = mha(x)

print(z)


