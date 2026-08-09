import random
import pandas as pd
import torch
from torch import nn
from torch import optim

device = {
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
}

print(f"Using device {device}")

class EncoderRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.part1 = nn.Sequential(
            nn.Linear(10,20,bias=False))
        self.part1h = nn.Sequential(
            nn.Linear(20,20,bias=False))
        self.part2 = nn.Sequential(
            nn.Linear(20,10,bias=False),
            nn.Softmax(dim=0))
        self.activation = nn.Sigmoid()

    def forward(self, x, h) -> [torch.Tensor,torch.Tensor]:
        h = self.activation(self.part1(x) + self.part1h(h.detach()))
        y = self.part2(h)
        return y, h
    

class DecoderRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.part1 = nn.Sequential(
            nn.Linear(10,20,bias=False))
        self.part1h = nn.Sequential(
            nn.Linear(20,20,bias=False))
        self.part1c = nn.Sequential(
            nn.Linear(20,20,bias=False))
        self.part2 = nn.Sequential(
            nn.Linear(20,10,bias=False),
            nn.Softmax(dim=0))
        self.activation = nn.Sigmoid()

    def forward(self, x, h, c) -> [torch.Tensor,torch.Tensor]:
        h = self.activation(self.part1(x) + self.part1h(h.detach()) + self.part1c(c))
        y = self.part2(h)
        return y, h
    

device = device.pop()
enc = EncoderRNNModel().to(device)
print(enc)

dict = {
    "is":    [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    "has":   [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ],
    "Bob":   [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ],
    "bear":  [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "Alice": [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ],
    "beer":  [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ],
    "brown": [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ],
    "tall":  [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ],
    "green": [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 ],
    "<eos>": [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ]
}

def normalizeDict():
    for w in dict:
        print(dict[w])
        dict[w] = torch.Tensor(dict[w]).float().to(device)
        n = torch.sqrt(torch.dot(dict[w],dict[w])).float().to(device)
        if n != 0:
            dict[w] = dict[w].div(n).float().to(device)
        print(dict[w])

normalizeDict()

def getTensor(word):
    return dict[word]

def getWord(tensor):
    word = ""
    prod = 0
    for w in dict:
        p = torch.dot(getTensor(w),tensor)
        if p > prod:
            word = w
            prod = p

    return word

t = getTensor("Bob")
print(getWord(t))

t = getTensor("Alice")
print(getWord(t))



corpus = "Bob has beer <eos> Alice is tall <eos> Alice has beer <eos> Bob has beer <eos> bear is brown <eos> Alice has beer <eos> Bob is tall <eos> Alice is tall <eos> bear is tall <eos> beer is green <eos> Bob is Alice <eos> brown bear is tall <eos> bear has beer <eos> Bob has beer <eos> bear is brown <eos> Alice has beer <eos> Bob is tall <eos> Alice is tall <eos> bear is tall <eos> beer is green <eos> Bob is Alice <eos> brown bear is tall <eos> bear has beer <eos>"
words = corpus.split()

torch.autograd.set_detect_anomaly(True)

dec = DecoderRNNModel().to(device)
lossFn = nn.MSELoss()
optimizer = optim.SGD(enc.parameters(), lr=0.2)
enc.train()
enc.zero_grad()
batchSize = 1
epochs = 15

for n in range(0,epochs):
    loss = torch.zeros(1).to(device)
    h = torch.zeros(20).to(device)
    c = torch.zeros(20).to(device)
    optimizer.zero_grad() 
    inEncoder = True

    for i in range(0,len(words)-1):
        x = getTensor(words[i])
        expected_y = getTensor(words[i+1])
        if inEncoder:
            y,h = enc(x,h)
        else:
            y,h = dec(x,h,c)

        loss = loss + lossFn(y, expected_y)

        if (i % batchSize) == 0 or words[i] == '<eos>': 
            print(i,words[i],x,loss.item())
            loss.backward()
            optimizer.step()   
            optimizer.zero_grad()
            loss = torch.zeros(1).to(device)

        if words[i] == '<eos>':
            inEncoder = not inEncoder
            if not inEncoder:
                c = h.detach()            
            h = torch.zeros(20).to(device)

# Using the trained model
word = "Bob"
h = torch.zeros(20).to(device)

for i in range(0,10):
    print(word)
    x = getTensor(word)
    #print(h)
    y,h = dec(x,h,c)
    word = getWord(y)
