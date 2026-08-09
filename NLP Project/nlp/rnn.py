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

class RNNModel(nn.Module):  # all operations are adding more paramters to the model for better representations
    def __init__(self):
        super().__init__()
        self.part1 = nn.Sequential(
            nn.Linear(10,20,bias=False)) # matrix size is 10 so it is size 10
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
    

device = device.pop()
rnn = RNNModel().to(device)
print(rnn)

dict = {
    "is":    [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    "has":   [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ],
    "Bob":   [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ],
    "bear":  [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ],
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

#normalizeDict()

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

lossFn = nn.MSELoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.2) # tunning the hyperparameters for better learning rate
rnn.train()
rnn.zero_grad()
batchSize = 1 # increase the batch size for updating the gradients # its is fixed in size.
epochs = 15 # how many times the code should iterarte the corpus

for n in range(0,epochs):
    loss = torch.zeros(1).to(device)
    h = torch.zeros(20).to(device)
    optimizer.zero_grad() 
    stop = False

    for i in range(0,len(words)-1):
        x = getTensor(words[i])
        expected_y = getTensor(words[i+1])
        y,h = rnn(x,h)
        loss = loss + lossFn(y, expected_y)

        if (i % batchSize) == 0: 
            print(i,words[i],x,loss.item())
            loss.backward()
            optimizer.step()   
            optimizer.zero_grad()
            if loss.item() < 0.04:
                stop = True
                break
            loss = torch.zeros(1).to(device)

    if stop:
        break

# Using the trained model
word = "Bob"
h = torch.zeros(20).to(device)

for i in range(0,10):
    print(word)
    x = getTensor(word)
    #print(h)
    y,h = rnn(x,h)
    word = getWord(y)
