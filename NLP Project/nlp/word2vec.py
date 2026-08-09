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

class Word2VecModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(160,20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,640),
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        return self.model(x)

device = device.pop()
w2v = Word2VecModel().to(device)
print(w2v)

td = pd.read_csv('training.csv',sep=',', header=None)

n=int(len(td.values)/5)
td_sample = random.sample(range(0,n),n)
m = int(n*0.7);

train = pd.DataFrame(columns=td.columns)
pos = 0
for idx in range(0,m):
    i = td_sample[idx]*5+2;
    train.loc[pos] = td.values[i-2]
    pos+=1
    train.loc[pos] = td.values[i-1]
    pos+=1
    train.loc[pos] = td.values[i]
    pos+=1
    train.loc[pos] = td.values[i+1]
    pos+=1
    train.loc[pos] = td.values[i+2]
    pos+=1

test = pd.DataFrame(columns=td.columns)

pos = 0
for idx in range(m+1,n):
    i = td_sample[idx]*5+2;
    test.loc[pos] = td.values[i-2]
    pos+=1
    test.loc[pos] = td.values[i-1]
    pos+=1
    test.loc[pos] = td.values[i]
    pos+=1
    test.loc[pos] = td.values[i+1]
    pos+=1
    test.loc[pos] = td.values[i+2]
    pos+=1

batchSize=20
lossFn = nn.MSELoss()
optimizer = optim.SGD(w2v.parameters(), lr=0.3) # lr= learning rate hyperparameter that controls the size of the steps taken during the optimization process to minimize the loss function

w2v.train()
td = train
n=int(len(td.values)/5)

for epoch in range(0,10):
    td_sample = random.sample(range(0,n),n)
    output = torch.zeros(4*len(td.values[0])).to(device)
    expected = torch.zeros(4*len(td.values[0])).to(device)

    for idx in range(0,len(td_sample)):        
        i = td_sample[idx]*5+2
        input = torch.from_numpy(td.values[i]).float().to(device)
        expected = torch.cat((torch.from_numpy(td.values[i-2]).float().to(device),
            torch.from_numpy(td.values[i-1]).float().to(device),
            torch.from_numpy(td.values[i+1]).float().to(device),
            torch.from_numpy(td.values[i+2]).float().to(device))) + expected

        output = w2v(input) + output

        if (idx % batchSize) == 0:            
            loss = lossFn(output, expected)
            output = torch.zeros(len(output)).to(device)
            expected = torch.zeros(len(expected)).to(device)
            print(loss)

            loss.backward()
            optimizer.step() 
            optimizer.zero_grad() 

td = test
n = int(len(td.values)/5)
loss = 0

for idx in range(0,n):
    i = idx*5+2
    input = torch.from_numpy(td.values[i]).float().to(device)
    expected = torch.cat((torch.from_numpy(td.values[i-2]).float().to(device),
        torch.from_numpy(td.values[i-1]).float().to(device),
        torch.from_numpy(td.values[i+1]).float().to(device),
        torch.from_numpy(td.values[i+2]).float().to(device)))

    output = w2v(input)

    out = torch.split(output,160)
    exp = torch.split(expected,160)

    for j in range(0,3):
        m1 = out[j].argmax().item()
        m2 = exp[j].argmax().item()
        if m1 == m2:
            loss = loss + 0.25
        print(m1,m2)

print(loss/n)



