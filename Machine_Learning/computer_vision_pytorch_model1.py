import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

#print(torch.__version__)
#print(torchvision.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

train_data = datasets.FashionMNIST(
    root= 'data',
    train= True,
    download= True,
    transform= torchvision.transforms.ToTensor(),
    target_transform= None
)

test_data = datasets.FashionMNIST(
    root= 'data',
    train= False,
    download= True,
    transform= ToTensor(),
    target_transform= None

)
class_names = train_data.classes
class_to_idx = train_data.class_to_idx

batchSize = 12

train_dataloader = DataLoader(dataset=train_data, batch_size= batchSize, shuffle=True)

test_dataloader = DataLoader(dataset=test_data, batch_size= batchSize, shuffle=False)


train_features_batch, train_labels_batch = next(iter(train_dataloader))


class FashionMNSTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features= hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features= output_shape),
            
        )
    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(42)

model_1 = FashionMNSTModelV1(
       input_shape= 784, # this is 28 * 28
       hidden_units= 10, # how many units in the hidden layer
       output_shape= len(class_names)

).to(device)

from Helper_functions import accuracy_fn

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),lr=0.1)

from tqdm.auto import tqdm

torch.manual_seed(42)

epochs = 3

for epoch in tqdm(range(epochs)):
    print(epoch)

    train_loss = 0
    for batch, (X,y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        model_1.train()

        y_pred = model_1(X)
        loss = loss_fn(y_pred, y)
        train_loss +=  loss

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if batch % 400 == 0:
            print(f"looked at {batch * len(X)/len(train_dataloader.dataset)} samples")

    train_loss /= len(train_dataloader)

    test_loss, test_acc = 0, 0
    model_1.eval()
    with torch.inference_mode():
        for X_test,y_test in test_dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            test_pred = model_1(X_test)
            test_loss += loss_fn(test_pred,y_test)
            test_acc += accuracy_fn(y_true=y_test, y_pred= test_pred.argmax(dim=1))
    
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    print(f"\nTrain loss: {train_loss:.4f} | Test loss:  {test_loss:.4f} Test acc: {test_acc}")

'''
***************************************************************************
                            CONFUSION MATRIX
***************************************************************************
'''
new topic started with pytorch custom datasets

start from 
chapter 4 pytorch custom datasets https://prnt.sc/yS0JyPfHicum
19:44:02 / 01:01:37:25


  

    

