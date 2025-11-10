#1 make classififcation data and get it ready
#data
import sklearn
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_circles
from torchmetrics import Accuracy
 
#Make 1000 samples

n_samples = 1000

#create circles
X,y = make_circles(n_samples,noise=0.03,random_state=42)

len(X)
len(y)

#print(X[0][0])
circles = pd.DataFrame({"X1" : X[:,0], "X2":X[:,1], "label" : y})
  
plt.scatter(x=X[:,0],y=X[:,1],c=y,cmap=plt.cm.RdYlBu)

#plt.show()

# print(X.shape)
# print(y.shape)

X_sample = X[0]
y_sample = y[0]
 
print(f"Values for one sample of X: {X_sample} and the sample for y:{y_sample}")
# converting data into tensors
# default pytorch data type is float 32
# for numpy default data type is float 64

# converting to 
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

#now check the data type (now it will float 32)
type(X), X.dtype, y.dtype

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#print(len(X_train), len(X_test), len(y_train), len(y_test))

# Building a jmodel
#device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")

# 1. Construct a model that subclasses nn.Module
# class CircleModelV0(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 2. Create 2 nn.Linear Layers capable of handling the shapes of our data
#         self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features and upscale to 5 features
#         self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features from previous layer and outputs a single feature (same shape as y)

#     # 3. Define a forward() method that outlines the forward pass 
#     def forward(self,x):
#         return self.layer_2(self.layer_1(x)) # x-> layer_1 -> layer_2 -> output

# #4. Instantiate an instance of our model class and send it to the target device
# model_0 = CircleModelV0().to(device)
# next(model_0.parameters()).device

#Let's replicate the model above using nn.sequnetial
#above code's short form of below code
model_0 = nn.Sequential(
      nn.Linear(in_features=2,out_features=5),
      nn.Linear(in_features= 5,out_features=1)
).to(device)

#make predictions
# with torch.inference_mode():
#     untraine_preds = model_0(X_test.to(device))
# print(f"Length of prediction: {len(untraine_preds)}, Shape: {untraine_preds.shape}")
# print(f"Length of the test samples: {len(X_test)}, Shape: {X_test.shape}")
# print(f"\n First 10 predictions: {torch.round(untraine_preds[:10])}")
# print(f"\n First 10 labels:\n {y_test[:10]}")

#loss_fn = nn.BCELoss()

loss_fn = nn.BCEWithLogitsLoss() # this is sigmoid activation function used
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.1)

#calculate the accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

# training model
# training loop
'''
1 Fwd pass
2 calculate the lose
3 optimizer zero grad
4 Loss backward (backpropogation)
5 Optimizer (gradient descent)

going from raw logits --> predictions probabilites --> prediction labels
'''
# with torch.inference_mode():
#     y_logits = model_0(X_test.to(device))[:5]

# # use sigmoid -activation function
# Y_pred_probs = torch.sigmoid(y_logits)

# #find the predicated labels
# y_preds = torch.round(Y_pred_probs)

# #in full
# y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# cgeck for equality
#print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))



# building a training and tesing loop
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set the number of epochs
#pochs = 800

#Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test,y_test = X_test.to(device), y_test.to(device)

# #Build training and evalutation loop
# for epoch in range(epochs):
#     # training
#     model_0.train()

#     #1. Forward Pass
#     y_logits = model_0(X_train).squeeze()
#     y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits

#     #2. calculate loss/accuracy
#     # y_logits - real number before applying sigmoid so its real number not probability data and its model output
#     # Y_train - its real number not probabitlity data which is expected output
#     loss = loss_fn(y_logits,y_train)

#     #y_train - expected output in real number
#     #y_pred - probability data aftr applying sigmoid function which is gain from model ouput number
#     acc = accuracy_fn(y_true=y_train, y_pred = y_pred)

#     #3. Optimizer zero grad
#     optimizer.zero_grad()

#     #4. Loss backward (backpropogation)
#     loss.backward()

#     #5. Optimizer step (gradient decent)
#     optimizer.step()

#     ##testing
#     model_0.eval()
#     with torch.inference_mode():
#         #1. Forwad pass
#         test_logits = model_0(X_test).squeeze()
#         test_pred = torch.round(torch.sigmoid(test_logits))

#         # calculate test loss/acc
#         test_loss = loss_fn(test_logits,y_test)
#         test_acc = accuracy_fn(y_true=y_test, y_pred = test_pred)

               
#     if epoch % 10 == 0:
#         print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# make predictions and evalate the model
# import requests
# from pathlib import Path

# #download helperfuntion fromlearnpytorch repo (if its not downloaded)
# if Path("Helper_functions.py").is_file():
#     print("skipped download")
# else:
#     request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
#     with open("Helper_functions.py", "wb") as f:
#         f.write(request.content)

# from Helper_functions import plot_predictions, plot_decision_boundary

# # plot decision boundary of the model
# plt.figure(figsize=(12,6))

# plt.subplot(1,2,1)
# plt.title("Train")
# plot_decision_boundary(model_0.cpu(), X_train.cpu(), y_train.cpu())

# plt.subplot(1,2,2)
# plt.title("Test")
# plot_decision_boundary(model_0.cpu(), X_test.cpu(), y_test.cpu())

# plt.show()

# How to improve the model ?
# add more layers - give the model chance to learn about patterns in the data
# add more hidden units- go from 5 hidden units to 10 hidden units
# Fit for longer (more ecpocs)
# changing activation functions
# change the learning rate
# Change the loss function
# These options are all from a model's perspective because they deal directly with the model, rather than the data
# And beacause these options are all values we (as mahine learning engineers and data scientists) can change, they are hyperprameters

# Let's ty and improve our above model by:
'''
* Adding more hidden units: 5 -> 10
* Increase the number of layers: 2 -> 3
* Increase the number of epochs: 100 -> 1000

'''


class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features= 2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
    
    def forward(self, x):
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z)
        return self.layer_3(self.layer_2(self.layer_1(x)))

model_1 = CircleModelV1().to(device)

#create a loss function
loss_fn = nn.BCEWithLogitsLoss()

#create optimizer
optimizer = torch.optim.SGD(params= model_1.parameters(), lr=0.1)

#write training valuation loop in model_1
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Train for longer
epochs = 1000

# #Put data to target device
# X_train, y_train = X_train.to(device), y_train.to(device)
# X_test,y_test = X_test.to(device), y_test.to(device)


# for epoch in range(epochs):
#     ## Training
#     model_1.train()

#     #1. Forward pass
#     y_logits = model_1(X_train).squeeze() # output  of final layer
#     y_pred = torch.round(torch.sigmoid(y_logits))

#     #2 Calculate Loss/acc
#     loss = loss_fn(y_logits, y_train)
#     acc = accuracy_fn(y_true = y_train, y_pred = y_pred )

#     #3. Optimizer zero_grad() - zero old gradients bcz gradient parameters is getting accumulatin in each epoc so, 
#     # in this case learning will not be done properly - in this case autograde is on so automatically calculate gredient
#     optimizer.zero_grad()

#     #4. Loss backward (backpropogation) and calculating new gredient
#     loss.backward()

#     #Update weights
#     optimizer.step()

#     #Testing
#     model_1.eval()
#     with torch.inference_mode():
#         #1. Forward pass
#         test_logits = model_1(X_test).squeeze()

#         # here we are using sigmoid which converts values in range 0 to 1 into probabilities and sigmoid is used in minary classification problem
#         test_pred = torch.round(torch.sigmoid(test_logits))

#         # calculae the loss usgin this test data
#         test_loss = loss_fn(test_logits,y_test)

#         test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

#     #print out
#     if epoch % 10 == 0:
#         print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# from Helper_functions import plot_predictions, plot_decision_boundary

# # plot decision boundary of the model
# plt.figure(figsize=(12,6))

# plt.subplot(1,2,1)
# plt.title("Train")
# plot_decision_boundary(model_1.cpu(), X_train.cpu(), y_train.cpu())

# plt.subplot(1,2,2)
# plt.title("Test")
# plot_decision_boundary(model_1.cpu(), X_test.cpu(), y_test.cpu())

# plt.show()

# Preparing data to see if our model can fit a straight line
# one way to troubleshoot to a larger problem is to test out a smaller problem
# above model after improving also we are not able to train a model
# so, the problem in our above model is our data is in circular and model is strctured in linear to resolve there are 2 solutions
 # 1 - change model structure and add non linear activation funciton like RELU
 # 2 - Chnage circular sample data set to linear data set so, in below code we are implementing option 2
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.01
# create data - using linear regression formula which is linear data
X_regression = torch.arange(start,end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias #Linear regression formula (without epsilon)

# Check the data
print(len(X_regression))

# create training and test splits
train_split = int(0.8* len(X_regression))
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

#So, here the above data set the problem is X_train_regression(input featuers are 1) and y_test_regression (output features are 1) now
# here in our above model structure (at line no 228) we have 2 input features and 1 output feature so we need to adjust the data set according to our model
model_2 = nn.Sequential(
    nn.Linear(in_features= 1, out_features=10),
    nn.Linear(in_features= 10, out_features=10),
    nn.Linear(in_features= 10, out_features=1),
).to(device)

# Loss and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_2.parameters(),lr=0.01)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Train for longer
epochs = 1000

#Put data to target device
X_train_regression, y_train_regression  = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)


for epoch in range(epochs):
    y_pred = model_2(X_train_regression)
    loss = loss_fn(y_pred,y_train_regression)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Testing
    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)
        test_loss = loss_fn(test_pred,y_test_regression)

    #print out
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}  | Test Loss: {test_loss:.5f}")

#Tuen on evaluation mode
model_2.eval()

#Make predictions
with torch.inference_mode():
    y_preds = model_2(X_test_regression)

# plot data 
from Helper_functions import plot_predictions, plot_decision_boundary

plot_predictions(train_data=X_train_regression.cpu(),train_labels= y_train_regression.cpu(), test_data=X_test_regression.cpu(), test_labels=y_test_regression.cpu(),predictions=y_preds.cpu())
plt.show()


