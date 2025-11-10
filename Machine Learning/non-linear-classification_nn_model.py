import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import torch
from torch import nn
from sklearn.model_selection import train_test_split
#sklearn always returns numoy we need to convert it later to usein pytorch to tensors
#Make 1000 samples

n_samples = 1000

#create circles
X,y = make_circles(n_samples,noise=0.03,random_state=42)

plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.RdYlBu)

# convert data to tensors and then to train and test splits

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")

#Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

#Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42) 

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

#build a model using non linear function in nn

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features= 2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # relu is the non linear activation function
        #self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CircleModelV2().to(device)

# visit playground.tensorflow.org and set the parameters and check whether its working with circular data
# setup loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(),lr=0.1)

# Train model with non linear activation function
torch.manual_seed(42)
torch.cuda.manual_seed(42)

X_train , y_train = X_train.to(device), y_train.to(device)
X_test,y_test = X_test.to(device), y_test.to(device)

# Train for longer
epochs = 1000

# for epoch in range(epochs):
#     model_3.train()

#     # 1. Forward pass
#     y_logits = model_3(X_train).squeeze()
#     y_pred = torch.round(torch.sigmoid(y_logits)) # logits --> prediction probabilities 

#     #2. Calculate the loss
#     loss = loss_fn(y_logits,y_train) # BCEwith Logit loss
#     acc = accuracy_fn(y_true = y_train, y_pred=y_pred)

#     #3. Optimizer zero grad - before going to next epoch reseting gredient descent
#     optimizer.zero_grad()

#     #4. loss backwards
#     loss.backward()

#     #5 step the optimizer
#     optimizer.step()

#     #6 training mode
#     model_3.eval()
#     with torch.inference_mode():
#         test_logits = model_3(X_test).squeeze()
#         test_pred = torch.round(torch.sigmoid(test_logits))

#         test_loss = loss_fn(test_logits,y_test)
#         test_acc = accuracy_fn(y_true = y_test, y_pred=test_pred)

#     #print out
#     if epoch % 100 == 0:
#         print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# evaluaing a model trained with non-linear activation function

model_3.eval()
with torch.inference_mode():
       y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()

plt.figure(figsize=(12,6))
plt.subplot(1,2,1) # 1row, 2column and 1st graph
plt.title("Train")

from Helper_functions import plot_predictions, plot_decision_boundary
plot_decision_boundary(model_3,X_train,y_train)

plt.subplot(1,2,2) # 1row, 2column and 2nd graph
plt.title("Test")
plot_decision_boundary(model_3,X_test,y_test)
plt.show()
# Now here we are replcing non-linear activation function
A = torch.arange(-10, 10, 1, dtype=torch.float32)

# if you plot this numbers on graph then it will ploton curve
def relu(x: torch.Tensor) -> torch.Tensor:
      return torch.maximum(torch.tensor(0), x)

# if you plot this numbers on graph then it will ploton curve
def sigmoid(x):
     return 1 / (1 + torch.exp(-x))