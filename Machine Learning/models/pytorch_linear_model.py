import torch
import torch.nn as nn
import matplotlib.pyplot as plt

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8 * len(X))
X_train= X[:train_split]
y_train  = y[:train_split]

X_test = X[train_split:]
y_test = y[train_split:]

class LinearRegressionModel_v2(nn.Module):
    def __init__(self):
        super().__init__()
        #use nn.Linear() for creating the modwl parameters / also called linear tranform, probing layer, fully connected layer, dense
        self.linear_layer = nn.Linear(in_features=1,out_features=1)
    def forward(self,x: torch.Tensor)-> torch.Tensor:
        return self.linear_layer(x)

#set the manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModel_v2()
#print(model_1,model_1.state_dict())

device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"using device {device}")
#print(next(model_1.parameters()).device)
model_1.to(device)
#print(next(model_1.parameters()).device)

# setup loss function
loss_fn = nn.L1Loss()

#setup our optimizer
optimizer = torch.optim.SGD(params = model_1.parameters(),lr=0.01)

#let's write a training loop
torch.manual_seed(42)
epochs = 200

#put data on the target device (device agnostic code for data)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)


for epoch in range(epochs):
    model_1.train()

    # forward pass
    y_pred = model_1(X_train)

    #calculate the loss
    loss = loss_fn(y_pred,y_train)

    # optimizer zero
    optimizer.zero_grad()

    # backpropogation
    loss.backward()

    #optimizer step
    optimizer.step()

    ### switching model to testing mode
    model_1.eval()
    with torch.inference_mode():
        text_pred = model_1(X_test)
        test_loss = loss_fn(text_pred,y_test)

    if epoch % 10 == 0:
        print(f"EPOCH :{epoch} | Loss {loss} || Test loss: {test_loss}")

print(model_1.state_dict())

def plot_prediction(train_data=X_train, train_labels=y_train,test_data=X_test,test_labels=y_test,prediction=None):
    plt.figure(figsize=(10,7))

    train_data = train_data.cpu()
    train_labels = train_labels.cpu()
    test_data = test_data.cpu()
    test_labels = test_labels.cpu()


    #ploting training data into blue
    plt.scatter(train_data,train_labels,c="b",s=4,label="Training data")

    #plot test data in green
    plt.scatter(test_data,test_labels,c="g",s=4,label="Testing data")

    #Are there predictions?
    if prediction is not None:
        #Plot the predition if they exist
        plt.scatter(test_data,prediction,c="r",s=4,label="Prediction")

    plt.legend(prop={"size":14})
    plt.show()

#     plot_prediction(X_train,y_train,X_test,y_test)

model_1.eval()

with torch.inference_mode():
    y_preds = model_1(X_test)

plot_prediction(prediction=y_preds.cpu())

