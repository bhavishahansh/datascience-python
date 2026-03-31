import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.version.cuda)  # Should NOT be None
print(torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")

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

# Define a custom Linear Regression model using nn.Parameter
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize weight and bias manually as trainable parameters
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x:torch.Tensor):
        # Manually compute the linear transformation: y = xW^T + b
        return self.weight * x + self.bias

torch.manual_seed(42)
model_0 = LinearRegressionModel()

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params = model_0.parameters(),lr=0.01)

epochs = 100

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_fn(y_pred,y_train)
    #print(f"Loss: {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()
    # in inference_mode (testing the model) disable auto_grad means automtic gradient decent calculation will not execute in background
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred,y_test)

    #if epoch % 10 == 0:
        #print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")        
        #print(model_0.state_dict())
    
# Create a models Directory
print(model_0.state_dict())
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Create model save path

MODEL_NAME = "PyTorch_linear_regression_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Load the saved state_dict of model_0 this update the model with new params so below loaded_model_0.state_dict() having
# random prameters but ater below line it wlll update to new model varibel 

print(loaded_model_0.state_dict())

# predection using saved model
loaded_model_0.eval()
with torch.inference_mode():
    loded_model_pred = loaded_model_0(X_test)
print(loded_model_pred)    

