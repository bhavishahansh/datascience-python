'''
two types of classification
binary classification - ex - dog or cat, spam or not spam
muticlass classification - ex - it an bifurgate on single input to diff output can be expected
'''

# create a multiclass classification
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from torch import nn
from sklearn.model_selection import train_test_split

# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42 

#1. create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000, n_features= NUM_FEATURES, centers= NUM_CLASSES, cluster_std=1.5, #give the custers a little shakeup, 
                             random_state = RANDOM_SEED)

#2. Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.float)

#3 split into train and test
X_blob_train,X_blob_test,y_blob_train,y_blob_test = train_test_split(X_blob,y_blob,test_size=0.2,random_state = RANDOM_SEED)

#plot data
# plt.figure(figsize=(10,7))
# plt.scatter(X_blob[:,0],X_blob[:,1],c=y_blob,cmap=plt.cm.RdYlBu)
# plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"
#Build a multiclass classfication model
class BlobModel(nn.Module):
    def __init__(self,input_features, output_features,hidden_units = 8 ):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features = hidden_units ),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features = output_features )

        )

    def forward(self,x):
        # we are calling this bcz linear_layer_stack is type of nn.sequentail and this is callable model When you do this, PyTorch automatically:
        #Passes input_data through each layer in order Applies each layer’s .forward() internally
        return self.linear_layer_stack(x)

# create an instance of Blobmodel and sent it to the target device
model_4 = BlobModel(input_features=2, output_features=4).to(device)
#create a loss function for multilass lassifiation

loss_fun = nn.CrossEntropyLoss()
optimier = torch.optim.SGD(params=model_4.parameters(),lr=0.1)

#training loop for multilass lassifiation problem
torch.manual_seed(42)
torch.cuda.manual_seed(42)

loops = 100


model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test.to(device))
    y_pred_probs = torch.softmax(y_logits,dim=1)

#next(model_4.parameters)
strt from 13:52 - https://www.youtube.com/watch?v=V_xro1bcAuA&t=11404s

