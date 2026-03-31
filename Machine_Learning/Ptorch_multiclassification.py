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
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

#3 split into train and test
X_blob_train,X_blob_test,y_blob_train,y_blob_test = train_test_split(X_blob,y_blob,test_size=0.2,random_state = RANDOM_SEED)

#plot data
# plt.figure(figsize=(10,7))
# plt.scatter(X_blob[:,0],X_blob[:,1],c=y_blob,cmap=plt.cm.RdYlBu)
# plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

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

epochs = 100

# put data to target device
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    model_4.train()
    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fun(y_logits,y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train, y_pred=y_pred)

    optimier.zero_grad()
    loss.backward()
    optimier.step()   


    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test.to(device))
        test_preds = torch.softmax(test_logits,dim=1).argmax(dim=1)

        test_loss = loss_fun(test_logits,y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test, y_pred=test_preds)

    if epochs % 10 == 0:
        print(f"Epoch: {epochs} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

#Making predictions
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)

#go from logits --> Predictions
y_pred_probs = torch.softmax(y_logits, dim=1)

#go from pred probs to pred labels
y_preds = torch.argmax(y_pred_probs, dim=1)

from Helper_functions import plot_predictions, plot_decision_boundary

plt.figure(figsize=(12,16))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_4,X_blob_train,y_blob_train)


plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_4,X_blob_test,y_blob_test)
plt.show()

# classification matrics  (to evaluate our classification model) search for this
'''
* accuracy
* Precision
* Recall
* F1-score
* confusion matrix
* classification report

FP = false positive
FN = false negative

| Metric                | Measures                                  | Best for                          |
| --------------------- | ----------------------------------------- | --------------------------------- |
| Accuracy              | Overall correctness                       | Balanced datasets                 |
| Precision             | How many predicted positives were correct | When FP is costly                 |
| Recall                | How many actual positives found           | When FN is costly                 |
| F1-score              | Balance of precision & recall             | Imbalanced datasets               |
| Confusion Matrix      | What errors happen where                  | Understanding class-wise mistakes |
| Classification Report | Detailed metrics per class                | Multi-class evaluation            |

TorchMetrics is a library built by PyTorch Lightning that includes:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

AUC / ROC

IoU (for segmentation)

BLEU / ROUGE (for NLP)

PSNR / SSIM (for image quality)

Many more…

'''
from torchmetrics import Accuracy

torchmetrics_accuracy = Accuracy().to(device)

torchmetrics_accuracy(y_preds, y_blob_test)















