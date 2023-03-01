#!/usr/bin/env python
# coding: utf-8

# ## Assessment 1 - Image Classification [100 marks]
# 
# <div class="logos"><img src="https://drive.google.com/uc?id=132BXgkV5w1bpXlVpdr5BtZdpagqYvna7" width="220px" align="right"></div>
# 
# 
# 
# ### Motivation 
# 
# Through this assessment, you will gain practical experience in:
# 
# > 1. Implementing and evaluating a multi-layer perceptron (MLP) and convolutional neural network (CNN) in solving a classification problem
# > 2. Building, evaluating, and finetuning a CNN on an image dataset from development to testing 
# > 3. Tackling overfitting using strategies such as data augmentation and drop out
# > 4. Fine tuning a model 
# > 5. Comparing the performance of a new model with an off-the-shelf model (AlexNet)
# > 6. Gaining a deeper understanding of model performance using visualisations from Grad-CAM.
# 
# 
# ### Setup and resources 
# 
# You must work using this template notebook.
# 
# Having a GPU will speed up the training process. See the provided document on Minerva about setting up a working environment for various ways to access a GPU. We highly recommend you use platforms such as Colab.
# 
# Please implement the coursework using **Python and PyTorch**, and refer to the notebooks and exercises provided.
# 
# This coursework will use a subset of images from Tiny ImageNet, which is a subset of the [ImageNet dataset](https://www.image-net.org/update-mar-11-2021.php). Our subset of Tiny ImageNet contains **30 different categories**, we will refer to it as TinyImageNet30. The training set has 450 resized images (64x64 pixels) for each category (13,500 images in total). You can download the training and test set from a direct link or the Kaggle challenge website:
# 
# >[Direct access to data is possible by clicking here, please use your university email to access this](https://leeds365-my.sharepoint.com/:u:/g/personal/scssali_leeds_ac_uk/ESF87mN6kelIkjdISkaRow8BublW27jB-P8eWV6Rr4rxtw?e=SPASDB)
# 
# >[Access data through Kaggle webpage](https://www.kaggle.com/t/917fe52f6a3c4855880a24b34f26db07) 
# 
# 
# ### Required submissions
# 
# ##### 1. Kaggle Competition
# To participate in the submission of test results, you will need an account. Even if you have an existing Kaggle account, please carefully adhere to these instructions, or we may not be able to locate your entries:
# 
# > 1. Use your **university email** to register a new account.
# > 2. Set your **Kaggle account NAME** to your university username.
# 
# The class Kaggle competition also includes a blind test set, which will be used in Question 1 for evaluating your custom model's performance on a test set. The competition website will compute the test set accuracy, as well as position your model on the class leaderboard. [Link to submit your results on Kaggle competition](https://www.kaggle.com/competitions/comp5623m-artificial-intelligence/submissions). 
# 
# Please submit only your predictions from test set - detailed instructions are provided in (3)
# 
# ##### 2. Submission of your work
# 
# Please submit the following:
# 
# > 1. Your completed Jupyter notebook file, without removing anything in the template, in **.ipynb format.**
# > 2. The **.html version** of your notebook; File > Download as > HTML (.html). Check that all cells have been run and all outputs (including all graphs you would like to be marked) displayed in the .html for marking.
# > 3. Your selected images from section 6 "Failure/success analysis" (outputs from gradcam, for example you can put these images into failure and succcess folders).
# 
# **Final note:**
# 
# > **Please display everything that you would like to be marked. Under each section, put the relevant code containing your solution. You may re-use functions you defined previously, but any new code must be in the relevant section.** Feel free to add as many code cells as you need under each section.
# 

# ## Required packages
# 
# [1] [numpy](http://www.numpy.org) is a package for scientific computing with python
# 
# [2] [h5py](http://www.h5py.org) is a package to interact with compactly stored dataset
# 
# [3] [matplotlib](http://matplotlib.org) can be used for plotting graphs in python
# 
# [4] [pytorch](https://pytorch.org/docs/stable/index.html) is a library widely used for bulding deep-learning frameworks
# 
# Feel free to add to this section as needed - examples of importing libraries are provided below.
# 
# You may need to install these packages using [pip](https://pypi.org/project/opencv-python/) or [conda](https://anaconda.org/conda-forge/opencv).

# In[8]:


import math
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import VisionDataset
from PIL import Image
import matplotlib.pyplot as plt


# In[9]:


# always check your version
print(torch.__version__)


# One challenge of building a deep learning model is to choose an architecture that can learn the features in the dataset without being unnecessarily complex. The first part of the coursework involves building a CNN and training it on TinyImageNet30. 
# 
# ### **Overview:**
# 
# **1. Function implementation** (12 marks)
# 
# *   **1.1** PyTorch ```Dataset``` and ```DataLoader``` classes (4 marks)
# *   **1.2** PyTorch ```Model``` class for a simple MLP model (4 marks)
# *   **1.3** PyTorch ```Model``` class for a simple CNN model (4 marks)
# 
# **2. Model training** (20 marks)
# *   **2.1** Train on TinyImageNet30 dataset (7 marks)
# *   **2.2** Generate confusion matrices and ROC curves (4 marks)
# *   **2.3** Strategies for tackling overfitting (9 marks)
#     *   **2.3.1** Data augmentation
#     *   **2.3.2** Dropout
#     *   **2.3.3** Hyperparameter tuning (e.g. changing learning rate)
#             
# 
# **3. Model Fine-tuning on CIFAR10 dataset** (20 marks)
# *   **3.1** Fine-tune your model (initialise your model with pretrained weights from (2)) (8 marks)
# *   **3.2** Fine-tune model with frozen base convolution layers (8 marks)
# *   **3.3** Compare complete model retraining with pretrained weights and with frozen layers. Comment on what you observe? (4 marks) 
# 
# **4. Model testing** (18 marks)
# *   **4.1**   Test your final model in (2) on test set - code to do this (10 marks)
# *   **4.2**   Upload your result to Kaggle  (8 marks)
# 
# **5. Model comparison** (14 marks)
# *   **5.1**   Load pretrained AlexNet and finetune on TinyImageNet30 until model convergence (6 marks)
# *   **5.2**   Compare the results of your CNN model with pretrained AlexNet on the same validation set. Provide performance values (loss graph, confusion matrix, top-1 accuracy, execution time) (8 marks)
# 
# **6. Interpretation of results** (16 marks)
# *   **6.1** Use grad-CAM on your model and on AlexNet (6 marks)
# *   **6.2** Visualise and compare the results from your model and from AlexNet (4 marks)
# *   **6.3** Comment on (6 marks):
#     - why the network predictions were correct or not correct in your predictions? 
#     - what can you do to improve your results further?

# ## 1 Function implementations [12 marks]
# 

# ### 1.1 Dataset class [4 marks]
# 
# Write a PyTorch ```Dataset``` class (an example [here](https://www.askpython.com/python-modules/pytorch-custom-datasets) for reference) which loads the TinyImage30 dataset and ```DataLoaders``` for training and validation sets.

# In[2]:


# TO COMPLETE
from natsort import natsorted
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
#module 'torchvision.datasets' has no attribute 'TinyImage30'
#Creating Custom Datasets in PyTorch
class TinyImage30Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # List all images in folder and count them
        all_imgs = os.listdir(root_dir)
        self.total_imgs = natsorted(all_imgs)
        # self.dataset = datasets.ImageFolder(root=self.root_dir, transform=self.transform)
        pass
#Defining __len__ function This function will allow us to identify the number of items that have been successfully loaded from our custom dataset.     
    def __len__(self):
        return len(self.total_imgs) #len(self.dataset)
#Defining __getitem__ function  
    def __getitem__(self, idx):
        img_loc = os.path.join(self.root_dir, self.total_imgs[idx])
        # Use PIL for image loading
        image = Image.open(img_loc).convert("RGB")
        # Apply the transformations
        tensor_image = self.transform(image)
        return tensor_image #self.dataset[idx]

# Set the root directory where the TinyImage30 dataset is located
root_dir = 'E:\\vencen\\Project\\Pycharm\\geng\\TinyImage30'

# Define your result path in the Mydrive --> avoids re-running the same network after session 

ROOT = '.' + os.sep
# uncomment if you are using colab
# ROOT = '/content/drive/MyDrive/' 

DataPath = ROOT + 'TinyImage30'
ResultPath = ROOT + 'E:\\vencen\\Project\\Pycharm\\geng\\TinyImage30'
# Define transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # 归一化 (0, 1)  -> (-1, 1)  0 margin 0.5
    #mean std
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 标准化
#   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) ref

])


# Create training and validation datasets
# train_dataset = TinyImage30Dataset(root_dir=root_dir+'\\train_set', transform=transform)
train_dataset = ImageFolder(root=root_dir+'\\train_set\\train_set', transform=transform)
test_dataset = TinyImage30Dataset(root_dir=root_dir+'\\test_set\\test_set', transform=transform)

# Create DataLoaders for training and validation datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# prints shape of image with single batch
print(f"train_loader:{next(iter(train_loader))[0].shape}")
print(f"test_loader:{next(iter(test_loader)).shape}")


# ### 1.2 Define a MLP model class [4 marks]
# 
# <u>Create a new model class using a combination of:</u>
# - Input Units
# - Hidden Units
# - Output Units
# - Activation functions
# - Loss function
# - Optimiser

# In[1]:


# TO COMPLETE
# define a MLP Model class
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

# ### 1.3 Define a CNN model class [4 marks]
# 
# <u>Create a new model class using a combination of:</u>
# - Convolution layers
# - Activation functions (e.g. ReLU)
# - Maxpooling layers
# - Fully connected layers 
# - Loss function
# - Optimiser

# In[ ]:

import torch.nn as nn
import torch.optim as optim

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool1(x)
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the model hyperparameters
learning_rate = 0.01

# Instantiate the model and define the loss function and optimizer
model = CustomCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# ## 2 Model training [20 marks]
# 
# 
# ### 2.1 Train both MLP and CNN models - show loss and accuracy graphs side by side [7 marks]
# 
# Train your model on the TinyImageNet30 dataset. Split the data into train and validation sets to determine when to stop training. Use seed at 0 for reproducibility and test_ratio=0.2 (validation data)
# 
# Display the graph of training and validation loss over epochs and accuracy over epochs to show how you determined the optimal number of training epochs. Top-*k* accuracy implementation is provided for you below.
# 
# > Please leave the graph clearly displayed. Please use the same graph to plot graphs for both train and validation.
# 

# In[10]:

net1 = nn.Sequential(
    nn.Flatten(),
    # single layer
    # nn.Linear(28*28, 10)
    nn.Linear(224*224*3, 32)
)

for param in net1.parameters():
    print(param.shape)

# In[11]:

net2 = nn.Sequential(
    nn.Flatten(),
    # single layer
    #nn.Linear(28*28, 10)
    # two layers
    nn.Linear(224*224*3, 300),
    nn.Sigmoid(),
    nn.Linear(300, 10)
)

for param in net2.parameters():
    print(param.shape)


# In[12]:


# Define top-*k* accuracy 
def topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# In[13]:


#TO COMPLETE --> Running you MLP model class
from torch import optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import os

def timshow(x):
    xa = np.transpose(x.numpy(),(1,2,0))
    plt.imshow(xa)
    plt.axis('off')
    plt.show()
    return xa

# Train the MLP model
nepochs = 100    # number of epochs --> you can change as you want, for e.g., try 200 epochs
net = net1       # 1-layer model
results_path = ResultPath + 'linear1layer200epochs.pt'
# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
# initialise ndarray to store the mean loss in each epoch (on the training data)
losses = np.zeros(nepochs)
# Use a loss function and optimiser provided as part of PyTorch.
# The chosen optimiser (Stochastic Gradient Descent with momentum) needs only to be given the parameters (weights and biases)
# of the network and updates these when asked to perform an optimisation step below.
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(nepochs):  # loop over the dataset multiple times

    # initialise variables for mean loss calculation
    running_loss = 0.0
    n = 0
    
    for data in train_loader:#train_dataset:
        inputs, labels = data
        
        # Zero the parameter gradients to remove accumulated gradient from a previous iteration.
        optimizer.zero_grad()

        # Forward, backward, and update parameters
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    
        # accumulate loss and increment minibatches
        running_loss += loss.item()
        n += 1
       
    # record the mean loss for this epoch and show progress
    losses[epoch] = running_loss / n
    print(f"epoch: {epoch+1} loss: {losses[epoch] : .3f}")
    
# save network parameters and losses
torch.save({"state_dict": net.state_dict(), "losses": losses}, results_path)


# In[6]:


# Your graph
dataiter =iter(test_loader)
images, labels = dataiter.next()

# 打印图片
timshow(torchvision.utils.make_grid(images))
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
print('GroundTruth: ', ' '.join('%5s'% classes[labels[j]] for j in range(4)))


# In[14]:


#TO COMPLETE --> Running you CNN model class
#TO COMPLETE --> Running you CNN model class
nepochs = 50
results_path = ROOT+'results/cnnclassifier50epochs.pt'
statsrec = np.zeros((4,nepochs))

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(nepochs):  # loop over the dataset multiple times
    correct = 0          # number of examples predicted correctly (for accuracy)
    total = 0            # number of examples
    running_loss = 0.0   # accumulated loss (for mean loss)
    n = 0                # number of minibatches
    for data in train_dataset:
        inputs, labels = data
        
         # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward, backward, and update parameters
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    
        # accumulate loss
        running_loss += loss.item()
        n += 1
        # 0 1 2 3
        # 0.1 0.2 0.4 0.7
        # 0 1 2 3
        # accumulate data for accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)    # add in the number of labels in this minibatch
        correct += (predicted == labels).sum().item()  # add in the number of correct labels
    
    # collect together statistics for this epoch
    ltrn = running_loss/n
    atrn = correct/total 
    ltst, atst = stats(test_loader, net)
    statsrec[:,epoch] = (ltrn, atrn, ltst, atst)
    print(f"epoch: {epoch} training loss: {ltrn: .3f} training accuracy: {atrn: .1%}  test loss: {ltst: .3f} test accuracy: {atst: .1%}")

# save network parameters, losses and accuracy
torch.save({"state_dict": net.state_dict(), "stats": statsrec}, results_path)




# In[15]:


# Your graph
# Your graph
results_path = ROOT+'results/cnnclassifier50epochs.pt'
data = torch.load(results_path)
statsrec = data["stats"]
fig, ax1 = plt.subplots()
plt.plot(statsrec[0], 'r', label = 'training loss', )
plt.plot(statsrec[2], 'g', label = 'test loss' )
plt.legend(loc='lower right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and test loss, and test accuracy')
ax2=ax1.twinx()
ax2.plot(statsrec[1], 'm', label = 'training accuracy')
ax2.plot(statsrec[3], 'b', label = 'test accuracy')
ax2.set_ylabel('accuracy')
plt.legend(loc='upper right')
fig.savefig("roc.svg")
plt.show()


# > Comment on your model and results that should include number of parameters in each model and why CNN over MLP for image classification task?

# ### 2.2 Generating confusion matrix and ROC curves [4 marks]
# - Use your CNN architecture with best accuracy to generate two confusion matrices, one for the training set and another for the validation set. Remember to use the whole validation and training sets, and to include all your relevant code. Display the confusion matrices in a meaningful way that clearly indicates what percentage of the data is represented in each position.
# - Display ROC curve for 5 top classes with area under the curve

# In[ ]:


# Your code here!


# **Note: All parts below here relate to the CNN model only and not the MLP! You are advised to use your final CNN model only for each of the following parts.**

# ### 2.3 Strategies for tackling overfitting (9 marks)
# Using your (final) CNN model, use the strategies below to avoid overfitting. You can reuse the network weights from previous training, often referred to as ``fine tuning``. 
# *   **2.3.1** Data augmentation
# *   **2.3.2** Dropout
# *   **2.3.3** Hyperparameter tuning (e.g. changing learning rate)
# 
# Plot loss and accuracy graphs per epoch side by side for each implemented strategy.

# #### 2.3.1 Data augmentation
# 
# > Implement at least five different data augmentation techniques that should include both photometric and geometric augmentations. 
# 
# > Provide graph and comment on what you observe
# 

# In[ ]:


# Your code here!


# #### 2.3.2 Dropout
# 
# > Implement dropout in your model 
# 
# > Provide graph and comment on your choice of proportion used

# In[ ]:


# Your code here!


# #### 2.3.3 Hyperparameter tuning
# 
# > Use learning rates [0.1, 0.001, 0.0001]
# 
# > Provide separate graphs for loss and accuracy, each showing performance at three different learning rates

# In[ ]:


# Your code here!


# In[ ]:


# Your graph


# ### 3 Model testing [18 marks]
# Online evaluation of your model performance on the test set.
# 
# > Prepare the dataloader for test set
# 
# > Write evaluation code for writing predictions
# 
# > Upload it to Kaggle submission page [link](https://www.kaggle.com/t/917fe52f6a3c4855880a24b34f26db07) 

# 
# #### 3.1 Test class and predictions [10 marks]
# 
# > Build a test class, prepare a test dataloader and generate predictions 
# 
# Create a PyTorch ```Dataset``` for the unlabeled test data in the test_set folder of the Kaggle competition and generate predictions using your final model. )

# In[ ]:


# Your code here!


# #### 3.2 Prepare your submission and upload to Kaggle [8 marks]
# 
# Save all test predictions to a CSV file and submit it to the private class Kaggle competition. **Please save your test CSV file submissions using your student username (the one with letters, ie., ````, not the ID with only numbers)**, for example, `mt20jb.csv`. That will help us to identify your submissions.
# 
# The CSV file must contain only two columns: ‘Id’ and ‘Category’ (predicted class ID) as shown below:
# 
# ```txt
# Id,Category
# 28d0f5e9_373c.JPEG,2
# bbe4895f_40bf.JPEG,18
# ```
# 
# Please note you will get marks for higher performance.
# 
# The ‘Id’ column should include the name of the image. It is important to keep the same name as the one on the test set. Do not include any path, just the name of file (with extension). Your csv file must contain 1501 rows, one for each image in the test set and 1 row for the headers. [To submit please visit](https://www.kaggle.com/t/917fe52f6a3c4855880a24b34f26db07)
# 
# > You may submit multiple times. We will use your personal top entry for allocating marks for this [8 marks]. The class leaderboard will not affect marking (brownie points!).
# 

# In[ ]:


# Your code here! 


# 
# ### 4 Model Fine-tuning/transfer learning on CIFAR10 dataset  [20 marks]
# 
# Fine-tuning is a way of applying or utilizing transfer learning. It is a process that takes a model that has already been trained for one task and then tunes or tweaks the model to make it perform a second similar task. You can perform finetuning in the following way:
# - Train an entire model from scratch (large dataset, more computation)
# - Freeze convolution base and train only last FC layers (small dataset and lower computation) 
# 
# > **Configuring your dataset**
#    - Download your dataset using ``torchvision.datasets.CIFAR10``, [explained here](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)
#    - Split training dataset into training and validation set similar to above. *Note that the number of categories here are only 10*
# 

# In[ ]:


# Your code here! 


# > Load pretrained AlexNet from PyTorch - use model copies to apply transfer learning in different configurations

# In[ ]:


# Your code here! 


# #### 4.1 Apply transfer learning initialise with pretrained model weights
# Use pretrained weights from AlexNet only (on the right of figure) to initialise your model. 
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/c/cc/Comparison_image_neural_networks.svg" style="width:1000px;height:400px;">
# <caption><center> <u>Figure</u>: Two models are given here: LeNet and AlexNet for image classification. However, you have to use **only AlexNet**.</center></caption>
# 
# 
# > Configuration 1: No frozen layers

# In[ ]:


# Your model changes here - also print trainable parameters


# #### 4.2 Fine-tuning model with frozen layers
# 
# > Configuration 2: Frozen base convolution blocks

# In[ ]:


# Your changes here - also print trainable parameters


# #### 4.3 Compare above configurations and comment on comparative performance

# In[ ]:


# Your graphs here and please provide comment in markdown in another cell


# ### 5 Model comparisons
# We often need to compare our model with other state-of-the-art methods to understand how well it performs compared to existing architectures. Here you will thus compare your model design with AlexNet on the TinyImageNet30 dataset

# #### 5.1 Finetune AlexNet on TinyImageNet30
# > Load AlexNet as you did above
# 
# > Train AlexNet on TinyImageNet30 dataset until convergence. Make sure you use the same dataset

# In[ ]:


# Your code here! 


# #### 5.2 Compare results on validation set of TinyImageNet30
# > Loss graph, top1 accuracy, confusion matrix and execution time for your model (say, mymodel and AlexNet)
# 

# In[ ]:


# Your code here! 


# ### 6 Interpretation of results (16 marks)
# 
# > Please use TinyImageNet30 dataset for all results

# 
# #### 6.1-6.2 Implement grad-CAM and visualise results (10 marks)
# 
# - Use an existing library to initiate grad-CAM 
# 
#         - To install: !pip install torchcam
#         - Call SmoothGradCAMpp: from torchcam.methods import SmoothGradCAMpp
#         - Apply to your model 
# 
# You can see the details here: https://github.com/frgfm/torch-cam
# 
# - Apply grad-CAM to your model on at least four correctly classified images
# - Apply grad-CAM on retrained AlexNet on at least four incorrectly classified images
# 
# >It is recommended to first read the relevant paper [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391), and refer to relevant course material.
# 
# 
# **HINT for displaying images with grad-CAM:**
# 
# Display ```heatmap``` as a coloured heatmap superimposed onto the original image. We recommend the following steps to get a clear meaningful display. 
# 
# From torchcam.utils import overlay_mask. But remember to resize your image, normalise it and put a 1 for the batch dimension (e.g, [1, 3, 224, 224]) 
# 

# In[ ]:


# Your code here!


# In[ ]:


# Your code here!


# In[ ]:


# Your code here!


# #### 6.3 Your comments on (6 marks):
# > a) Why model predictions were correct or incorrect? You can support your case from 6.2
# 
# > b) What can you do to improve your results further?

# ---> Double click to respond here

# **Please refer to the submission section at the top of this notebook to prepare your submission. Use our teams channel to seek any help!**
# 
