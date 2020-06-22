import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
from baseline_model import Baseline

def getFrame(clip, sec):
    clip.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames, image = clip.read()
    # Convert image from 3 channels to 1 channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hasFrames, image

# Set seed for reproducibility
seed = 1
torch.manual_seed(seed)

input_file_prefix = "../../Data/Clipped/backhand/backspin/clip"

clips_list = []

num_clips = 300

# Get all clips into a list
for num in range(num_clips):
    file_name = input_file_prefix+str(num)+".mov"
    clips_list.append(cv2.VideoCapture(file_name))

input_file_prefix = "../../Data/Clipped/backhand/topspin/clip"

for num in range(num_clips):
    file_name = input_file_prefix+str(num)+".mov"
    clips_list.append(cv2.VideoCapture(file_name))

y_true = np.array([0]*num_clips + [1]*num_clips)


# Convert each clip into a series of images

frameRate = 0.1
num_frames = int(1.0/frameRate)
image_height = 1080
image_width = 1920
# Input tensor to model
X = np.zeros((num_clips*2 , num_frames, image_height, image_width))

for clip_num, clip in enumerate(clips_list):
    for sec_num, sec in enumerate(np.arange(0.0, 1.0, frameRate)):
        isSuccess, image = getFrame(clip, sec)
        if not isSuccess:
            print("Something went wrong :(")
            exit()

        X[clip_num, sec_num] = image

X = torch.from_numpy(X)
X = X.float()

y_true = torch.from_numpy(y_true)
y_true = y_true.float()

# Mini-batch size
bs = 10
epochs = 2
lr = 1e-3

# Store all training dataset in a single wrapped tensor
train_ds = TensorDataset(X, y_true)

# Use DataLoader to handle minibatches easily
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)

# Construct model
my_model = Baseline(image_height, image_width)
my_model = my_model.float()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(my_model.parameters(), lr=lr)

# Pass through Model

for epoch in range(epochs):
    my_model.train()
    for xb, yb in train_dl:

        # Forward pass
        y_pred = my_model.forward(X)
        # Compute CrossEntropyLoss
        loss = criterion(y_pred, y_true)

        # Zero gradients, backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Got here")

    # Report results at end of epoch
    my_model.eval()
    y_pred = my_model.forward(X)
    loss = criterion(y_pred, y_true)

    print("Epoch: ", epoch, "Loss: ", loss.item())


# Save the model to a file
file_path = 'baseline_trained_seed'+str(seed)+'.pt'
torch.save(my_model, file_path)
