import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
from baseline_model import Baseline

def getFrame(clip, sec, image_height, image_width):
    clip.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames, image = clip.read()
    # Convert image from 3 channels to 1 channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Reduce image size
    dim = (image_width, image_height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return hasFrames, image

# Set seed for reproducibility
seed = 1
torch.manual_seed(seed)

# Set device
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()

input_file_prefix = "../../Data/Clipped/backhand/backspin/clip"

clips_list = []

# TODO - data structure to be changed to jumble and sort into folders the input data

start_clip = 0
end_clip = 50
num_clips = (end_clip - start_clip) * 2

# Get all clips into a list
for num in range(start_clip, end_clip):
    file_name = input_file_prefix+str(num)+".mov"
    clips_list.append(cv2.VideoCapture(file_name))

start_clip = 300
end_clip = 350

for num in range(start_clip, end_clip):
    file_name = input_file_prefix+str(num)+".mov"
    clips_list.append(cv2.VideoCapture(file_name))


input_file_prefix = "../../Data/Clipped/backhand/topspin/clip"

start_clip = 0
end_clip = 50

for num in range(start_clip, end_clip):
    file_name = input_file_prefix+str(num)+".mov"
    clips_list.append(cv2.VideoCapture(file_name))

start_clip = 300
end_clip = 350

for num in range(start_clip, end_clip):
    file_name = input_file_prefix+str(num)+".mov"
    clips_list.append(cv2.VideoCapture(file_name))

y_true = np.array([0]*num_clips + [1]*num_clips)


# Convert each clip into a series of images

frameRate = 0.1
num_frames = int(1.0/frameRate)
image_height = 540
image_width = 960
# Input tensor to model
X = np.zeros((num_clips*2 , num_frames, image_height, image_width))

for clip_num, clip in enumerate(clips_list):
    for sec_num, sec in enumerate(np.arange(0.0, 1.0, frameRate)):
        isSuccess, image = getFrame(clip, sec, image_height, image_width)
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
epochs = 8
lr = 1e-1

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
    total_loss = 0
    counter = 0
    for xb, yb in train_dl:

        # Forward pass
        y_pred = my_model.forward(xb)
        # Compute CrossEntropyLoss
        loss = criterion(y_pred, yb)

        # Zero gradients, backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        counter+=1
        print(counter)

    # Report results at end of epoch
    avg_loss = total_loss/counter

    print("Epoch: ", epoch, "Loss: ", avg_loss)


# Save the model to a file
file_path = 'baseline_trained_seed'+str(seed)+'.pt'
torch.save(my_model, file_path)
