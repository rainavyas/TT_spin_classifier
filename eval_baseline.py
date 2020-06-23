import torch
import cv2
from baseline_model import Baseline
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np

def getFrame(clip, sec, image_height, image_width):
    clip.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames, image = clip.read()
    # Convert image from 3 channels to 1 channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Reduce image size
    dim = (image_width, image_height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return hasFrames, image

# Load up the trained model
model_path = 'baseline_trained_seed1.pt'
model = torch.load(model_path)
model.eval()

input_file_prefix = "../../Data/Clipped/backhand/backspin/clip"

clips_list = []

# TODO - data structure to be changed to jumble and sort into folders the input data

start_clip = 50
end_clip = 100
num_clips = (end_clip - start_clip) * 2

# Get all clips into a list
for num in range(start_clip, end_clip):
    file_name = input_file_prefix+str(num)+".mov"
    clips_list.append(cv2.VideoCapture(file_name))

start_clip = 250
end_clip = 300

for num in range(start_clip, end_clip):
    file_name = input_file_prefix+str(num)+".mov"
    clips_list.append(cv2.VideoCapture(file_name))


input_file_prefix = "../../Data/Clipped/backhand/topspin/clip"

start_clip = 50
end_clip = 100

for num in range(start_clip, end_clip):
    file_name = input_file_prefix+str(num)+".mov"
    clips_list.append(cv2.VideoCapture(file_name))

start_clip = 250
end_clip = 300

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

bs = 5
train_ds = TensorDataset(X, y_true)
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)


trueTop_predTop = 0
trueTop_predBack = 0
trueBack_predBack = 0
trueBack_predTop = 0


# Assume threshold is 0.5 for now

for xb, yb in train_dl:

    # Forward pass
    y_pred = model.forward(xb)

    y_pred_list = y_pred.tolist()
    y_true_list = yb.tolist()

    for yp, yt in zip(y_pred_list, y_true_list):
        if yp > 0.5:
            pred = 'top'

        else:
            pred = 'back'

        if pred == 'top' and yt == 1:
            trueTop_predTop += 1
        elif pred == 'top' and yt == 0:
            trueBack_predTop += 1
        elif pred == 'back' and yt == 1:
            trueBack_predTop += 1
        elif pred == 'back' and yt == 0:
            trueBack_predBack += 1

print("True Positive: ", trueTop_predTop)
print("False Negative: ", trueTop_predBack)
print("True Negative: ", trueBack_predBack)
print("False Positive: ", trueBack_predTop)
