import torch
import cv2
from baseline_model import Baseline
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def getFrame(clip, sec, image_height, image_width):
    clip.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames, image = clip.read()
    # Convert image from 3 channels to 1 channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Reduce image size
    dim = (image_width, image_height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # Normalise frame such that it has zero mean and unit variance
    image = image.astype(np.float32) / 255
    image -= image.mean()
    image /= image.std()

    return hasFrames, image

# Load up the trained model
model_path = 'youtube_normalised_trained_seed1.pt'
model = torch.load(model_path)
model.eval()

input_file_prefix = "../../Data/Clipped/Youtube/Backhand/Evaluation/Back/clip"

clips_list = []


start_clip = 30
end_clip = 50
num_clips = end_clip - start_clip

# Backspins
# Get all clips into a list
for num in range(start_clip, end_clip):
    file_name = input_file_prefix+str(num)+".mov"
    clips_list.append(cv2.VideoCapture(file_name))

# Repeat for top spins
input_file_prefix = "../../Data/Clipped/Youtube/Backhand/Evaluation/Top/clip"

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


y_ts = []
y_ps = []

for xb, yb in train_dl:

    # Forward pass
    y_pred = model.forward(xb)

    y_pred_list = y_pred.tolist()
    y_true_list = yb.tolist()

    y_ts += y_true_list
    y_ps += y_pred_list


# Collect values for graph plot
accs = []
threshs = []

# Find best threshold to separate into back and top spin
best = {'acc': 0, 'TruePositive':0, 'FalseNegative': 0, 'TrueNegative':0, 'FalsePositive':0, 'threshold':0}

for thresh in np.arange(0, 1, 0.02):

    trueTop_predTop = 0
    trueTop_predBack = 0
    trueBack_predBack = 0
    trueBack_predTop = 0

    for yp, yt in zip(y_ps, y_ts):
        #print(yp)
        if yp > thresh:
            pred = 'top'

        else:
            pred = 'back'

        if pred == 'top' and yt == 1:
            trueTop_predTop += 1
        elif pred == 'top' and yt == 0:
            trueBack_predTop += 1
        elif pred == 'back' and yt == 1:
            trueTop_predBack += 1
        elif pred == 'back' and yt == 0:
            trueBack_predBack += 1

    acc = (trueTop_predTop + trueBack_predBack)/(trueTop_predTop + trueBack_predBack+trueTop_predBack+trueBack_predTop)
    if acc > best['acc']:
        best['acc'] = acc
        best['TruePositive'] = trueTop_predTop
        best['FalseNegative'] = trueTop_predBack
        best['TrueNegative'] = trueBack_predBack
        best['FalsePositive'] = trueBack_predTop
        best['threshold'] = thresh

    accs.append(acc)
    threshs.append(thresh)
print(best)

# Plot accuracy vs threshold graph
plt.plot(threshs, accs)
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.show()
