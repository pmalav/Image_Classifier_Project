# TODO: Write a function that loads a checkpoint and rebuilds the model
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
import json
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--load_dir', action = 'store', type = str, default = 'checkpoint.pth', help = 'Load file directory')
parser.add_argument('--json_dir', action = 'store', type = str, default = 'ImageClassifier/cat_to_name.json', help = 'JSON file to map class values to category names' )
parser.add_argument('--image_dir', action = 'store', type = str, default = 'ImageClassifier/flowers/test/88/image_00540.jpg', help = 'Image test file for model prediction' )
parser.add_argument('--topk_classes', action = 'store', type = int, default = 5, help = 'Top K classes')
parser.add_argument('--gpu', action = 'store', default = 'cuda', help = 'Type of device to be used')

args = parser.parse_args()

with open(args.json_dir, 'r') as f:
    cat_to_name = json.load(f)
    
checkpoint = torch.load(args.load_dir)

if checkpoint['arch'] == 'vgg16':
    model = models.vgg16(pretrained=True)
if checkpoint['arch'] == 'vgg19':
    model = models.vgg19(pretrained=True)
if checkpoint['arch'] == 'alexnet': 
    model = models.alexnet(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(25088, checkpoint['hidden_units_1']),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(checkpoint['hidden_units_1'], checkpoint['hidden_units_2']),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(checkpoint['hidden_units_2'], checkpoint['hidden_units_3']),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(checkpoint['hidden_units_3'], 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), checkpoint['learning_rate'])

device = torch.device(args.gpu)
model.to(device);
    
model.load_state_dict(checkpoint['model_state_dict'])
model_class_to_idx = checkpoint['model_class_to_index']
model_index_to_class = dict([[x,y] for y,x in model_class_to_idx.items()])

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img.load()
    
    size = (256, 256)
    img.thumbnail(size)
    
    x1 = img.width/2 - 224/2
    x2 = img.width/2 + 224/2
    y1 = img.height/2 - 224/2
    y2 = img.height/2 + 224/2
    area = (x1, y1, x2, y2)
    img = img.crop(area)
    
    img = np.array(img)
    img = img/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean)/std
    
    final_image = img.transpose(2,0,1)
    final_image = torch.from_numpy(final_image).float() 
    
    return final_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    device = torch.device(args.gpu)
    model.to(device)
    model.eval()
    
    image = process_image(image_path)
    image.unsqueeze_(0)
    image = image.to(device)
    logps = model.forward(image)
    ps = torch.exp(logps)
    probabilities, indices = ps.topk(topk, dim=1)
    probabilities = probabilities[0].tolist() 
    indices = indices[0].tolist()
    
    classes = []
    for index in indices:
        classes.append(model_index_to_class[index])
    
    flower_names = []
    for element in classes:
        flower_names.append(cat_to_name[element])
        
    return probabilities, flower_names

probs, flower_names = predict(args.image_dir, model, args.topk_classes)
print("The most likely image class for the image is: {}\n".format(flower_names[0]),\
      "The associated probability is: {}\n".format(probs[0]),\
      "The Top K Classes along with their probabilites are:\n{}\n{}".format(flower_names, probs))