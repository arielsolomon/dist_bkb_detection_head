import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.resnet import resnet18
import glob, os
from PIL import Image
import cv2

# Load a pre-trained ResNet-18 model

resnet18_backbone = resnet18(pretrained=True)

# Remove the fully connected layers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modules = list(resnet18_backbone.children())[:-2]
resnet18_backbone = torch.nn.Sequential(*modules)
resnet18_backbone.out_channels = 512  # Number of output channels for ResNet-18

#Define the Anchor Generator

anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),))

# Define the ROI Align Layer
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'],
    output_size=7,
    sampling_ratio=2)


mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'],
    output_size=14,
    sampling_ratio=2)

#Create the Mask R-CNN Model

model = MaskRCNN(
    resnet18_backbone,
    num_classes=91,  # Number of classes (including the background)
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler,
    mask_roi_pool=mask_roi_pooler)
model.to(device)
#Train and Evaluate the Model

# Dummy input to check if the model works
root = '/Data/federated_learning/dist_bkb_detection_head/data/images/test/'
images = [torch.rand(3, 800, 800)]  # Example with a batch of one image
img_list = glob.glob(os.path.join(root, '*jpg'))
model.eval()
conv_image = transforms.ToTensor()
image_list = []
for image in img_list[:5]:
    image1 = Image.open(image)
    image = conv_image(image1)
    image_list.append(image.to(device))
ouputs = model(image_list)
print(ouputs)
