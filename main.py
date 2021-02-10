import torch
import numpy as np
import cv2
import torchvision
from PIL import Image



def people_on_image(path_to_image):

    color_map = [
               (255, 255, 255),  # background
               (255, 255, 255), # aeroplane
               (255, 255, 255), # bicycle
               (255, 255, 255), # bird
               (255, 255, 255), # boat
               (255, 255, 255), # bottle
               (255, 255, 255), # bus
               (255, 255, 255), # car
               (255, 255, 255), # cat
               (255, 255, 255), # chair
               (255, 255, 255), # cow
               (255, 255, 255), # dining table
               (255, 255, 255), # dog
               (255, 255, 255), # horse
               (255, 255, 255), # motorbike
               (255, 0, 0), # person
               (255, 255, 255), # potted plant
               (255, 255, 255), # sheep
               (255, 255, 255), # sofa
               (255, 255, 255), # train
               (255, 255, 255) # tv/monitor
    ]
    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize(540),
        torchvision.transforms.CenterCrop(520),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
    model.eval()

    image = Image.open(path_to_image)
    image = trans(image)
    image = image.unsqueeze(0)
    out = model(image)

    labels = torch.argmax(out['out'].squeeze(), dim=0).detach().cpu().numpy()

    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(color_map)):
        index = labels == label_num
        red_map[index] = np.array(color_map)[label_num, 0]
        blue_map[index] = np.array(color_map)[label_num, 1]
        green_map[index] = np.array(color_map)[label_num, 2]

    ready_image = np.stack([red_map, green_map, blue_map], axis=2)

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ready_image = cv2.cvtColor(ready_image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(ready_image, 0.6, image, 0.4, 0)
    return ready_image

