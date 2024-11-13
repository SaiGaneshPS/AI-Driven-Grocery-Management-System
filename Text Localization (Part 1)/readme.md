The train.py code was used to compare and evaluate the folllowing models.

- fasterrcnn_resnet50_fpn, 
- fasterrcnn_mobilenet_v3_large_fpn, 
- fcos_resnet50_fpn, 
- retinanet_resnet50_fpn_v2,
- retinanet_resnet50_fpn, 
- ssd300_vgg16, 
- ssdlite320_mobilenet_v3_large

The dataset contains 626 annotated images out of which 20% was used for test dataloader.
A sample image with the bounding boxes plotted looks like this:
![image](https://github.com/user-attachments/assets/195c8ecc-9878-41f8-a7c2-6091aa4a8288)

A sample prediction using FastRCNN model looks like this:
![image](https://github.com/user-attachments/assets/3ef2471e-09d4-48e4-899b-2a9770face2c)
