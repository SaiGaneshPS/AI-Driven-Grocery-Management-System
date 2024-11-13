Currently, the app has a main welcome screen which takes the user to a screen with three buttons:

1. Open Camera Screen: This screan is used to capture the receipts, check the bounding boxes and capture the receipt if the user is satisfied.
2. Inventory Screen: This screen shows existing inventory based on a json file stored on the device.
3. Generate Recipe Screen: This screen generates the custom recipes based on inventory available.

For the usage of heavier models such as FasterRCNN and GPT-2, a 'server.py' file is used. Requests are made to and from this server with the help of
Flask. 
