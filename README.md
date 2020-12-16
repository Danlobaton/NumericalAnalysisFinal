# Optical Character Recognition

## Packages
Python v3.6.1

1. PyTorch
2. PIL
3. NumPy
4. TorchVision

## Train.py
Uncomment line 162 to train model using the train function
Train:
  Parameters
  1. images_path : File path to .npy file containing all image files to be used in training  [string]
  2. labels_path : File path to .npy file containing training labels for the provided traing images [string]
  3. validation_percent : validation split percetange to be used during traing [float] initially set to 0.2
  4. epochs : Number of of epochs for training [int] initially set to 30
  5. model_name : File name in which trained model will be saved in. Must in in .pth [string]
  
## Test.py
Uncomment line 26 to test model.
Use provided Final_Model.pth file for model_path parameter to test highest performing model trained in a Cloud GPU server
 Parameters:
 1. path_x : path to images file to test
 2. model_path : path to saved neural network model to be used for testing. Must end in .pth [string]
 3. certainty_threshold : confidence threshold to be used to unknown image detection [int] originally set to 5
 
 Generates a .npy file called predicted.npy containing all predicted labels
 RETURNS: vector of predicted labels as a NumPy array
