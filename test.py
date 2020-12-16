import torch
import numpy as np
from train import Net
from train import process_image

path_x = 'ClassData.npy' 
model_path = 'Final_Model.pth'
certainty_threshold = 20


def test(path_x, model_path):
    model = Net()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    tensor_x = process_image(path_x)
    output = model(tensor_x)
    pred_certainty, pred = torch.max(output, 1)

    predictions = pred.numpy()
    for i in range(0, len(predictions)):
        if pred_certainty[i] < certainty_threshold:
            predictions[i] = -1
    np.save('predicted.npy', predictions)
    return predictions

predicted = test(path_x, model_path)