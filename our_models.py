import torch
import torch.nn as nn


def model_building_rgb(n_output, backbone, device, bb_name='resnet50'):
    model = backbone
    if bb_name == 'inception_v3':
        model.Conv2d_1a_3x3.conv = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, bias=False)
        model._transform_input = lambda x: x
    else:
        # modify for single channel input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_output)
    model.to(device)
    model.eval()
    return model


def model_building_flow(n_output, backbone, device, n_input=2):
    model = backbone
    # modify for single channel input
    model.conv1 = nn.Conv2d(n_input, 64, kernel_size=7,
                            stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_output)
    model.to(device)
    model.eval()
    return model
