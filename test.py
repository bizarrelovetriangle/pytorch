import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pathlib
import struct
from torchinfo import summary
import os
import datetime
from pathlib import Path
from common import SchedulerParams
from common import Common

if __name__ == '__main__':
    tensor = torch.tensor([[1, 2], [3, 4]])
    print(tensor)