import torch
import torch.nn as nn
import numpy as np

from utils import *
from envs import *
from models import *

def main():
    agent = TFTAgent()
    agent.train()
    agent.save()

    

if __name__ == '__main__':
    main()