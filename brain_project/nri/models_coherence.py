"""Models used for the coherence data. This is similar to Melanies work.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()


    def forward(self, X):
        """
        Args:
         - X : tensor [batch_size, num_freq, num_nodes, num_nodes]
        Returns:
         - pred : tensor [batch_size, num_classes]
        """



