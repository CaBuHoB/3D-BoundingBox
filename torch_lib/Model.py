"""
model class
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def orientation_loss(orient_batch, orient_gt_batch, conf_gt_batch):
    """ function orientation loss """
    batch_size = orient_batch.size()[0]
    indexes = torch.max(conf_gt_batch, dim=1)[1]

    # extract just the important bin
    orient_gt_batch = orient_gt_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orient_gt_batch[:, 1], orient_gt_batch[:, 0])
    estimated_theta_diff = torch.atan2(orient_batch[:, 1], orient_batch[:, 0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean()

class Model(nn.Module):
    """ class model """
    def __init__(self, features=None, bins=2, weights=0.4):
        """ function initialization """
        super(Model, self).__init__()
        self.bins = bins
        self.weights = weights
        self.features = features
        self.orientation = nn.Sequential(\
                    nn.Linear(512 * 7 * 7, 256),\
                    nn.ReLU(True),\
                    nn.Dropout(),\
                    nn.Linear(256, 256),\
                    nn.ReLU(True),\
                    nn.Dropout(),\
                    nn.Linear(256, bins*2) \
                        )
        self.confidence = nn.Sequential(\
                    nn.Linear(512 * 7 * 7, 256),\
                    nn.ReLU(True),\
                    nn.Dropout(),\
                    nn.Linear(256, 256),\
                    nn.ReLU(True),\
                    nn.Dropout(),\
                    nn.Linear(256, bins),\
                )
        self.dimension = nn.Sequential(\
                    nn.Linear(512 * 7 * 7, 512),\
                    nn.ReLU(True),\
                    nn.Dropout(),\
                    nn.Linear(512, 512),\
                    nn.ReLU(True),\
                    nn.Dropout(),\
                    nn.Linear(512, 3)\
                )

    def forward(self, val_x):
        """ function forwarding """
        val_x = self.features(val_x) # 512 x 7 x 7
        val_x = val_x.view(-1, 512 * 7 * 7)
        orientation = self.orientation(val_x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)
        confidence = self.confidence(val_x)
        dimension = self.dimension(val_x)
        return orientation, confidence, dimension
