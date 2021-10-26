"""
Train script
"""

import os
import argparse
import torch
import torch.nn as nn

from torch_lib.Dataset import Dataset
from torch_lib.Model import Model, OrientationLoss
from torchvision.models import vgg
from torch.utils import data


PARSER = argparse.ArgumentParser()

PARSER.add_argument("--dataset-path", default="Kitti/training",
                    help="Path to directory with dataset")

PARSER.add_argument("--calib-path", default="camera_cal/calib_cam_to_cam.txt",
                    help="Path file with calibrating data for camera")

PARSER.add_argument("--weights-path", default="weights/",
                    help="Path to folder, where weights will be saved. \
                        By default, this is weights/")

PARSER.add_argument("--device", default="cuda",
                    help="PyTorch device: cuda/cpu")


def main():
    """ main function """
    flags = PARSER.parse_args()

    device = flags.device

    # hyper parameters
    epochs = 100
    batch_size = 8
    alpha = 0.6
    weights = 0.4

    print("Loading all detected objects in dataset...")

    dataset = Dataset(flags.dataset_path, flags.calib_path)

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 6}

    generator = data.DataLoader(dataset, **params)

    my_vgg = vgg.vgg19_bn(pretrained=True)
    model = Model(features=my_vgg.features).to(device)
    opt_sgd = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    conf_loss_func = nn.CrossEntropyLoss().to(device)
    dim_loss_func = nn.MSELoss().to(device)
    orient_loss_func = OrientationLoss

    # load any previous weights
    model_path = flags.weights_path
    latest_model = None
    first_epoch = 0
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except ValueError:
            pass

    if latest_model is not None:
        checkpoint = torch.load(os.path.join(model_path, latest_model))
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_sgd.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print('Found previous checkpoint: %s at epoch %s' %(latest_model, first_epoch))
        print('Resuming training....')

    total_num_batches = int(len(dataset) / batch_size)

    for epoch in range(first_epoch+1, epochs+1):
        curr_batch = 0
        passes = 0
        for local_batch, local_labels in generator:

            truth_orient = local_labels['Orientation'].float().to(device)
            truth_conf = local_labels['Confidence'].long().to(device)
            truth_dim = local_labels['Dimensions'].float().to(device)

            local_batch = local_batch.float().to(device)
            [orient, conf, dim] = model(local_batch)

            orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
            dim_loss = dim_loss_func(dim, truth_dim)

            truth_conf = torch.max(truth_conf, dim=1)[1]
            conf_loss = conf_loss_func(conf, truth_conf)

            loss_theta = conf_loss + weights * orient_loss
            loss = alpha * dim_loss + loss_theta

            opt_sgd.zero_grad()
            loss.backward()
            opt_sgd.step()

            if passes % 10 == 0:
                print("--- epoch {} | batch {}/{} --- [loss: {}]"\
                    .format(epoch, curr_batch, total_num_batches, loss.item()))
                passes = 0

            passes += 1
            curr_batch += 1

        # save after every 10 epochs
        if epoch % 10 == 0:
            name = os.path.join(model_path, 'epoch_%s.pkl' % epoch)
            print("====================")
            print("Done with epoch %s!" % epoch)
            print("Saving weights as %s ..." % name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt_sgd.state_dict(),
                'loss': loss
            }, name)
            print("====================")


if __name__ == '__main__':
    main()
