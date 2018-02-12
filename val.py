from __future__ import print_function
import os
import argparse
from dataset import CarvanaDataset
from net import CarvanaFvbNet
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='Carvana Front vs Back - PyTorch')
parser.add_argument('--dataroot', type=str,
                    help='location of dataset')
parser.add_argument('--mode', type=str, default='full',
                    help='full|incorrect|vis')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--which-epoch', type=int, default=1, metavar='N',
                    help='initialize the model with parameters from epoch N')
parser.add_argument('--epochs', type=int, default=10, metavar='N', 
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--vis-aimg', type=int, default=0,
                    help='index of the image to visualize activations for')
parser.add_argument('--vis-alayer', type=int, default=0,
                    help='index of the layer to visualize (0-23)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

val_ds = CarvanaDataset()
val_ds.initialize(args, phase='val')
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
val_loader = DataLoader(val_ds, batch_size=args.batch_size, drop_last=True, **kwargs)

model = CarvanaFvbNet()
print(model)
print('\nloading model params')
model.load_state_dict(torch.load('./checkpoints/latest_{}.pth'.format(args.which_epoch)))
if args.cuda:
    model.cuda()
print('\nload complete!')

def val():
    model.eval()
    print('\nbegin val')
    val_loss = 0
    correct = 0
    for data, target, dsidx in val_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.squeeze(1)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        val_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    val_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

def collect_incorrect():
    import visdom
    vis = visdom.Visdom()
    """
    batch size should be 1 for this proc to work correctly
    """
    model.eval()
    print('\nbegin logging incorrect predictions')
    val_loss = 0
    correct = 0
    incorrect_idx = []
    for data, target, dsidx in val_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.squeeze(1)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        val_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        is_pred_correct = pred.eq(target.data.view_as(pred)).cpu()
        correct += is_pred_correct.sum()
        is_pred_correct_raw = is_pred_correct.numpy().tolist()[0][0]
        if is_pred_correct_raw == 0:
            img_in = data.squeeze(0)
            if args.cuda:
                img_in = img_in.cpu()
            img_in = img_in.data
            mis_idx = dsidx.numpy().tolist()[0][0]
            iter_pred = pred.cpu().numpy().tolist()[0][0]
            iter_target = target.data.cpu().numpy().tolist()[0]
            img_path = val_ds._dat[mis_idx][1]

            caption = 'pred: {}. target: {}. path: {}.'.format(
                    val_ds._idx2label[iter_pred], val_ds._idx2label[iter_target], img_path)
            vis.image(img_in, opts={ 
                'caption': caption 
            })

    val_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

def vis_activations(image_idx, layer_idx):
    import visdom
    vis = visdom.Visdom()
    model.eval()
    transform_list = [transforms.ToTensor()]
    transform = transforms.Compose(transform_list)
    lbl, path = val_ds._dat[image_idx]
    x_img = Image.open(path).convert('RGB')
    x = transform(x_img)
    vis.image(x)
    x = x.unsqueeze(0).cuda()
    x = Variable(x, volatile=True)
    output = model.forward_features(x, layer_idx).squeeze(0)
    output = output.data.cpu()
    for ch in range(output.size(0)):
        res = output[ch].squeeze(0)
        vis.image(res)




    return output

if args.mode == 'full':
    val()
elif args.mode == 'incorrect':
    collect_incorrect()
elif args.mode == 'vis':
    vis_activations(args.vis_aimg, args.vis_alayer)
