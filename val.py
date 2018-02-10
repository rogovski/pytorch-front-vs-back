from __future__ import print_function
import argparse
from dataset import CarvanaDataset
from net import CarvanaFvbNet
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Carvana Front vs Back - PyTorch')
parser.add_argument('--dataroot', type=str,
                    help='location of dataset')
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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

val_ds = CarvanaDataset()
val_ds.initialize(args, phase='test')
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
val_loader = DataLoader(val_ds, batch_size=args.batch_size, drop_last=True, **kwargs)

model = CarvanaFvbNet()
model.load_state_dict(torch.load('./checkpoints/latest_{}.pth'.format(args.which_epoch)))
if args.cuda:
    model.cuda()

def val():
    model.eval()
    print(model)
    val_loss = 0
    correct = 0
    for data, target in val_loader:
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

val()
