from __future__ import print_function
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from dataloader_load_by_epoch import ALLSlideBags
from model import Attention, GatedAttention
from tensorboardX import SummaryWriter
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='PyTorch Transfer MIL Example')
# For dataset root
parser.add_argument('--root_train', type=str, default='./train_datasets/',
                    help='root of the train dataset')
parser.add_argument('--root_val', type=str, default='./valid_datasets/',
                    help='root of the val dataset')
# Training settings
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=5e-5, metavar='R',
                    help='weight decay')
# For dataloader
parser.add_argument('--bag_length', type=int, default=200, metavar='ML',
                    help='bag length for per slide')
# For random seed
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')

# For model settings
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')
parser.add_argument('--model_classes', type=int, default=1000, help='feature extractor1 output classes')
parser.add_argument('--model_layers', type=int, default=50, help='feature extractor1 resnet layers')
parser.add_argument('--model_pretrain', type=bool, default=True, help='feature extractor1 resnet pretrain')
parser.add_argument('--model_savepath', type=str, default='./NPclass_final/checkpoints/', help='model savepath')
parser.add_argument('--summary_name', type=str, default='NPclassfinal1_', help='Name of the summary')

args = parser.parse_args()
sum_writer = SummaryWriter(comment=args.summary_name)

args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

loader_kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

if args.model == 'attention':
    model = Attention(interlayer_classes=args.model_classes,
                      num_layers=args.model_layers,
                      pretrain=args.model_pretrain)
elif args.model == 'gated_attention':
    model = GatedAttention()

model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

print('******************************** uploading val data *************************************')
val_loader = data_utils.DataLoader(ALLSlideBags(
    bag_length=args.bag_length,
    seed=args.seed,
    root=args.root_val,
    train=False),
    batch_size=1,
    shuffle=False,
    **loader_kwargs)


# Training part
def train(epoch):
    model.train(mode=True)

    print('******************************  uploading train data ***********************************')
    train_loader = data_utils.DataLoader(ALLSlideBags(
        bag_length=args.bag_length,
        seed=args.seed,
        root=args.root_train,
        train=True),
        batch_size=1,
        shuffle=True,
        **loader_kwargs)

    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]

        data = data.cuda()
        bag_label = bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(mode=True):
            error, pred_train, loss, attention_weights = model.calculate_classification_error(data, bag_label)
            train_loss += loss.item()
            train_error += error

            # backward pass
            loss.backward()
            # step
            optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    attentionmap = np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()
    sum_writer.add_scalar('train_loss', train_loss, epoch)
    sum_writer.add_scalar('train_error', train_error, epoch)

    print('Attention Weights/epoch: {}'.format(attentionmap))
    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))

    save_path = args.model_savepath + f"resnet{args.model_layers}_e{epoch}_error{train_error:.5f}.pt"
    torch.save(obj={
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch + 1
    },
        f=str(save_path))


# Val part
def val():
    # model.eval()
    model.train(mode=False)
    val_loss = 0.
    val_error = 0.
    for batch_idx, (data, label) in enumerate(val_loader):
        bag_label = label[0]
        data = data.cuda()
        bag_label = bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        with torch.set_grad_enabled(mode=False):
            error, predicted_label, loss, attention_weights = model.calculate_classification_error(data, bag_label)
            val_loss += loss.item()
            val_error += error

        if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
            bag_level = (int(bag_label.cpu().data.numpy()), int(predicted_label.cpu().data.numpy()[0]))
            instance_level = list(zip(np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

            print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                  'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

    val_error /= len(val_loader)
    val_loss /= len(val_loader)
    sum_writer.add_scalar('val_loss', val_loss, epoch)
    sum_writer.add_scalar('val_error', val_error, epoch)

    print('\nVal Set, Loss: {:.4f}, Val error: {:.4f}'.format(val_loss, val_error))


if __name__ == "__main__":
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        since = time.time()
        train(epoch)
        if epoch % 5 == 0:
            print('Start Val')
            val()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    print('Finish Training')
