from __future__ import print_function
import os
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from sklearn.metrics import precision_recall_curve
from torch.autograd import Variable
from dataloader_load_by_epoch import ALLSlideBags
from model import Attention, GatedAttention
import time
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    # plt.plot(recs, precs)
    # plt.title('PR curve')
    # plt.show()

    f1s = 2 * precs * recs / (precs + recs)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    bestf1 = f1s[np.argmax(f1s)]
    return best_thr, bestf1


parser = argparse.ArgumentParser(description='PyTorch Transfer MIL Example')
parser.add_argument('--root_test', type=str, default='./test_datasets/',
                    help='root of the test dataset')
parser.add_argument('--model_path', type=str, default='./checkpoints_aug/resnet50_e363_error0.25984.pt',
                    help='root of the test dataset')
# For dataloader
parser.add_argument('--bag_length', type=int, default=200, metavar='ML',
                    help='bag length for per slide')
# For random seed
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')

# For model settings
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')
parser.add_argument('--model_classes', type=int, default=1000, help='feature extractor1 output classes')
parser.add_argument('--model_layers', type=int, default=50, help='feature extractor1 resnet layers')
parser.add_argument('--model_pretrain', type=bool, default=True, help='feature extractor1 resnet pretrain')

args = parser.parse_args()

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
print('******************************  uploading test data ***********************************')
test_loader = data_utils.DataLoader(ALLSlideBags(
    bag_length=args.bag_length,
    seed=args.seed,
    root=args.root_test,
    train=False),
    batch_size=1,
    shuffle=False,
    **loader_kwargs)


# Val part
def funing(path):
    ckpt = torch.load(f=path)["model_state_dict"]
    model.load_state_dict(state_dict=ckpt)
    test_loss = 0.
    test_error = 0.
    label_bag = []
    predict_bag = []
    train_pred_original_bag = []
    confidence_bag = []
    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0]
        data = data.cuda()
        bag_label = bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        with torch.set_grad_enabled(mode=False):
            error, predicted_label, loss, attention_weights, confidence, train_pred_original = model.calculate_classification_error_fortest_thresholds(
                data, bag_label)
            test_loss += loss.item()
            test_error += error

        bag_level = (int(bag_label.cpu().data.numpy()), int(predicted_label.cpu().data.numpy()[0]))
        instance_level = list(zip(np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))
        label_bag.append(int(bag_label.cpu().data.numpy()))
        predict_bag.append(int(predicted_label.cpu().data.numpy()[0]))
        train_pred_original_bag.append(int(train_pred_original.cpu().data.numpy()[0]))
        # confidence_bag.append(np.round(confidence.cpu().data.numpy(),1))
        confidence_bag.append(confidence.cpu().data.numpy())

        print('\nTrue Bag Label, Predicted Bag Label: {}\n'
              'True Instance Labels, Attention Weights: {}\n'
              'Confidence:{}'.format(bag_level, instance_level, confidence.cpu().data.numpy()[0]))

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))

    return test_error, confidence_bag, label_bag, predict_bag, train_pred_original_bag


if __name__ == "__main__":
    print('******************************Start Testing*******************************')
    since = time.time()
    test_error, confidence_bag, label_bag, predict_bag, train_pred_original_bag = funing(path=args.model_path)

    # Writing ROC curve and calculating AUC
    for i in range(len(label_bag)):
        if train_pred_original_bag[i] == 0:
            confidence_bag[i] = 1 - confidence_bag[i]

    y_label = (label_bag)
    # y_pre = (predict_bag)
    y_con = (confidence_bag)

    fpr, tpr, thersholds = roc_curve(y_label, y_con)
    # for i, value in enumerate(thersholds):
    #     print("%f %f %f" % (fpr[i], tpr[i], value))
    roc_auc = auc(fpr, tpr)
    print('\nTest Set, AUC: {:.4f}'.format(roc_auc))

    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    plt.plot([0, 1], [0, 1], 'k--', color='orange')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # calculate precision, recall, f1, acc, auc and print
    prec, rec, f1, _ = precision_recall_fscore_support(y_label, predict_bag, average="binary")
    acc = accuracy_score(y_label, predict_bag)
    print("ACC: %.4f Prec: %.4f Rec: %.4f F1: %.4f" % (acc, prec, rec, f1))

    bestthr, bestf1 = return_best_thr(y_label, y_con)
    print('best f1: ', bestf1)
    print('best threshold: ', bestthr)
    # print('best threshold: ', return_best_thr(y_label, y_con))

    C = confusion_matrix(y_label, predict_bag)

    time_elapsed = time.time() - since

    classes = ['N', 'P']
    confusion_matrix = np.array([(C[0, 0], C[0, 1]), (C[1, 0], C[1, 1])], dtype=np.float64)

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-45)
    plt.yticks(tick_marks, classes)

    thresh = confusion_matrix.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    iters = np.reshape([[[i, j] for j in range(2)] for i in range(2)], (confusion_matrix.size, 2))
    for i, j in iters:
        plt.text(j, i, format(confusion_matrix[i, j]))

    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()

    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('****************************Finish Testing*******************************')
