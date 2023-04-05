from __future__ import print_function
import os
import numpy as np
import argparse
import torch
import torch.utils.data as data_utils
from torch.autograd import Variable
from dataloader_allfeaturefound import ALLSlideBags
from model import Attention, GatedAttention
import time
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='PyTorch Transfer MIL Example')
# For dataset root
parser.add_argument('--root_test', type=str, default='./featuresfound/datasets/',
                    help='root of the test dataset')
parser.add_argument('--model_path', type=str, default='./final_model/final_model_scale1/',
                    help='root of the test dataset')
# For dataloader
parser.add_argument('--bag_length', type=int, default=200, metavar='ML',
                    help='bag length for per slide')
# For random seed
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


def drawing_ROC(fpr, tpr, roc_auc):
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


# Val part
def funing(path, j):
    instance_weight_list = []
    instance_name_list = []
    ckpt = torch.load(f=path)["model_state_dict"]
    model.load_state_dict(state_dict=ckpt)
    test_loss = 0.
    test_error = 0.
    label_bag = []
    predict_bag = []
    confidence_bag = []
    for batch_idx, (data, label, name) in enumerate(test_loader):
        bag_label = label[0]
        data = data.cuda()
        bag_label = bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        with torch.set_grad_enabled(mode=False):
            error, predicted_label, loss, attention_weights, confidence = model.calculate_classification_error_fortest(
                data, bag_label)
            test_loss += loss.item()
            test_error += error

        bag_level = (int(bag_label.cpu().data.numpy()), int(predicted_label.cpu().data.numpy()[0]))
        judge = int(bag_label.cpu().data.numpy()) == int(predicted_label.cpu().data.numpy()[0])
        instance_level = list(zip(np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))
        instance_weight_list.append(instance_level)
        instance_name_list.append(name)
        label_bag.append(int(bag_label.cpu().data.numpy()))
        predict_bag.append(int(predicted_label.cpu().data.numpy()[0]))

        confidence_bag.append(confidence.cpu().data.numpy())
        j.append(judge)

        print('\nTrue Bag Label, Predicted Bag Label: {}\n'
              'True Instance Labels, Attention Weights: {}\n'
              'Confidence:{}'.format(bag_level, instance_level, confidence.cpu().data.numpy()[0]))

        # print(judge)

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))

    return test_error, confidence_bag, label_bag, predict_bag, j, instance_weight_list, instance_name_list


if __name__ == "__main__":
    judgement = []
    instance_weight_list_path = './finalfeature_found/instance_weight_listN-1.csv'

    name = []
    confidence = []
    # print('******************************Start Testing*******************************')
    for idx, filename in enumerate(os.listdir(args.model_path)):
        search_model_path = args.model_path + '/' + filename
        name.append(filename)

        for seed in tqdm(range(1, 2)):
            j = []
            torch.manual_seed(seed)
            if args.cuda:
                torch.cuda.manual_seed(seed)

            loader_kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

            if args.model == 'attention':
                model = Attention(interlayer_classes=args.model_classes,
                                  num_layers=args.model_layers,
                                  pretrain=args.model_pretrain)
            elif args.model == 'gated_attention':
                model = GatedAttention()

            model = model.cuda()
            # print('******************************  uploading test data ***********************************')
            test_loader = data_utils.DataLoader(ALLSlideBags(
                bag_length=args.bag_length,
                seed=seed,
                root=args.root_test,
                train=False),
                batch_size=1,
                shuffle=False,
                **loader_kwargs)

            since = time.time()
            test_error, confidence_bag, label_bag, predict_bag, j, instance_weight_list, instance_name_list = funing(
                path=search_model_path, j=j)

            time_elapsed = time.time() - since

            # Writing ROC curve and calculating AUC

            for i in range(len(label_bag)):
                if predict_bag[i] == 0:
                    confidence_bag[i] = 1 - confidence_bag[i]

            y_label = (label_bag)
            # y_pre = (predict_bag)
            y_con = (confidence_bag)

            fpr, tpr, thersholds = roc_curve(y_label, y_con)
            roc_auc = auc(fpr, tpr)
            # print('\nSeed: ' + str(seed) + ' Test Set, confidence: {:.4f}'.format(y_con))
            print('\nSeed: ' + str(seed) + ' Test Set, AUC: {:.4f}'.format(roc_auc))
            judgement.append(confidence_bag)

    instance_weight_listfinal = [b for a in instance_weight_list for b in a]
    instance_name_listfinal = [b for a in instance_name_list for b in a]
    # instance_weight_listfinal = np.array(instance_weight_listfinal)
    # instance_name_listfinal = np.array(instance_name_listfinal)
    feature_list = [instance_weight_listfinal, instance_name_listfinal]
    dataframe2 = pd.DataFrame({'name': instance_name_listfinal, 'weights': instance_weight_listfinal})
    dataframe2.to_csv(instance_weight_list_path, index=False, sep=',', header=False)

    print('****************************Finish Testing*******************************')
