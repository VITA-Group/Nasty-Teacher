# train a nasty teacher with an adversarial network

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam


from tqdm import tqdm
import argparse
import os
import logging
import numpy as np

from utils.utils import RunningAverage, set_logger, Params
from model import *
from data_loader import fetch_dataloader

# ************************** random seed **************************
seed = 0

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='experiments/CIFAR10/adversarial_teacher/resnet18_self', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=[0], type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

device_ids = args.gpu_id
torch.cuda.set_device(device_ids[0])


# ************************** training function **************************
def train_epoch_kd_adv(model, model_ad, optim, data_loader, epoch, params):
    model.train()
    model_ad.eval()
    tch_loss_avg = RunningAverage()
    ad_loss_avg = RunningAverage()
    loss_avg = RunningAverage()

    with tqdm(total=len(data_loader)) as t:  # Use tqdm for progress bar
        for i, (train_batch, labels_batch) in enumerate(data_loader):
            if params.cuda:
                train_batch = train_batch.cuda()  # (B,3,32,32)
                labels_batch = labels_batch.cuda()  # (B,)

            # compute (teacher) model output and loss
            output_tch = model(train_batch)  # logit without SoftMax

            # teacher loss: CE(output_tch, label)
            tch_loss = nn.CrossEntropyLoss()(output_tch, labels_batch)

            # ############ adversarial loss ####################################
            # computer adversarial model output
            with torch.no_grad():
                output_stu = model_ad(train_batch)  # logit without SoftMax
            output_stu = output_stu.detach()

            # adversarial loss: KLdiv(output_stu, output_tch)
            T = params.temperature
            adv_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output_stu / T, dim=1),
                                      F.softmax(output_tch / T, dim=1)) * (T * T)   # wish to max this item

            # total loss
            loss = tch_loss - params.weight * adv_loss + 100.0  # make the loss positive by adding a constant

            # ############################################################

            optim.zero_grad()
            loss.backward()
            optim.step()

            # update the average loss
            loss_avg.update(loss.item())
            tch_loss_avg.update(tch_loss.item())
            ad_loss_avg.update(adv_loss.item())

            # tqdm setting
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    return loss_avg(), tch_loss_avg(), ad_loss_avg()


def evaluate(model, loss_fn, data_loader, params):
    model.eval()
    # summary for current eval loop
    summ = []

    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch, labels_batch in data_loader:
            if params.cuda:
                data_batch = data_batch.cuda()          # (B,3,32,32)
                labels_batch = labels_batch.cuda()      # (B,)

            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()
            # calculate accuracy
            output_batch = np.argmax(output_batch, axis=1)
            acc = 100.0 * np.sum(output_batch == labels_batch) / float(labels_batch.shape[0])

            summary_batch = {'acc': acc, 'loss': loss.item()}
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    return metrics_mean


def train_and_eval_kd_adv(model, model_ad, optim, train_loader, dev_loader, params):
    best_val_acc = -1
    best_epo = -1
    lr = params.learning_rate

    for epoch in range(params.num_epochs):
        lr = adjust_learning_rate(optim, epoch, lr, params)
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        logging.info('Learning Rate {}'.format(lr))

        # ********************* one full pass over the training set *********************
        train_loss, train_tloss, train_aloss = train_epoch_kd_adv(model, model_ad, optim,
                                                                  train_loader, epoch, params)
        logging.info("- Train loss : {:05.3f}".format(train_loss))
        logging.info("- Train teacher loss : {:05.3f}".format(train_tloss))
        logging.info("- Train adversarial loss : {:05.3f}".format(train_aloss))

        # ********************* Evaluate for one epoch on validation set *********************
        val_metrics = evaluate(model, nn.CrossEntropyLoss(), dev_loader, params)  # {'acc':acc, 'loss':loss}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
        logging.info("- Eval metrics : " + metrics_string)

        # save model
        save_name = os.path.join(args.save_path, 'last_model.tar')
        torch.save({
            'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optim.state_dict()},
            save_name)

        # ********************* get the best validation accuracy *********************
        val_acc = val_metrics['acc']
        if val_acc >= best_val_acc:
            best_epo = epoch + 1
            best_val_acc = val_acc
            logging.info('- New best model ')
            # save best model
            save_name = os.path.join(args.save_path, 'best_model.tar')
            torch.save({
                'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optim.state_dict()},
                save_name)

        logging.info('- So far best epoch: {}, best acc: {:05.3f}'.format(best_epo, best_val_acc))


def adjust_learning_rate(opt, epoch, lr, params):
    if epoch in params.schedule:
        lr = lr * params.gamma
        for param_group in opt.param_groups:
            param_group['lr'] = lr
    return lr


if __name__ == "__main__":
    # ************************** set log **************************
    set_logger(os.path.join(args.save_path, 'training.log'))

    # #################### Load the parameters from json file #####################################
    json_path = os.path.join(args.save_path, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    params.cuda = torch.cuda.is_available() # use GPU if available

    for k, v in params.__dict__.items():
        logging.info('{}:{}'.format(k, v))

    # ########################################## Dataset ##########################################
    trainloader = fetch_dataloader('train', params)
    devloader = fetch_dataloader('dev', params)

    # ############################################ Model ############################################
    if params.dataset == 'cifar10':
        num_class = 10
    elif params.dataset == 'cifar100':
        num_class = 100
    elif params.dataset == 'tiny_imagenet':
        num_class = 200
    else:
        num_class = 10

    logging.info('Number of class: ' + str(num_class))

    logging.info('Create Model --- ' + params.model_name)

    # ResNet 18 / 34 / 50 ****************************************
    if params.model_name == 'resnet18':
        model = ResNet18(num_class=num_class)
    elif params.model_name == 'resnet34':
        model = ResNet34(num_class=num_class)
    elif params.model_name == 'resnet50':
        model = ResNet50(num_class=num_class)

    # PreResNet(ResNet for CIFAR-10)  20/32/56/110 ***************
    elif params.model_name.startswith('preresnet20'):
        model = PreResNet(depth=20, num_classes=num_class)
    elif params.model_name.startswith('preresnet32'):
        model = PreResNet(depth=32, num_classes=num_class)
    elif params.model_name.startswith('preresnet44'):
        model = PreResNet(depth=44, num_classes=num_class)
    elif params.model_name.startswith('preresnet56'):
        model = PreResNet(depth=56, num_classes=num_class)
    elif params.model_name.startswith('preresnet110'):
        model = PreResNet(depth=110, num_classes=num_class)


    # DenseNet *********************************************
    elif params.model_name == 'densenet121':
        model = densenet121(num_class=num_class)
    elif params.model_name == 'densenet161':
        model = densenet161(num_class=num_class)
    elif params.model_name == 'densenet169':
        model = densenet169(num_class=num_class)

    # ResNeXt *********************************************
    elif params.model_name == 'resnext29':
        model = CifarResNeXt(cardinality=8, depth=29, num_classes=num_class)

    elif params.model_name == 'mobilenetv2':
        model = MobileNetV2(class_num=num_class)

    elif params.model_name == 'shufflenetv2':
        model = shufflenetv2(class_num=num_class)

    # Basic neural network ********************************
    elif params.model_name == 'net':
        model = Net(num_class, params)

    elif params.model_name == 'mlp':
        model = MLP(num_class=num_class)

    else:
        model = None
        print('Not support for model ' + str(params.model_name))
        exit()

    # Adversarial model *************************************************************
    logging.info('Create Adversarial Model --- ' + params.adversarial_model)

    # ResNet 18 / 34 / 50 ****************************************
    if params.adversarial_model == 'resnet18':
        adversarial_model = ResNet18(num_class=num_class)
    elif params.adversarial_model == 'resnet34':
        adversarial_model = ResNet34(num_class=num_class)
    elif params.adversarial_model == 'resnet50':
        adversarial_model = ResNet50(num_class=num_class)

    # PreResNet(ResNet for CIFAR-10)  20/32/56/110 ***************
    elif params.adversarial_model.startswith('preresnet20'):
        adversarial_model = PreResNet(depth=20)
    elif params.adversarial_model.startswith('preresnet32'):
        adversarial_model = PreResNet(depth=32)
    elif params.adversarial_model.startswith('preresnet56'):
        adversarial_model = PreResNet(depth=56)
    elif params.adversarial_model.startswith('preresnet110'):
        adversarial_model = PreResNet(depth=110)

    # DenseNet *********************************************
    elif params.adversarial_model == 'densenet121':
        adversarial_model = densenet121(num_class=num_class)
    elif params.adversarial_model == 'densenet161':
        adversarial_model = densenet161(num_class=num_class)
    elif params.adversarial_model == 'densenet169':
        adversarial_model = densenet169(num_class=num_class)

    # ResNeXt *********************************************
    elif params.adversarial_model == 'resnext29':
        adversarial_model = CifarResNeXt(cardinality=8, depth=29, num_classes=num_class)

    elif params.adversarial_model == 'mobilenetv2':
        adversarial_model = MobileNetV2(class_num=num_class)

    elif params.adversarial_model == 'shufflenetv2':
        adversarial_model = shufflenetv2(class_num=num_class)

    # Basic neural network ********************************
    elif params.adversarial_model == 'net':
        adversarial_model = Net(num_class, params)

    elif params.adversarial_model == 'mlp':
        adversarial_model = MLP(num_class=num_class)

    else:
        adversarial_model = None
        print('Not support for model ' + str(params.adversarial_model))
        exit()

    if params.cuda:
        model = model.cuda()
        adversarial_model = adversarial_model.cuda()

    if len(args.gpu_id) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        adversarial_model = nn.DataParallel(adversarial_model, device_ids=device_ids)

    # checkpoint ********************************
    if args.resume:
        logging.info('- Load checkpoint from {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        logging.info('- Train from scratch ')

    # load trained Adversarial model ****************************
    logging.info('- Load Trained adversarial model from {}'.format(params.adversarial_resume))
    checkpoint = torch.load(params.adversarial_resume)
    adversarial_model.load_state_dict(checkpoint['state_dict'])

    # ############################### Optimizer ###############################
    if params.model_name == 'net' or params.model_name == 'mlp':
        optimizer = Adam(model.parameters(), lr=params.learning_rate)
        logging.info('Optimizer: Adam')
    else:
        optimizer = SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
        logging.info('Optimizer: SGD')

    # ************************** train and evaluate **************************
    train_and_eval_kd_adv(model, adversarial_model, optimizer, trainloader, devloader, params)

