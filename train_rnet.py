# coding: utf-8
import sys
from data.DataSouce import DataSource
from data.augmentation import *
from network import *
from util.Logger import Logger
import os
import random
import time
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn

if not os.path.exists("./log/"):
    os.mkdir("./log/")
log = Logger("./log/{}_{}.log".format(__file__.split('/')[-1],
                                      time.strftime("%Y%m%d-%H%M%S"), time.localtime), level='debug').logger

train_batch = 400
display = 100

base_lr = 0.001
momentum = 0.9
weight_decay = 0.0005

lr_steps = [80000, 140000, 200000, 250000]
lr_decay = 0.1
max_iter = 300000
ctx = mx.gpu()

save_dir = "./models"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

root_dir = r"dataset/"
INPUT_IMAGE_SIZE = 24

MEANS = [127.5, 127.5, 127.5]
train_anno_path = []
val_anno_path = []

train_anno_path += [os.path.join(root_dir, "train_faces_r/pos/image_pos")]
train_anno_path += [os.path.join(root_dir, "train_faces_r/pos/label_pos")]

train_anno_path += [os.path.join(root_dir, "train_faces_r/part/image_part")]
train_anno_path += [os.path.join(root_dir, "train_faces_r/part/label_part")]

train_anno_path += [os.path.join(root_dir, "train_faces_r/neg/image_neg")]
train_anno_path += [os.path.join(root_dir, "train_faces_r/neg/label_neg")]
start_epoch = 0

net = RNet1()
net.initialize( ctx=ctx)
net.hybridize()
trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': base_lr, 'wd': weight_decay, 'momentum': momentum})

lossFun = LossFun()
eval = Evaluate()
train_dataset = DataSource(train_anno_path, transform=Compose([
        RandomMirror(0.5), SubtractFloatMeans(
            MEANS), ToPercentCoords(), PermuteCHW()
    ]), ratio=2, image_shape=(24,24,3))

save = './models/rnet1_'

import time
for k in range(start_epoch, max_iter + 1):
    while lr_steps and k >= lr_steps[0]:
        new_lr = trainer.learning_rate * lr_decay
        lr_steps.pop(0)
        trainer.set_learning_rate(new_lr)

    images, targets = train_dataset.getbatch(train_batch)
    images = images.as_in_context(ctx)
    targets = targets.as_in_context(ctx)
    with autograd.record():
        cls, box = net(images)

        cls = cls.reshape(cls.shape[0], cls.shape[1])
        box = box.reshape(box.shape[0], box.shape[1])
        cls_loss = lossFun.AddClsLoss(cls, targets)
        box_loss = lossFun.AddRegLoss(box, targets)
        loss = cls_loss + box_loss
        
    loss.backward()
    trainer.step(1)
    
    if k % 100 == 0:
        cls_auc = eval.AddClsAccuracy(cls, targets)
        reg_auc = eval.AddBoxMap(box, targets, 24, 24)
        log.info("iter: {}, cls_loss: {:.4f}, box_loss: {:.4f}, toal loss: {:.4f}, lr: {}, cls_auc: {:.4f}, reg_auc: {:.4f} ".format(
            k, cls_loss.asscalar(), box_loss.asscalar(), loss.asscalar(), trainer.learning_rate, 
            cls_auc.asscalar(), reg_auc ) )

    if k % 5000 == 0:
        net.save_parameters(save + str(k))

