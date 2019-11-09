from mxnet import nd
from mxnet.gluon import loss as gloss, nn
from util.utility import IOU
import numpy as np

class PNet(nn.Block):

    def __init__(self, test=False, **kwargs):
        super(PNet, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        self.net.add(
            nn.Conv2D(channels=10, kernel_size=3),
            nn.PReLU(),
            nn.MaxPool2D(2, strides=2),
            nn.Conv2D(channels=16, kernel_size=3),
            nn.PReLU(),
            nn.Conv2D(channels=32, kernel_size=3),
            nn.PReLU()
        )
      
        self.test = test
        self.cls = nn.Conv2D(channels=2, kernel_size=1)
        self.box = nn.Conv2D(channels=4, kernel_size=1)
        
    def forward(self, x):
        
        features = self.net(x)
        cls = self.cls(features)
        if self.test:
            cls = nd.softmax(cls, axis=1)
        box = self.box(features)

        return cls, box

class PNet1(nn.Block):

    def __init__(self, test=False, **kwargs):
        super(PNet1, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        self.net.add(
            nn.Conv2D(channels=10, kernel_size=3, strides=2),
            nn.PReLU(),
            nn.Conv2D(channels=16, kernel_size=3),
            nn.PReLU(),
            nn.Conv2D(channels=32, kernel_size=3),
            nn.PReLU()
        )
      
        self.test = test
        self.cls = nn.Conv2D(channels=2, kernel_size=1)
        self.box = nn.Conv2D(channels=4, kernel_size=1)
    

    def forward(self, x):
        
        features = self.net(x)
        cls = self.cls(features)
        if self.test:
            cls = nd.softmax(cls, axis=1)
        box = self.box(features)

        return cls, box

class RNet(nn.Block):

    def __init__(self, test=False, **kwargs):
        super(RNet, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        self.net.add(
            nn.Conv2D(channels=28, kernel_size=(3,3)),
            nn.PReLU(),
            nn.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            
            nn.Conv2D(channels=48, kernel_size=(3,3)),
            nn.PReLU(),
            nn.MaxPool2D(pool_size=(3,3),strides=(2,2)),

            nn.Conv2D(channels=64, kernel_size=(2,2)),
            nn.PReLU(),
            nn.Dense(128),
            nn.PReLU()
        )
      
        self.test = test
        self.cls = nn.Dense(2)
        self.box = nn.Dense(4)


    def forward(self, x):
        
        features = self.net(x)
        cls = self.cls(features)
        if self.test:
            cls = nd.softmax(cls, axis=1)
        box = self.box(features)

        return cls, box

class RNet1(nn.Block):

    def __init__(self, test=False, **kwargs):
        super(RNet1, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        self.net.add(
            nn.Conv2D(channels=28, kernel_size=3, strides=2),
            nn.PReLU(),
                      
            nn.Conv2D(channels=48, kernel_size=3, strides=2),
            nn.PReLU(),

            nn.Conv2D(channels=64, kernel_size=3),
            nn.PReLU(),
            nn.Dense(128),
            nn.PReLU()
        )
      
        self.test = test
        self.cls = nn.Dense(2)
        self.box = nn.Dense(4)   
    
    def forward(self, x):
        
        features = self.net(x)
        cls = self.cls(features)
        if self.test:
            cls = nd.softmax(cls, axis=1)
        box = self.box(features)

        return cls, box

class ONet(nn.Block):

    def __init__(self, test=False, **kwargs):
        super(ONet, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        self.net.add(
            nn.Conv2D(channels=32, kernel_size=(3,3)),
            nn.PReLU(),
            nn.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            
            nn.Conv2D(channels=64, kernel_size=(3,3)),
            nn.PReLU(),
            nn.MaxPool2D(pool_size=(3,3), strides=(2,2)),

            nn.Conv2D(channels=64, kernel_size=(3,3)),
            nn.PReLU(),
            nn.MaxPool2D(pool_size=(2,2), strides=(2,2)),

            nn.Conv2D(channels=128, kernel_size=(2,2)),
            nn.PReLU(),

            nn.Dense(256),
            nn.PReLU()
        )
      
        self.test = test
        self.cls = nn.Dense(2)
        self.box = nn.Dense(4)
    
    def forward(self, x):
        
        features = self.net(x)
        cls = self.cls(features)
        if self.test:
            cls = nd.softmax(cls, axis=1)
        box = self.box(features)

        return cls, box

class ONet1(nn.Block):

    def __init__(self, test=False, **kwargs):
        super(ONet1, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        self.net.add(
            nn.Conv2D(channels=32, kernel_size=3, strides=2),
            nn.PReLU(),
            
            nn.Conv2D(channels=64, kernel_size=3, strides=2),
            nn.PReLU(),

            nn.Conv2D(channels=64, kernel_size=3, strides=2),
            nn.PReLU(),

            nn.Conv2D(channels=128, kernel_size=3),
            nn.PReLU(),

            nn.Dense(256),
            nn.PReLU()
        )
      
        self.test = test
        self.cls = nn.Dense(2)
        self.box = nn.Dense(4)

    def forward(self, x):
        features = self.net(x)
        cls = self.cls(features)
        if self.test:
            cls = nd.softmax(cls, axis=1)
        box = self.box(features)

        return cls, box

class ONet2(nn.Block):

    def __init__(self, test=False, **kwargs):
        super(ONet2, self).__init__(**kwargs)
        self.net = nn.HybridSequential()
        self.net.add(
            nn.Conv2D(channels=32, kernel_size=3, strides=2),
            nn.PReLU(),
            
            nn.Conv2D(channels=64, kernel_size=3, strides=2),
            nn.PReLU(),

            nn.Conv2D(channels=64, kernel_size=3, strides=2),
            nn.PReLU(),

            nn.Conv2D(channels=128, kernel_size=2),
            nn.PReLU(),

            nn.Dense(256),
            nn.PReLU()
        )
      
        self.test = test
        self.cls = nn.Dense(2)
        self.box = nn.Dense(4)
    
    def forward(self, x):
        
        features = self.net(x)
        cls = self.cls(features)
        if self.test:
            cls = nd.softmax(cls, axis=1)
        box = self.box(features)

        return cls, box


class LossFun:
    def __init__(self): 
        self.classifier = gloss.SoftmaxCrossEntropyLoss()
        self.regresser = gloss.HuberLoss()
    
    def AddClsLoss(self, pred, targets, k=0.7):

        label = targets[:, -1].reshape(targets.shape[0], -1)
        tmp = label.asnumpy()
        valid_idx = np.where(tmp < 2)
  
        label_use = targets[:, -1][valid_idx[0]]
        pred_use = pred[valid_idx[0]]
 
        loss = self.classifier(pred_use, label_use)
  
        topk = int(k * loss.shape[0])
        
        return loss[loss.topk(k=topk)].mean()
    
    def AddRegLoss(self, pred, targets):
        label = targets[:, -1].reshape(targets.shape[0], -1)
        tmp = label.asnumpy()
        valid_idx = np.where(tmp > 0)

        label_use = targets[:, 0:4][valid_idx[0]]
        pred_use = pred[valid_idx[0]]

        loss = self.regresser(pred_use, label_use)

        return loss.mean()


class Evaluate():
    def AddClsAccuracy(self, pred, targets):
        label = targets[:, -1].reshape(targets.shape[0], -1)
        idx = nd.where(label<2, nd.ones_like(label), nd.zeros_like(label))
        
        label_use = label*idx
        pred_use = pred*idx

        validate = idx.sum() 

        pred_idx = nd.argmax(pred_use, axis=1)

        error = pred_idx != label_use.reshape(label_use.shape[0],)
        errors = nd.sum(error)

        return (validate - errors) / validate

    def AddBoxMap(self, pred, targets, image_width, image_height):

        label = targets.asnumpy()
        pred = pred.asnumpy()

        label_valid_idx = np.where(label[:, -1] > 0)

        label_valid = label[:, 0:4][label_valid_idx] 
        pred_valid = pred[label_valid_idx] 

        map = 0.0
        label_valid[:, 0] *= image_width
        label_valid[:, 1] *= image_height

        pred_valid[:, 0] *= image_width
        pred_valid[:, 1] *= image_height
        
        num = label_valid.shape[0]
        for i in range(num):
            b1 = pred_valid[i, :]
            b2 = label_valid[i, :]

            b1[2] = np.exp(b1[2]) * image_width
            b1[3] = np.exp(b1[3]) * image_height

            b2[2] = np.exp(b2[2]) * image_width
            b2[3] = np.exp(b2[3]) * image_height

            map += IOU(b1, b2)

        return map / num   