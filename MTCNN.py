# coding: utf-8

import cv2
import time
import os
import numpy as np
from network import *
from pylab import plt
import mxnet as mx
from util.utility import pad_bbox, square_bbox, py_nms

SHOW_FIGURE = False

def Image2NArray(img, mean):
    src = img.astype(np.float32) - np.array([127.5,127.5,127.5], dtype=np.float32)
    src = src.swapaxes(1, 2).swapaxes(0, 1)
    
    input = nd.expand_dims( nd.array(src), axis=0 )
    return input

class MTCNN(object):

    def __init__(self, detectors=[None, None, None], min_face_size=24, scalor=0.709, threshold=[0.6, 0.7, 0.7],
                   ctx = mx.cpu()  ):
        self.pnet = detectors[0]
        self.rnet = detectors[1]
        self.onet = detectors[2]
        self.min_face_size = min_face_size
        self.scalor =  scalor
        self.threshold = threshold
        self.ctx = ctx

    def detect(self, img):
        bboxes = None

        # pnet
        if not self.pnet:
            return None
        bboxes = self.detect_pnet(img)

        if bboxes is None:
            return None

        if SHOW_FIGURE:
            plt.figure()
            tmp = img.copy()
            for i in bboxes:
                x0 = int(i[0])
                y0 = int(i[1])
                x1 = x0 + int(i[2])
                y1 = y0 + int(i[3])
                cv2.rectangle(tmp, (x0, y0), (x1, y1), (0, 0, 255), 2)
            plt.imshow(tmp[:, :, ::-1])
            plt.title("pnet result")

        # rnet
        if not self.rnet:
            return bboxes
        bboxes = bboxes[:, 0:4].astype(np.int32)
        bboxes = self.detect_ronet(img, bboxes, 24)

        if bboxes is None:
            return None

        if SHOW_FIGURE:
            plt.figure()
            tmp = img.copy()
            for i in bboxes:
                x0 = int(i[0])
                y0 = int(i[1])
                x1 = x0 + int(i[2])
                y1 = y0 + int(i[3])
                cv2.rectangle(tmp, (x0, y0), (x1, y1), (0, 0, 255), 2)
            plt.imshow(tmp[:, :, ::-1])
            plt.title("rnet result")

        if not self.onet:
            return bboxes
        bboxes = bboxes[:, 0:4].astype(np.int32)
        bboxes = self.detect_ronet(img, bboxes, 48)

        return bboxes

    def detect_pnet(self, im):
        #print('pnet.......')
        h, w, c = im.shape
        net_size = 12
        minl = np.min((w, h))
        base_scale = net_size / float(self.min_face_size)
        scales = []
        face_count = 0
        while minl > net_size:
            s = base_scale * self.scalor ** face_count
            if np.floor(minl * s) <= 12:
                break
            scales += [s]
            face_count += 1

        total_boxes = []
        for scale in scales:
            hs = np.ceil(h * scale)
            ws = np.ceil(w * scale)
            hs = int(hs)
            ws = int(ws)

            im_data = cv2.resize(im, (ws, hs))
            input = Image2NArray(im_data, [127.5,127.5,127.5] )
            input = input.as_in_context(self.ctx)
            
            output_cls, output_reg = self.pnet(input)
            
            output_cls = output_cls.asnumpy().squeeze(axis=0)
            output_reg = output_reg.asnumpy().squeeze(axis=0)

            bboxes = self.generate_bbox(output_cls, output_reg, scale, self.threshold[0])

            if len(bboxes) <= 0:
                continue
            keep = py_nms(bboxes, 0.5, 'Union')

            if len(keep) <= 0:
                continue

            bboxes = bboxes[keep]
            #
            total_boxes.extend(bboxes)

        # NMS
        if len(total_boxes) <= 0:
            return None
        total_boxes = np.array(total_boxes)
        keep = py_nms(total_boxes, 0.7, 'Union')
        if len(keep) <= 0:
            return None
        return total_boxes[keep]

    def generate_bbox(self, cls_map, reg, scale, threshold):
        """
        ----------
            cls_map: numpy array , 2*h*w
                detect score for each position
            reg: numpy array , 4*h*w
                reg bbox
            scale: float number
                scale of this image pyramid from original image
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array in original image， num*5, [x,y,w,h,score]
        """
        stride = 2
        cellsize = 12
        face_map = cls_map[1, :, :]
        t_index = np.where(face_map > threshold)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])
        # offset
        # dx, dy, dw, dh = [reg[t_index[0], t_index[1], i] for i in range(4)]
        dx, dy, dw, dh = [reg[i, t_index[0], t_index[1]] for i in range(4)]

        dx *= cellsize
        dy *= cellsize
        dw = np.exp(dw) * cellsize
        dh = np.exp(dh) * cellsize

        score = face_map[t_index[0], t_index[1]]

        Gx = np.round(stride * t_index[1] / scale)
        Gy = np.round(stride * t_index[0] / scale)
        dx = dx / scale + Gx
        dy = dy / scale + Gy
        dw = dw / scale
        dh = dh / scale

        bbox = np.vstack([dx, dy, dw, dh, score])
        bbox = bbox.T
        return bbox

    def detect_ronet(self, img, bboxes, image_size):
        H, W, C = img.shape
        IMAGE_SIZE = image_size
      
        sb = []
        for i in range(bboxes.shape[0]):
            box = bboxes[i, :]
            sq = square_bbox(box)
            sb += [sq]

        #pad
        crops = []
        origin_bbox = []
        for i in sb:
            size = i[2]
            sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1 = pad_bbox(i, W, H)
            crop = np.zeros((size, size, 3), dtype=np.uint8)
            if sx0 < 0 or sy0 < 0 or dx0 < 0 or dy0 < 0 or sx1 > W or sy1 > H or dx1 > size or dy1 > size:
                continue
            crop[dy0:dy1, dx0:dx1, :] = img[sy0:sy1, sx0:sx1, :]
            out = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
            out = out.astype(np.float32) - np.array([127.5,127.5,127.5], dtype=np.float32)
            out = out.swapaxes(1, 2).swapaxes(0, 1)
            crops += [out]
            origin_bbox += [i]

        origin_bbox = np.array(origin_bbox)
        crops = nd.array(crops)
        input =  crops.as_in_context(self.ctx)
        detector = self.rnet
        threshold = self.threshold[1]
        if image_size == 48:
            detector = self.onet
            threshold = self.threshold[2]

        out = detector(input)

        cls_map = out[0].asnumpy()
        reg = out[1].asnumpy()
        face_map = cls_map[:, 1]
        t_index = np.where(face_map > threshold)
        if t_index[0].shape[0] <= 0:
            return None

        origin_bbox = origin_bbox[t_index]
        score = face_map[t_index]
        reg_map = reg[t_index]

        dx = reg_map[:, 0]
        dy = reg_map[:, 1]
        dw = reg_map[:, 2]
        dh = reg_map[:, 3]

        dx *= IMAGE_SIZE
        dy *= IMAGE_SIZE
        dw = np.exp(dw) * IMAGE_SIZE
        dh = np.exp(dh) * IMAGE_SIZE

        # add Gx AND Gy
        G = origin_bbox
        G = G.astype(np.float32)

        dx = dx / (float(IMAGE_SIZE) / G[:, 2]) + G[:, 0]
        dy = dy / (float(IMAGE_SIZE) / G[:, 3]) + G[:, 1]
        dw = dw / (float(IMAGE_SIZE) / G[:, 2])
        dh = dh / (float(IMAGE_SIZE) / G[:, 3])

        # compose
        bbox = np.vstack([dx, dy, dw, dh, score])
        bbox = bbox.T

        # do nms
        if image_size == 24:
            keep = py_nms(bbox, 0.7, "Union")
            if len(keep) <= 0:
                return None
            return bbox[keep]

        if image_size == 48:
            keep = py_nms(bbox, 0.7, "Minimum")
            if len(keep) <= 0:
                return None
            return bbox[keep]


if __name__ == "__main__":
    
    pnet = PNet1(test=True)
    rnet = RNet1(test=True)
    onet = ONet1(test=True)
    ctx = mx.cpu()

    pnet.load_parameters('./models/pnet1_150000', ctx = ctx )
    pnet.hybridize()

    rnet.load_parameters('./models/rnet1_300000', ctx=ctx)
    rnet.hybridize()

    onet.load_parameters('./models/onet_80000',ctx=ctx)
    onet.hybridize()
    mtcnn = MTCNN(detectors=[pnet, rnet, onet], min_face_size = 24, scalor = 0.709,threshold=[0.6, 0.7, 0.7], ctx = ctx )

    img_path = "image/1.jpg"
    new_name =  img_path.split('.')[0] + '_result.jpg'
    img =cv2.imread(img_path)
    b = time.time()
    bboxes = mtcnn.detect(img)

    e = time.time()
    print("time cost: {} ms".format((e-b) * 1000.0))
    
    if True:
        if bboxes is not None:
            plt.figure()
            tmp = img.copy()
            for i in bboxes:
                x0 = int(i[0])
                y0 = int(i[1])
                x1 = x0 + int(i[2])
                y1 = y0 + int(i[3])
                cv2.rectangle(tmp, (x0, y0), (x1, y1), (0, 0, 255), 2)
                cv2.imwrite(new_name , tmp)
            plt.imshow(tmp[:, :, ::-1])
            plt.title("result")
        plt.show()
    








