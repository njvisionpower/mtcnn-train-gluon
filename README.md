# mtcnn-train-gluon
Implement MTCNN train pipeline with MXNet gluon
## Introduction
This repo mainly implement MTCNN([Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf)) with mxnet gluon framework. I train a faster network instead of original version(only replace pooling with conv + stride), just design your own network and you may get faster and more accurate face detection model.
<p align="center"> 
<img src="https://github.com/njvisionpower/mtcnn-train-gluon/blob/master/image/14_result.jpg" width =90% height = 90%>
</p>  
<p align="center"> 
  
## How to run
```
python MTCNN.py
```
## How to train
The WIDER-FACE dataset is needed, download and unzip to datset directory such as:
```
---dataset   
  ---WIDER_FACE
    ---wider_face_split    
    ---WIDER_train  
    ---WIDER_val   
```
First train PNet, then RNet and finally ONet. The dataset process part reference at [repo](https://github.com/beichen2012/mtcnn-pytorch). Steps list:  
**1. Generate PNet data**
```
python generate_data\Generate_PNet_data.py
```
**2. Train PNet**
```
python train_pnet.py
```
**3. Generate RNet data with trained PNet**
```
python generate_data\Generate_PNet_data.py
```
**4. Train RNet**
```
python train_rnet.py
```
**5. Generate ONet data with trained PNet and RNet**
```
python generate_data\Generate_RNet_data.py
```
**6. Train ONet**
```
python train_onet.py
```
  
## Demo
<p align="center"> 
<img src="https://github.com/njvisionpower/mtcnn-train-gluon/blob/master/image/7_result.jpg" width = 70% height = 70%>
</p> 
<p align="center"> 
<img src="https://github.com/njvisionpower/mtcnn-train-gluon/blob/master/image/9_result.jpg" width = 70% height = 70%>
</p> 
<p align="center"> 
<img src="https://github.com/njvisionpower/mtcnn-train-gluon/blob/master/image/11_result.jpg" width = 70% height = 70%>
</p> 
<p align="center"> 
<img src="https://github.com/njvisionpower/mtcnn-train-gluon/blob/master/image/13_result.jpg" width = 70% height = 70%>
</p> 
<p align="center"> 
<img src="https://github.com/njvisionpower/mtcnn-train-gluon/blob/master/image/20_result.jpg" width = 70% height = 70%>
</p> 

## Reference
[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf)  
[mtcnn-pytorch](https://github.com/beichen2012/mtcnn-pytorch)
