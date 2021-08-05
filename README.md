# Simpler is Better: Few-shot Semantic Segmentation with Classifier Weight Transformer. ICCV2021.

## Introduction

We proposed a novel model training paradigm for few-shot semantic segmentation. Instead of meta-learning the whole, complex segmentation model, we focus on the simplest
classifier part to make new-class adaptation more tractable. Also, a novel meta-learning algorithm that leverages a Classifier Weight Transformer (CWT) for adapting dynamically the classifier weights to every query sample is introduced to eliminate the impact of intra-class discripency. 

## Architecture
<a href="url"><img src="https://github.com/zhiheLu/CWT-for-FSS/blob/main/doc/framework.jpg" align="center" height="350" width="900" ></a>

## Environment
Other configurations can also work, but the results may be slightly different.
- torch==1.6.0
- numpy==1.19.1
- cv2==4.4.0
- pyyaml==5.3.1
