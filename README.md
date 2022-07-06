# Switchable Online Knowledge Distillation [ECCV 2022]
This repository is the official code for the paper "Switchable Online Knowledge Distillation" by *Biao Qian, Yang Wang (corresponding author: yangwang@hfut.edu.cn), Hongzhi Yin, Richang Hong, Meng Wang* (ECCV 2022, Tel-Aviv, Israel).

## Introduction
To break down the bottlenecks over the gap between teacher and student --- e.g., *Why and when does a large gap harm the performance, especially for student? How to quantify the gap between teacher and student?* we deeply analyze the adversarial impact
of large gap on student, and propose Switchable Online Knowledge Distillation (**SwitOKD**); see Figure 1. Instead of focusing on the accuracy gap at test phase by the existing arts, the core idea of SwitOKD is to **adaptively** calibrate *the gap at training phase*, namely **distillation gap**, via a switching strategy between two training modes — **expert mode** (pause the teacher while keep the student learning) and **learning mode** (restart the teacher and reciprocally train from scratch). To endow SwitOKD with the capacity to yield an appropriate distillation gap, we further devise an **adaptive switching threshold** (*i.e.*, ![](http://latex.codecogs.com/svg.latex?\delta=||p_s^{\tau}-y||_1-e^{-\frac{||p_t^{\tau}-y||_1}{||p_s^{\tau}-y||_1+||p_t^{\tau}-y||_1}}||p_t^{\tau}-y||_1)), which provides a formal criterion as to when to switch to learning mode or expert mode, and thus improves the student’s performance. Meanwhile, the teacher keeps
basically on a par with other online arts.



![overview](https://github.com/hfutqian/SwitOKD/blob/main/images/overview.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Figure 1 Illustration of the proposed SwitOKD framework.


Based on the above, to endow SwitOKD with the extendibility to **multi-network setting** with large distillation gap, we build two types of fundamental *basis topologies* below: multiple teachers *vs* one student and one teacher *vs* multiple students. In the implementation, we take 3 networks as an example and denote the basis topologies as **2T1S** and **1T2S**, respectively; see Figure 2.

![multi-net](https://github.com/hfutqian/SwitOKD/blob/main/images/multi-net.png)
Figure 2 The multi-network framework for training 3 networks simultaneously, including two fundamental basis topologies: **2T1S** (left) and **1T2S** (right).


## Requirements
* python3.6
* pytorch1.3.1
* cuda10.0

## Usages




## Results



![results_table4](https://github.com/hfutqian/SwitOKD/blob/main/images/results_table4.png)










