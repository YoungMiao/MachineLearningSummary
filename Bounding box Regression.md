###一．	问题的数学表达，回归的对象是什么？
&#8195;我们一般最开始都有一个初始的BBox，但是这只是个粗略的BBox。比如RCNN，fastrcnn用Selective Search方法生成的一系列Proposals，faster-rcnn的anchors对应的boxes。这些都是粗略的框。我们最终要得到精确的位置框，所以需要用回归的方法精修BBox。
&#8195;因此，我们的输入是初始的BBox，需要找到一个映射把初始的BBox精修到最后Ground Truth。对应的数学描述如下：
![这里写图片描述](https://img-blog.csdn.net/20180404094648496?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
### 二．	如何进行回归
![这里写图片描述](https://img-blog.csdn.net/20180404094732962?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
&#8195;下一步的问题就是设计算法得到这四个映射。
&#8195;当输入的Proposals与Ground Truth相差较小时，（RCNN中设置的是IOU>0.6），可以认为这种变化是一种线性变化，因此可以用线性回归对该问题建模，从而微调窗口。
![这里写图片描述](https://img-blog.csdn.net/2018040409490962?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![这里写图片描述](https://img-blog.csdn.net/20180404094924635?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![这里写图片描述](https://img-blog.csdn.net/20180404094941442?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
&#8195;这样以来，我们回归的结果是原始Proposals需要微调的dx,dy,dw,dh。最后再使用下面的公式也就是上面的公式（1）~（4）得到最后的回归框。
![这里写图片描述](https://img-blog.csdn.net/20180404095133273?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)![这里写图片描述](https://img-blog.csdn.net/20180404095145513?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
### 三．	rcnn中的实现
&#8195;rcnn使用Selective Search的方法从一张图像生成约2000-3000个候选区域。基本思路如下： 
- 使用一种过分割手段，将图像分割成小区域 
- 查看现有小区域，合并可能性最高的两个区域。
- 重复直到整张图像合并成一个区域位置 
- 输出所有曾经存在过的区域，所谓候选区域

&#8195;Rcnn中候选区域生成和后续步骤相对独立，也就是在rcnn网络中，提前用Selective Search的方法已经得到了上面将的候选框，即Px，Py，Pw，Ph。RCNN通过线性回归器进行每个候选框的精修工作，即回归器的输入为深度网络pool5层的4096维特征（这里使用的是alexnet），输出为xy方向的缩放和平移,即dx,dy,dh,dw。
&#8195;由于rcnn比较简单，这里不再讲述，这列出一张计算t的实例代码。
![这里写图片描述](https://img-blog.csdn.net/20180404095343288?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
下面会重点讲述faster rcnn。
### 四．	Faster rcnn中的实现
Faster –rcnn中Bbox回归的实现。
1.网络的RPN
&#8195;RPN的核心思想是使用卷积神经网络直接产生region proposal，使用的方法本质上就是滑动窗口。Region ProposalNetwork(RPN)如下图：
![这里写图片描述](https://img-blog.csdn.net/20180404095517945?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
&#8195;RPN网络结构图如上所示（ZF模型:256维），假设给定600*1000的输入图像，经过卷积操作得到最后一层的卷积feature map（大小约为40*60），最后一层卷积层共有256个feature map。
&#8195;在这个特征图上使用3*3的卷积核（滑动窗口）与特征图进行卷积，那么这个3*3的区域卷积后可以获得一个256维的特征向量。因为这个3*3的区域上，每一个特征图上得到一个1维向量，256个特性图即可得到256维特征向量。
&#8195;3*3滑窗中心点位置，对应预测输入图像3种尺度（128,256,512），3种长宽比（1:1,1:2,2:1）的regionproposal，这种映射的机制称为anchor，产生了k=9个anchor。即每个3*3区域可以产生9个region proposal。所以对于这个40*60的feature map，总共有40*60个anchor，约20000(40*60*9)个proposal，也就是预测20000个region proposal。
&#8195;下面是RPN回归的网络结构。

```
#========= RPN ============
layer {
  name: "rpn_conv/3x3"
  type: "Convolution"
  bottom: "conv5_3"
  top: "rpn/output"
  param { lr_mult: 1.0 }
  param { lr_mult: 2.0 }
  convolution_param {
    num_output: 512
    kernel_size: 3 pad: 1 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}
layer {
  name: "rpn_relu/3x3"
  type: "ReLU"
  bottom: "rpn/output"
  top: "rpn/output"
}

layer {
  name: "rpn_bbox_pred"
  type: "Convolution"
  bottom: "rpn/output"
  top: "rpn_bbox_pred"
  param { lr_mult: 1.0 }
  param { lr_mult: 2.0 }
  convolution_param {
    num_output: 36   # 4 * 9(anchors)
    kernel_size: 1 pad: 0 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}

layer {
  name: 'rpn-data'
  type: 'Python'
  bottom: 'rpn_cls_score'
  bottom: 'gt_boxes'
  bottom: 'im_info'
  bottom: 'data'
  top: 'rpn_labels'
  top: 'rpn_bbox_targets'
  top: 'rpn_bbox_inside_weights'
  top: 'rpn_bbox_outside_weights'
  python_param {
    module: 'rpn.anchor_target_layer'
    layer: 'AnchorTargetLayer'
    param_str: "'feat_stride': 16"
  }
}

layer {
  name: "rpn_loss_bbox"
  type: "SmoothL1Loss"
  bottom: "rpn_bbox_pred"
  bottom: "rpn_bbox_targets"
  bottom: 'rpn_bbox_inside_weights'
  bottom: 'rpn_bbox_outside_weights'
  top: "rpn_loss_bbox"
  loss_weight: 1
  smooth_l1_loss_param { sigma: 3.0 }
}

```
&#8195;rpn_bbox_pred层用卷积层，实现了线性回归，输出了36个通道的特征图。这里36表示每个anchors有9个boxes，每个box需要回归dx,dy,dw,dh四个参数。所以每个anchors一共对应4*9=36个值。如果输入40*60的feature map，总共有约40*60个anchors。
&#8195;rpn_loss_bbox层计算损失函数，也就是上面分析的损失函数，即
![这里写图片描述](https://img-blog.csdn.net/20180404095946251?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
&#8195;只不过这里计算的是SmoothL1Loss，不求平方，另外第二项就是这里rpn_bbox_pred的输出，即rpn_bbox_pred。第一项就是真实值，即rpn_bbox_targets。
&#8195;下面看看rpn_bbox_targets是怎么计算的。
&#8195;在AnchorTargetLayer 层中完成了rpn_bbox_targets的计算。
&#8195;AnchorTargetLayer位置在 py-faster-rcnn/lib/rpn中。
&#8195;AnchorTargetLayer类中forward函数中计算了bbox_targets。
![这里写图片描述](https://img-blog.csdn.net/20180404100115378?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
可见bbox_targets是用anchors和gt_box计算出来的。
下面分两步讲解，先讲一下_compute_targets的实现再讲anchors是怎么生成的。
（1）_compute_targets的实现。
AnchorTargetLayer类中实现了_compute_targets函数中。
![这里写图片描述](https://img-blog.csdn.net/2018040410015120?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
bbox_transform函数位于py-faster-rcnn/lib/fast-rcnn/bbox_transform.py文件中。
![这里写图片描述](https://img-blog.csdn.net/20180404100243172?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
可见，bbox_transform实现的功能，实际就是上面我们计算t的过程，即
![这里写图片描述](https://img-blog.csdn.net/20180404100307708?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
这里的Px,Py,Ph,Pw就是上面计算的anchors，Gx,Gy,Gh,Gw就是上面gt_box，即我们提前标注的边界框。  
（2）anchors怎么计算
下面是AnchorTargetLayer类中forward函数中用anchors的地方。
![这里写图片描述](https://img-blog.csdn.net/20180404100450922?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
可是anchors怎么创建的呢？AnchorTargetLayer类中setup函数实现了这个功能。![这里写图片描述](https://img-blog.csdn.net/20180404100511229?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
generate_anchors位于py-faster-rcnn/lib/rpn/generate_anchors.py中。![这里写图片描述](https://img-blog.csdn.net/20180404100539782?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
&#8195;可见，generate_anchors函数中定义的anchors的scale为2^3,2^4,2^5,即[8,16,32]，由于_ratio_enum函数会将scale乘以base_size（16），即对应的预测的尺度变为（128,256,512）。另外Ratios为[0.5,1,2]。这里，和之前讲述RPN中所说的，对应预测输入图像3种尺度（128,256,512），3种长宽比（1:1,1:2,2:1）的region proposal是一致的。
&#8195;base_anchor对应一个box的x,y,w,h。即一个region proposal的x,y,w,h。
ratio_anchors变量是对应3个不同比例的boxes的x,y,w,h；
&#8195;_scale_enum（ratio_anchors[i,:], scales）中scales是个二维矩阵，矩阵为3行，每一行对应一个box的x,y,w,h，ratio_anchors[i,:]也是二维矩阵，矩阵为3行，每一行对应一个box的x,y,w,h。_scale_enum()返回二维矩阵，矩阵为9行，每一行对应一个box的x,y,w,h。即9个不同尺寸，不同rate的box的x,y,w,h。
&#8195;anchors为特征图所有位置对应的box的x,y,w,h。每个位置对应9个不同尺寸，不同rate的box的x,y,w,h。
_ratio_enum函数定义如下：
![这里写图片描述](https://img-blog.csdn.net/20180404101040580?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
_scale_enum函数定义如下：
![这里写图片描述](https://img-blog.csdn.net/20180404101130427?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
### 五、SSD中的回归
Default Box 的生成
&#8195;SSD利用不同层的 feature map 来模仿学习不同尺度下物体的检测。假定使用 m 个不同层的feature map 来做预测，最底层的 feature map 的 scale 值S_min 为0.2，最高层的S_max为 0.95，其他层通过下面公式计算得到
![这里写图片描述](https://img-blog.csdn.net/20180404101539590?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![这里写图片描述](https://img-blog.csdn.net/20180404101626881?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
### 六．	mtcnn中的回归
&#8195;这里只讲述mtcnn从回归的数据中如果恢复出最终的bbox。因为回归出来的4个数值是dx,dy,dh,dw。如何把这些转换成x,y,w,h。即使用下面的公式得到最后的回归框。
![这里写图片描述](https://img-blog.csdn.net/20180404101759996?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)![这里写图片描述](https://img-blog.csdn.net/20180404101811684?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
下面以P网络的输出举例说明。
P网络的输出是out。
out第一行所有待选框回归出来的dx1,dy1, dx2,dy2, 
out第二行是所有待选框的概率值。
![这里写图片描述](https://img-blog.csdn.net/2018040410183718?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
1.	使用generateBoundingBox函数，将预测的概率的大于阈值的候选框的回归值挑选出来。
2.	用下式计算出来Bbox左上角的x,y,和右下角的x,y。这里的total_boxes[:,5]~ total_boxes[:,8]对应着回归出来的dx,dy。total_boxes[:,0]~ total_boxes[:,4]对应着原始候选框的左上角和右下角坐标。即下面的Px，Py。
![这里写图片描述](https://img-blog.csdn.net/20180404101919898?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
下面是generateBoundingBox函数. 
stride=2，表示候选框右下角的坐标就是左上角坐标的的2倍。
cellsize=12，表示选框右下角的坐标还需要平移12个单位。
P网络在训练时生成候选框的规则也应该是上面的方式。预测时，候选框的生成方式和训练是生成的候选框保持一致。
 由于mtcnn在候选框生成时没有像faster-rcnn，一个anchors会生成不同尺度，不同rate的的候选框，所以mtcnn会一开始先生成图像的金字塔，生成不同尺度的图片，这样以来，最后的候选框就相当于有了不同尺度的候选框。
q1是预测的概率的大于阈值的候选框的左上角坐标（除以scale，转换为在原图中的坐标）。
q2是预测的概率的大于阈值的候选框的右下角坐标（除以scale，转换为在原图中的坐标）。
reg是对应的候选框回归的dx1,dy1, dx2,dy2。
![这里写图片描述](https://img-blog.csdn.net/20180404102003472?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JhaWR1XzI2Nzg4OTUx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)