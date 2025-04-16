---
title: "【论文阅读】多模态分割网络MMNet（CBAM的应用）"
author: Xiaofei Su
tags: 论文笔记
date: 2025-04-16
---
在介绍[CBAM（Convolutional Block Attention Module）](https://xiaosuqaq.github.io/2025/04/12/SENet%E5%92%8CCBAM.html)之后，我们来看一下这个模块在多模态任务中的一个应用：《[MM-UNet: A multimodality brain tumor segmentation network in MRI images》](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2022.950706/full)（2区），面向脑部肿瘤分割的多模态网络——MMNet。

本文重点讨论**CBAM模块**如何添加到网络中，对其作用进行简要分析。论文的主要任务是给定4种模态的MRI图像，设计了MMNet（多编码器、单解码器架构），最终预测得到4类别的分割结果。

值得注意的是，和之前博客中的讨论一致，CBAM模块在网络中起到**通道注意**和**空间注意**的作用（也就是让网络学习关注什么、关注哪里），而**没有多模态数据融合的作用**（论文中的多模态融合使用简单的“特征在通道维度进行concat”的操作实现，并没有使用fancy的多模态融合技术），笔者认为论文的名字应该更加强调通道注意和空间注意，而非多模态。（？

##### MMNet模型结构

简单介绍一下MMNet模型：

- 多编码器、单解码器：多编码器分别提取每个模态的特征，特征融合之后输入解码器，得到分割结果；
- CBAM的应用：论文中将该模块叫做“Hybrid Attention block, HAB”，位于**跳跃连接处**；
- ASPP模块：论文中叫“Dilated Convolution Block, DCB”，位于编码层之间，捕获多尺度特征。

模型的结构图见下：

<div align="center">
  <img src="/assets/images/MMNet/MMNet.png" alt="MMNet" width="80%">
</div>

除此之外，这篇工作中也对比了先通道后空间的串行结构、先空间后通道的串行结构、通道空间注意的并行结构的效果，验证了“先通道后空间的注意力串行结构”可以获得最佳效果（与CBAM论文中一致）。

##### CBAM模块

CBAM模块放在残差连接处（即图中的HAB），依次为通道注意力、空间注意力。

通道注意力部分使用全局范围池化Range（最大池化减去最小池化得到的平均值）、最大池化Max、平均池化Arverage得到不同通道的权重，最后对3个池化的结果进行元素级加法（为之后与原数据相乘时的“广播机制”匹配），整体的过程与CBAM的通道注意力计算一致。

<div align="center">
  <img src="/assets/images/MMNet/channel.png" alt="通道注意力" width="50%">
</div>

空间注意力计算也使用了3种池化操作，且最后使用卷积操作融合池化操作的结果，见下图。

<div align="center">
  <img src="/assets/images/MMNet/spatial.png" alt="空间注意力" width="50%">
</div>

##### 小结

这篇工作是CBAM模块在UNet网络中的一个扩展，将CBAM模块放在跳跃连接处，在和解码器的上采样结果融合之前，对每个模态分别进行了通道注意力和空间注意力的计算，改善了模型效果。但**CBAM并不适用于多模态信息的融合**，要实现多模态信息融合应该使用非局部神经网络块等模块，未来笔者会继续学习多模态融合的方法。
