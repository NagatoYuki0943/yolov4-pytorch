https://blog.csdn.net/weixin_44791964/article/details/123480181?spm=1001.2014.3001.5501

# 什么是Focal Loss

Focal Loss是一种Loss计算方案。其具有两个重要的特点。

1. **控制正负样本的权重**
2. *控制容易分类和难分类样本的权重**

正负样本的概念如下：

**目标检测本质上是进行密集采样，在一张图像生成成千上万的先验框（或者特征点），将真实框与部分先验框匹配，匹配上的先验框就是正样本，没有匹配上的就是负样本**。

容易分类和难分类样本的概念如下：

**假设存在一个二分类问题，样本1和样本2均为类别1。网络的预测结果中，样本1属于类别1的概率=0.9，样本2属于类别1的概率=0.6，前者预测的比较准确，是容易分类的样本；后者预测的不够准确，是难分类的样本。**

如何实现权重控制呢，请往下看：

## 一、控制正负样本的权重

如下是常用的交叉熵loss，以二分类为例：
$$
\mathrm{CE}(p, y)
=
\left\{
\begin{array}{ll}
-\log (p) & \text { if } y=1 \\
-\log (1-p) & \text { otherwise }
\end{array}\right.
$$
我们可以利用如下Pt简化交叉熵loss。
$$
p_{\mathrm{t}}
=
\left\{\begin{array}{ll}
p & \text { if } y=1 \\
1-p & \text { otherwise }
\end{array}\right.
$$
此时：
$$
\mathrm{CE}(p, y)=\mathrm{CE}\left(p_{\mathrm{t}}\right)=-\log \left(p_{\mathrm{t}}\right)
$$
**想要降低负样本的影响，可以在常规的损失函数前增加一个系数αt。与Pt类似：**

**当label=1的时候，αt=α；**

**当label=otherwise的时候，αt=1 - α。**
$$
\alpha_{t}
=
\left\{\begin{array}{cc}
\alpha & \text { if } y=1 \\
1-\alpha & \text { otherwise }
\end{array}\right.
$$
**a的范围是0到1。此时我们便可以通过设置α实现控制正负样本对loss的贡献**。
$$
\mathrm{CE}\left(p_{\mathrm{t}}\right)=-\alpha_{\mathrm{t}} \log \left(p_{\mathrm{t}}\right)
$$
分解开就是：
$$
CE(p, y, \alpha)
=
\left\{\begin{array}{cc}
-\log (p) * \alpha & \text { if } y=1 \\
-\log (1-p) *(1-\alpha) & \text { if } y=0
\end{array}\right.
$$

## 二、控制容易分类和难分类样本的权重

**样本属于某个类，且预测结果中该类的概率越大，其越容易分类** ，在二分类问题中，正样本的标签为1，负样本的标签为0，p代表样本为1类的概率。

**对于正样本而言，1-p的值越大，样本越难分类。
对于负样本而言，p的值越大，样本越难分类。**

Pt的定义如下(同上)：
$$
p_{\mathrm{t}}
=
\left\{\begin{array}{ll}
p & \text { if } y=1 \\
1-p & \text { otherwise }
\end{array}\right.
$$
**所以利用1-Pt就可以计算出每个样本属于容易分类或者难分类。**

具体实现方式如下。
$$
\operatorname{FL}\left(p_{\mathrm{t}}\right)=-\left(1-p_{\mathrm{t}}\right)^{\gamma} \log \left(p_{\mathrm{t}}\right)
$$
其中：
$$
\left(1-\mathrm{p}_{\mathrm{t}}\right)^{\gamma}
$$
就是每个样本的容易区分程度，$γ$ 称为调制系数

1. **当pt趋于0的时候，调制系数趋于1，对于总的loss的贡献很大。当pt趋于1的时候，调制系数趋于0，也就是对于总的loss的贡献很小。**
2. **当γ=0的时候，focal loss就是传统的交叉熵损失，可以通过调整γ实现调制系数的改变。**

## 三、两种权重控制方法合并

通过如下公式就可以实现**控制正负样本的权重**和**控制容易分类和难分类样本的权重**。
$$
\mathrm{FL}\left(p_{\mathrm{t}}\right)
=
-\alpha_{\mathrm{t}}\left(1-p_{\mathrm{t}}\right)^{\gamma} \log \left(p_{\mathrm{t}}\right)
$$

分解开就是：
$$
\mathrm{FL}\left(p_{\mathrm{t}}\right)
=
\left\{\begin{array}{cc}
-\log (p) * \alpha * (1-p)^{\gamma} &  \text  { if } y=1 \\
-\log (1-p) *(1-\alpha) * p^{\gamma} & \text { if } y=0
\end{array}\right.
$$


# 实现方式

本文以Pytorch版本的YoloV4为例，给大家进行解析，YoloV4的坐标如下：
https://github.com/bubbliiiing/yolov4-pytorch

首先定位YoloV4中，**正负样本区分的损失部分**，YoloV4的损失由三部分组成，分别为：
loss_loc（回归损失）
loss_conf（目标置信度损失）
loss_cls（种类损失）

**正负样本区分的损失部分**是confidence_loss（目标置信度损失），因此我们在这一部分添加Focal Loss。

首先定位公式中的概率p。prediction代表每个特征点的预测结果，取出其中属于置信度的部分，取sigmoid，就是概率p

```python
conf = torch.sigmoid(prediction[..., 4])
```

首先进行正负样本的平衡，设立参数alpha。

`where, 正样本: alpha, 否样本: (1-alpha)`

```python
torch.where(obj_mask, torch.ones_like(conf) * self.alpha, torch.ones_like(conf) * (1 - self.alpha))
```

然后进行难易分类样本的平衡，设立参数gamma。

`where, 正样本: (1-p)^gamma, 负样本: p^gamma`

```python
torch.where(obj_mask, torch.ones_like(conf) - conf, conf) ** self.gamma
```

乘上原来的交叉熵损失即可。

`交叉熵 * pos_neg_ratio * hard_easy_ratio * focal_loss_ratio`

```python
ratio       = torch.where(obj_mask, torch.ones_like(conf) * self.alpha, torch.ones_like(conf) * (1 - self.alpha)) * torch.where(obj_mask, torch.ones_like(conf) - conf, conf) ** self.gamma
loss_conf   = torch.mean((self.BCELoss(conf, obj_mask.type_as(conf)) * ratio)[noobj_mask.bool() | obj_mask])
```

