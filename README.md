# Dense Correspondence

## Milestone

* Milestone 1.24
    1. 研究网络的训练, 提升加权和feature的效果.
* Milestone 1.18
    1. 提升C4 feature的效果.
    2. 用C4 feature做cost volume, 得到correspondence map, 预测selection map, 和C2, C3, C4做加权和, 输出新feature.
* Milestone 1.17
    1. 输入ground-truth correspondence map, 预测selection map. 让geometric distortion region的selection channel偏重于high level, 让appearance-preserved region的selection channel偏重于low level.
* Milestone 1.9
    1. 用contrastive loss学两对图片的feature.
    2. 用学好的feature做cost volume, 看score map中max score对应的channel的分布情况.
* Milestone 1.3
    1. 检查projection的正确性, lam参数对结果的影响.
    2. small baseline和large baseline用不一样的feature, 查看correspondence的效果.
* Milestone 12.31
    1. 完善整个训练流程, 包括模型参数的保存和加载, tensorboard对训练过程的记录, 模型参数的regularization, 模型learning rate的schedule.
    2. 将训练数据集中图片之间的baseline变小.
    3. 对比有projection和没有projection的训练情况.

## Accomplishment

* 1.22
    * 实现分阶段训练.
* 1.21
    * 输出1/4 resolution的feature map.
    * 将triplet的数据放在TripletLoss类中处理.
    * 增加对C4 feature的supervision.
* 1.19
    * 将原来的mini dataset分成mini small-baseline和mini large-baseline. 固定baseline, 因为baseline对训练的影响很大, 在测试的时候影响也很大.
    * 增加AdaptiveOneShot类, 将correspondence map和selection map的预测做end-to-end training.
* 1.18
    * 对contrastive loss中distance loss加了margin, 让两个对应的feature不用完全一样, 只要距离在margin以内.
    * 加了EPE和PCE metric, 用于测从两个feature maps得到的dense correspondences.
    * 加了一个ContrastiveDB类, 把collect_contrastive()的大部分代码移到这个类中, 不然之前这个函数太长了.
    * 获得foreground和background的像素对, 计算前景和背景的similarity loss, 对foreground和background的feature做分离.
    * 在Loss类中将loss装入一个dict返回.
    * 增加TripletLoss类.
* 1.17
    * 增加了mini dataset中左右图pose的多样性.
    * 减小了计算ground-truth correspondence时depth的阈值.
* 1.16
    * Backbone的网络就专门提取feature, 别的网络可以用resnet, 但不可以用Backbone的网络, 因为Backbone的网络就专门提feature.
    * 加了cfg.MODEL.BACKBONE.FEATURE_TOWER这个参数, 让backbone网络可以返回feature tower.
    * 加了build_selection(), 用于预测selection map, 给不同level的feature预测一个权重, 做加权和.
    * 在matcher中加了GroundTruthOneShot类, 输入ground-truth correspondence map, 预测selection map. 从backbone网络中得到feature tower, 用selection map做加权和.
    * 加了获得ground-truth correspondence map的代码, 如果是用GroundTurthOneShot类就返回. 加了validate_grad()来看ground-truth correspondence map.
    * 在evluation时加了render_selection(), 可视化每个像素偏重的feature level.
* 1.15
    * 将3D Cost Volume中H*W的index方式, 从idx=row+H*col改成idx=W*row+col, 也就是将[W, H]改成[H, W].
    * 加了一个validate_correspondence(), 基于feature1和feature2, 用flann和cost_volume都算一般correspondences, 查看两者算出来的correspondences的区别.
    * 加了build_corr_map(), 基于Cost Volume, 算soft argmax, 得到每个像素对应的像素, 可微分.
* 1.14
    * 用FLANN看dense correspondences, 查看feature的效果.
* 1.13
    * 在ResNet中指定从某一层开始feature map的分辨率不变, 具体做法是, 将strides变为1, 并将conv layer变成dilated conv layer, dilation大小逐层乘2, 保持网络可视域不变.
    * 在cfg中指定特定层数开始保持feature map的分辨率不变.
* 1.12
    * 只用Cosine Similarity计算Cost Volume, 增加4D Cost Volume.
    * 将evaluation封装成FlowEvaluator和FeatureEvalautor.
    * 从3D Cost Volume和4D Cost Volume中计算correspondence.
    * 之前的correspondences和non-correspondences采样有问题, 因为没有考虑到一对多和多对一的关系. 在large-baseline的情况下, distorted region pair经常出现, 就会发生一个像素对应多个像素和多个像素对应一个像素的情况. 之前correspondences只考虑左图对应到右图, 没考虑到右图对应到左图, 所以少了一个像素对应多个像素的情况. 这里又计算了右图对应到左图, 将一个像素对应多个像素的correspondences加到了correspondences. 这里correspondences只是少了数据, 而non-correspondences的correspondences的采样就直接导致了网络训练的错误. 之前没考虑到多个像素会对应一个像素, 直接将correspondences做了shuffle, 当做non-correspondences. 但是因为多个像素会对应一个像素. 所以可能造成shuffle以后, 把原来的correspondence当成了non-correspondence, 导致了错误. 这里用另一种方式生成non-correspondences, 先初始化随机的non-correspondences, 然后去除与correspondences冲突的non-correspondences.
* 1.11
    * Set baseline by setting affinity matrix.
    * Add random-baseline and mini_dataset.
    * Add two evaluation type: FLOW and FEATURE.
    * evaluation的时候从dataset得到evaluation要用的数据.
    * 可视化feature.
* 1.10
    * sepatate compute_correspondence() from compute_flow(), 将scale_flow()放到compute_correspondence()里面
    * 之前只有一种target, 是{"flow": ..., "matchability": ...}. 现在有两种target, FLOW类型的target, 是{"flow": ..., "matchability": ...}, 还有FEATURE类型的target, 是{"corr": ..., "non_corr": ...}.
    * 加了collect_contrastive(), 返回FEATURE类型的target, corr代表image A和image B之间的correspondences, non_corr代表image A和image B之间的non_correspondences. 因为面积比带来的概率, correspondences邻域作为non_correspondences的概率比较低, 网络倾向于让correspondences邻域区域以外的features之间距离更大.
    * 加了一种_MATCHING_META_ARCHITECTURES类型Contrastive, 目前由backbone和loss组成.
    * 加了ContrastiveLoss. 对于minimize distance, 这里收集了一个batch内的所有correspondences的distance, 再计算平均distance, 将gradient平均到每一个correspondence上. 对于minimize similairty, 这里收集了一个batch内所有non_correspondences的similarity, 再选出hard negatives, 再计算平均similarity.
* 1.5
    * 之前的training data和testing data混在一起, 就是都用同样的instance. 这次我把训练和测试的instance分开, 训练用前4/5的instance, 测试用后1/5的instance.
    * 将之前的ShapeNet_car数据集分成了small baseline和large baseline的数据集, 可以通过yaml的配置文件直接调用. 每次调用ShapeNetDataset都会重新生成ann_file.
* 1.4 enable training based on the features extracted from `C1, C2, C3, C4, C5`
* 1.3
    * 对比有projection和没有projection的训练情况.
    * **根据mask-rcnn benchmark的思路, 将training和evaluation分开. dataset只返回images和target, target是一个dict, 包含flow和matchability. model在training的时候返回losses和results, 在testing的时候返回results. result只包含flow和matchability.**
    * 实现EPE和PCE的metric, 实现dense correspondence的evaluation.
* 1.2
    * 将训练数据集中图片之间的baseline变小.
    * 将checkpoint改成一个list, 存储已经存放的model parameters. 之前的checkpoint数量不限, 现在的版本用cfg.SOLVER.NUM_CHECKPOINT指定.
    * 用cfg.SOLVER.RESUME指定whether to load pretrained model. 如果重新训练, 会将之前相应的tensorboard数据删除, 以免tensorboardX报错. **如果想尝试新配置, 请新建一个yaml. 如果是在原有的yaml上修改, 建议重新开始训练, 也就是将cfg.SOLVER.RESUME设为False.**
    * 将`losses.backward()`改成`losses_reduced.backward()`, 否则多GPU会报错.
* 1.1 实现tensorboard, 用cfg指定要记录的scalars和images, 用cfg.TENSORBOARD.LOG_DIR指定log存放的目录.
* 12.31 实现end-to-end training.
* 12.23 实现`net/data/build.py`, 可以用`make_data_loader()`和`cfg`直接得到`data_loader`.
* 12.22 实现`net/data/datasets/shapenet.py`里面比较基本的ShapeNetDataset.


## Perform training on ShapeNet dataset

You need to first download the ShapeNet dataset `10.76.6.174:/mnt/data/ShapeNet`, and then symlink the ShapeNet dataset to `datasets/` as follows

```
mkdir -p datasets
ln -s /path-to-ShapeNet datasets/ShapeNet
```

### Training

Run the following without modifications

```
python -m tools.train_net --config-file ./configs/flow/small-baseline_R_18_C4_no_projection.yaml
```


Note：
1. 路径：运行脚本时都应该在root下，因为寻找路径都默认从root开始。
2. 所有和配置(config)相关的，包括dataset的路径等，都统一在`./net/config`管理。`./configs`里面的`.yaml`用来覆盖默认配置。


## Contribution


### Motivation

This repo saves your time for preparation to try out a new idea in pytorch.
Main reference: maskrcnn-benchmark, facebook.

### Features

- Hold several configs that overwrite the default one
- Only need to implement dataset/loader, net-architecture, test-module
- Use Adam solver as default
- Support TensorboardX and average recorders

### Dependencies

- PyTorch 1.0
- yacs
- TensorboardX




