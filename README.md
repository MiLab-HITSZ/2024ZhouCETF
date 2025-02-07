# 0. 模型和数据
需要注意的是，用到了一个较大的pretrain开源模型和开源数据，所以本仓库难以上传展示。具体见Evolutionary Trigger Detection and Lightweight Model Repair Based Backdoor Defense实验细节。https://arxiv.org/abs/2407.05396

# 1. CETF触发器搜索

以下两个文件均用于触发器搜索，分别针对不同的模型。运行时可以修改`batch size`参数和第142行的程序终止条件，以决定演示时一次性运行多少个图像。

- `cam_mask.py`: 针对MobileNet模型
- `tcam_mask.py`: 针对VGG16模型

# 2. unlearning模型修复

以下两个文件均用于模型修复，分别针对不同的模型。模型地址无需修改。

- `unlearning.py`
- `unlearning_vgg16.py`

需要修改的输入参数（以`unlearning.py`为例）：

- 第40行：可选择unlearning方式，包括朴素去学习、bn-unlearning和bn-cleaning。
- 第42行和第43行：参数`poi_trainflag`和`poi_valflag`代表是否给微调数据和验证集数据投毒。

`poi_trainflag`：微调数据通常需要投毒，因为需要贴上触发器才能进行微调去学习。给微调数据投毒时，标签仍然是真实标签，因此可以将`poi_trainflag`赋值为1。

`poi_valflag`：

- 如果`poi_valflag=0`，则意味着使用干净数据进行评估。程序结束后，将输出unlearning前后的干净数据分类准确度。
- 如果`poi_valflag=1`，则意味着使用投毒数据进行评估（注意此时标签都更改为目标标签0）。输出的结果将是unlearning前后的攻击成功率。




