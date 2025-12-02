![FirstQuantization](./img/0_fq.png)

# FirstQuantization
你的第一个量化实例（其实是我的），大模型量化入门案例，从这里开始你的量化学习吧！

喜欢的话点个赞支持一下作者吧！😍

## 课程介绍
__基于conda+jupyter，推荐使用vscode进行学习。__


主要包含以下内容：

0. [大模型量化（作者的自制视频，持续更新中……）](https://www.bilibili.com/video/BV1E6UdBVEeS)
1. [数据类型](1_data_type.ipynb)
2. [对称量化](2_symmetric_quantization.ipynb)
3. [非对称量化](3_asymmetric_quantization.ipynb)
4. [封装量化工具](4_tool_function.ipynb.ipynb)
5. [线性层量化](5_linear_quantization.ipynb)
6. [【案例1】语言模型量化](6_llm_demo.ipynb)
7. [【案例2】目标识别模型量化](7_target_demo.ipynb)、

## 环境需求
```ssh
git clone https://github.com/LeeWant/FirstQuantization.git
conda create -n fquant python=3.10
conda activate fquant
pip install -r requirement.txt
```

## 参考资料
[A Visual Guide to Quantization](https://www.maartengrootendorst.com/blog/quantization/)

[Quantization Fundamentals](https://learn.deeplearning.ai/courses/quantization-fundamentals)

[Quantization in Depth](https://learn.deeplearning.ai/courses/quantization-in-depth)

## 感谢支持~~