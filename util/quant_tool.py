import torch
from dataclasses import dataclass
from typing import Optional, List
import torch.nn as nn
import torch.nn.functional as F

# 定义一个QuantizationParams类用于保存量化过程中的参数
@dataclass
class QuantizationParams:
    scale: torch.Tensor
    zero_point: torch.Tensor
    q_min: int
    q_max: int


# 对称量化参数计算,接收一个张量，返回量化参数对象，可选通道进行量化
def get_symmetric_qparams(x: torch.Tensor,
                          per_channel: bool = False,
                          channel_dim: int = 0,
                          dtype = torch.int8,
                          eps: float = 1e-8):

    # 获取最大和最小值
    q_max = torch.iinfo(dtype).max
    q_min = torch.iinfo(dtype).min

    if per_channel:
        # 在channel_dim维度上做
        max_val = x.abs().amax(dim=tuple(d for d in range(x.dim()) if d != channel_dim), keepdim=True)
    else:
        max_val = x.abs().max()
    
    # 避免除0
    scale = max_val / max(q_max,1)
    scale = torch.clip(scale,eps)

    # 对称量化，zero_point恒为0
    zero_point = torch.zeros_like(scale)
    return QuantizationParams(scale=scale, zero_point=zero_point, q_min=q_min, q_max=q_max)


# 非对称量化参数计算,接收一个张量，返回量化参数对象，可选通道进行量化
def get_asymmetric_qparams(x: torch.Tensor,
                            per_channel: bool = False,
                            channel_dim: int = 0,
                            dtype = torch.int8,
                            eps: float = 1e-8) -> QuantizationParams:

    # 获取最大和最小值
    q_max = torch.iinfo(dtype).max
    q_min = torch.iinfo(dtype).min

    if per_channel:
        reduce_dims = tuple(d for d in range(x.dim()) if d != channel_dim)
        x_min = x.amin(dim=reduce_dims, keepdim=True)
        x_max = x.amax(dim=reduce_dims, keepdim=True)
    else:
        x_min = x.min()
        x_max = x.max()
    
    # 避免0除的情况
    scale = (x_max-x_min) / max(q_max - q_min,1)  
    scale = torch.clip(scale,eps) # 当x_max == x_min的时候会出现0

    zore_point = q_min - torch.round(x_min/scale)  # 四舍五入
    zore_point = torch.clip(zore_point,q_min,q_max)

    return QuantizationParams(scale=scale, zero_point=zore_point,q_min=q_min,q_max=q_max)


# 对输入的x进行量化，返回量化后的tensor （int8）
def quantize_tensor(x: torch.Tensor, qparams: QuantizationParams) -> torch.Tensor:
    """
    通用的量化函数：
        q = clip(round(x / scale + zero_point), q_min, q_max)
    支持 per-tensor 和 per-channel（通过广播）。
    """
    scale = qparams.scale
    zero_point = qparams.zero_point

    # 确保能广播
    # 如果是标量，则直接用；如果是 per-channel，应该已经带 keepdim=True
    q = x / scale + zero_point
    q = torch.round(q)
    q = torch.clip(q, qparams.q_min, qparams.q_max)
    q = q.to(torch.int8)
    return q


# 对量化后的q进行反量化输出x_hat
def dequantize_tensor(q: torch.Tensor, qparams: QuantizationParams) -> torch.Tensor:
    """
    通用反量化：
        x_hat = (q - zero_point) * scale
    """
    q = q.to(torch.float32)
    scale = qparams.scale
    zero_point = qparams.zero_point
    x_hat = (q - zero_point) * scale
    return x_hat


# 量化+反量化函数入口
def quantize_dequant(
    x: torch.Tensor,
    per_channel: bool = False,
    channel_dim: int = 0,
    dtype=torch.int8,
    sym:bool = True,
):
    """
    一步完成：对称量化 + 反量化
    返回:
        x_hat: 反量化后的近似 x
        qparams: 量化参数，可重复使用
    """
    if sym:
        get_qparams = get_symmetric_qparams
    else:
        get_qparams = get_asymmetric_qparams
    
    # 获取量化参数
    qparams = get_qparams(
        x,
        dtype=dtype,
        per_channel=per_channel,
        channel_dim=channel_dim,
    )

    # 量化
    q = quantize_tensor(x, qparams)
    # 反量化
    x_hat = dequantize_tensor(q, qparams)
    return q, x_hat, qparams




# 定义权重量化模块（per-channel 对称 int8 量化）
class QuantLinear(nn.Module):
    """
    简化版的权重量化线性层：
    - 只量化 weight（symmetric per-channel int8）
    - bias 保持 FP32
    - 前向时：先临时反量化，再用 F.linear
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,dtype=torch.float32):
        super().__init__()


        # # qweight: int8，形状 [out_features, in_features]
        self.register_buffer(
            "qweight",
            torch.empty(out_features, in_features, dtype=torch.int8),
        )
        # scale: per-output-channel，形状 [out_features, 1]
        self.register_buffer(
            "scale",
            torch.ones(out_features, 1, dtype=dtype),
        )
        # 对称量化 zero_point 固定 0，这里留个占位方便扩展
        self.register_buffer(
            "zero_point",
            torch.zeros(out_features, 1, dtype=dtype),
        )
        
        if bias:
            self.register_buffer("bias", 
                                 torch.randn((1, out_features), 
                                             dtype=dtype))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, per_channel: bool=False, is_symmetric: bool=True,dtype=torch.float32,channel_dim=0) -> "QuantLinear":
        """
        给定一个 nn.Linear，构造对应的 QuantLinear 并完成权重量化。
        """
        qlinear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            dtype=dtype
        )

        with torch.no_grad():
            # 获取线性层的权重
            weight = linear.weight
            if is_symmetric:
                # 对线性层进行对称量化
                qparams = get_symmetric_qparams(weight,per_channel,channel_dim)
                qweight = quantize_tensor(weight,qparams)
            else:
                # 对线性层进行非对称量化
                qparams = get_asymmetric_qparams(weight,per_channel,channel_dim)
                qweight = quantize_tensor(weight,qparams)

            # 储存缩放信息
            qlinear.qweight = qweight
            qlinear.scale = qparams.scale
            qlinear.zero_point = qparams.zero_point

            # 对偏置项不做处理
            if linear.bias is not None:
                qlinear.bias = linear.bias
        return qlinear

    # 前向传播的时候需要进行反量化
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        
        # 反量化得到近似权重：w_hat = (q - z) * scale
        # qweight: [out, in], scale: [out, 1]
        w_hat = self.qweight.to(x.dtype)
        output = F.linear(x, w_hat) * self.scale
        if self.bias is not None:
            output = output + self.bias
        return output
    

# 对权重进行量化
def quantize_model_weights(
    model: nn.Module,
    per_channel:bool=False,
    is_symmetric:bool=True,
    channel_dim:int=0,
    modules_to_exclude: Optional[List[str]] = None,  # 可选参数，排除不需要量化的层
) -> nn.Module:
    """
    递归遍历模型，遇到 nn.Linear 就替换成 QuantLinear（权重量化）。
    可以通过 modules_to_exclude 按模块名排除不想量化的层。
    """
    if modules_to_exclude is None:
        modules_to_exclude = []

    for name, child in list(model.named_children()):
        full_name = name

        if isinstance(child, nn.Linear) and full_name not in modules_to_exclude:
            setattr(model, name, QuantLinear.from_linear(child,per_channel=per_channel,is_symmetric=is_symmetric,dtype=child.weight.dtype,channel_dim=channel_dim))
        else:
            quantize_model_weights(child,per_channel,is_symmetric,modules_to_exclude=modules_to_exclude)
    return model


import matplotlib.pyplot as plt
# 绘图工具
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(model, pil_img, results):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    scores, labels, boxes = results["scores"], results["labels"], results["boxes"]
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()