import torch
from dataclasses import dataclass

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