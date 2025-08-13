import cv2
import numpy as np
import torch

try:
    import lpips
    from skimage.metrics import structural_similarity as cal_ssim
except ImportError:
    lpips = None
    cal_ssim = None

# LPIPS实例缓存，避免多次初始化开销
_lpips_calculator = None

def rescale(x):
    """Rescale input to [-1, 1] range."""
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1

def _threshold(x, y, t):
    """Apply threshold and compute hits / false alarms etc."""
    t = np.greater_equal(x, t).astype(np.float32)
    p = np.greater_equal(y, t).astype(np.float32)
    is_nan = np.logical_or(np.isnan(x), np.isnan(y))
    t = np.where(is_nan, np.zeros_like(t, dtype=np.float32), t)
    p = np.where(is_nan, np.zeros_like(p, dtype=np.float32), p)
    return t, p

def MAE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean(np.abs(pred - true), axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean(np.abs(pred - true) / norm, axis=(0, 1)).sum()

def MSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean((pred - true) ** 2, axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred - true) ** 2 / norm, axis=(0, 1)).sum()

def RMSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.sqrt(np.mean((pred - true) ** 2, axis=(0, 1)).sum())
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred - true) ** 2 / norm, axis=(0, 1)).sum())

def PSNR(pred, true, min_max_norm=True):
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((pred.astype(np.float32) - true.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    else:
        if min_max_norm:  # normalized to [0,1]
            return 20. * np.log10(1. / np.sqrt(mse))
        else:  # normalized to [0,255]
            return 20. * np.log10(255. / np.sqrt(mse))

def SNR(pred, true):
    """Signal-to-Noise Ratio."""
    signal = ((true) ** 2).mean()
    noise = ((true - pred) ** 2).mean()
    return 10. * np.log10(signal / noise)

def safe_ssim(img1, img2, default_win=7, data_range=1.0, **kwargs):
    """
    更健壮的SSIM计算，自动调节win_size和channel_axis避免报错。
    默认data_range=1.0，适用于归一化[0,1]图像。

    - img1, img2: numpy数组，形状(H,W)或(H,W,C)
    - default_win: 默认窗口大小，必须奇数
    - data_range: 图像数据范围，必须指定
    - 返回 SSIM float
    """
    if img1.ndim == 2:  # 单通道
        h, w = img1.shape
        channel_axis = None
    elif img1.ndim == 3:  # 多通道，如(H,W,C)
        h, w, c = img1.shape
        channel_axis = 2
    else:
        raise ValueError("输入图像维度必须是2或3")

    win_size = default_win
    min_side = min(h, w)
    if min_side < win_size:
        win_size = min_side if min_side % 2 == 1 else min_side - 1
        if win_size < 3:  # 太小了，直接返回1
            return 1.0

    try:
        ssim_val = cal_ssim(img1, img2, win_size=win_size, channel_axis=channel_axis, data_range=data_range, **kwargs)
    except Exception as e:
        print(f"Warning: SSIM计算异常，返回1.0，异常信息: {e}")
        ssim_val = 1.0
    return ssim_val


def POD(hits, misses, eps=1e-6):
    return np.mean((hits + eps) / (hits + misses + eps))

def SUCR(hits, fas, eps=1e-6):
    return np.mean((hits + eps) / (hits + fas + eps))

def CSI(hits, fas, misses, eps=1e-6):
    return np.mean((hits + eps) / (hits + misses + fas + eps))

def sevir_metrics(pred, true, threshold):
    """
    计算hits, fas, misses，用于POD、SUCR、CSI
    pred, true: shape (T, B, H, W, C) 或 (B, T, C, H, W)视数据格式调整
    threshold: 阈值
    """
    # 按你实际数据格式修改转置
    pred = pred.transpose(1, 0, 2, 3, 4)  # (T,B,H,W,C)
    true = true.transpose(1, 0, 2, 3, 4)
    hits, fas, misses = [], [], []
    for i in range(pred.shape[0]):
        t, p = _threshold(pred[i], true[i], threshold)
        hits.append(np.sum(t * p))
        fas.append(np.sum((1 - t) * p))
        misses.append(np.sum(t * (1 - p)))
    return np.array(hits), np.array(fas), np.array(misses)

class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity."""

    def __init__(self, net='alex', use_gpu=True):
        super().__init__()
        assert net in ['alex', 'squeeze', 'vgg']
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if lpips is None:
            raise ImportError("LPIPS package not found.")
        self.loss_fn = lpips.LPIPS(net=net)
        if self.use_gpu:
            self.loss_fn.cuda()

    def forward(self, img1, img2):
        # img1, img2 shape: (H,W,C), values in [-1,1]
        img1_tensor = lpips.im2tensor(img1 * 255)
        img2_tensor = lpips.im2tensor(img2 * 255)
        if self.use_gpu:
            img1_tensor, img2_tensor = img1_tensor.cuda(), img2_tensor.cuda()
        return self.loss_fn(img1_tensor, img2_tensor).squeeze().detach().cpu().numpy()

def metric(pred, true, mean=None, std=None, metrics=['mae', 'mse'], clip_range=[0,1], channel_names=None,
           spatial_norm=False, return_log=True, threshold=74.0):
    """
    计算多种指标。

    pred, true: np.array, shape (B, T, C, H, W)
    mean, std: 标准化反归一化参数
    metrics: 需要计算的指标列表
    channel_names: 如果按通道分组计算，传列表
    spatial_norm: 是否对空间维度归一化
    clip_range: 预测裁剪范围
    threshold: 用于二分类指标的阈值
    """
    global _lpips_calculator
    if mean is not None and std is not None:
        pred = pred * std + mean
        true = true * std + mean

    eval_res = {}
    eval_log = ""

    allowed_metrics = ['mae', 'mse', 'rmse', 'ssim', 'psnr', 'snr', 'lpips', 'pod', 'sucr', 'csi', 'mae_per_frame']
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if invalid_metrics:
        raise ValueError(f"Unsupported metrics: {invalid_metrics}")

    if isinstance(channel_names, list):
        assert pred.shape[2] % len(channel_names) == 0 and len(channel_names) > 1
        c_group = len(channel_names)
        c_width = pred.shape[2] // c_group
    else:
        channel_names, c_group, c_width = None, None, None

    if 'mse' in metrics:
        if channel_names is None:
            eval_res['mse'] = MSE(pred, true, spatial_norm)
        else:
            mse_sum = 0.
            for i, c_name in enumerate(channel_names):
                mse_c = MSE(pred[:, :, i*c_width:(i+1)*c_width], true[:, :, i*c_width:(i+1)*c_width], spatial_norm)
                eval_res[f'mse_{c_name}'] = mse_c
                mse_sum += mse_c
            eval_res['mse'] = mse_sum / c_group

    if 'mae' in metrics:
        if channel_names is None:
            eval_res['mae'] = MAE(pred, true, spatial_norm)
        else:
            mae_sum = 0.
            for i, c_name in enumerate(channel_names):
                mae_c = MAE(pred[:, :, i*c_width:(i+1)*c_width], true[:, :, i*c_width:(i+1)*c_width], spatial_norm)
                eval_res[f'mae_{c_name}'] = mae_c
                mae_sum += mae_c
            eval_res['mae'] = mae_sum / c_group

    if 'rmse' in metrics:
        if channel_names is None:
            eval_res['rmse'] = RMSE(pred, true, spatial_norm)
        else:
            rmse_sum = 0.
            for i, c_name in enumerate(channel_names):
                rmse_c = RMSE(pred[:, :, i*c_width:(i+1)*c_width], true[:, :, i*c_width:(i+1)*c_width], spatial_norm)
                eval_res[f'rmse_{c_name}'] = rmse_c
                rmse_sum += rmse_c
            eval_res['rmse'] = rmse_sum / c_group

    if 'pod' in metrics:
        hits, fas, misses = sevir_metrics(pred, true, threshold)
        eval_res['pod'] = POD(hits, misses)
        eval_res['sucr'] = SUCR(hits, fas)
        eval_res['csi'] = CSI(hits, fas, misses)

    # 限制预测范围，防止异常
    pred = np.clip(pred, clip_range[0], clip_range[1])

    if 'ssim' in metrics:
        ssim_sum = 0
        count = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                ssim_val = safe_ssim(pred[b, f].transpose(1,2,0), true[b, f].transpose(1,2,0), default_win=7, multichannel=True)
                ssim_sum += ssim_val
                count += 1
        eval_res['ssim'] = ssim_sum / max(count, 1)

    if 'psnr' in metrics:
        psnr_sum = 0
        count = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                psnr_sum += PSNR(pred[b, f], true[b, f])
                count += 1
        eval_res['psnr'] = psnr_sum / max(count, 1)

    if 'snr' in metrics:
        snr_sum = 0
        count = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                snr_sum += SNR(pred[b, f], true[b, f])
                count += 1
        eval_res['snr'] = snr_sum / max(count, 1)

    if 'lpips' in metrics:
        if lpips is None:
            print("Warning: lpips package not installed, skipping LPIPS metric.")
        else:
            if _lpips_calculator is None:
                _lpips_calculator = LPIPS(net='alex', use_gpu=torch.cuda.is_available())
            lpips_val = 0
            pred_t = pred.transpose(0, 1, 3, 4, 2)  # (B,T,H,W,C)
            true_t = true.transpose(0, 1, 3, 4, 2)
            count = 0
            for b in range(pred_t.shape[0]):
                for f in range(pred_t.shape[1]):
                    lpips_val += _lpips_calculator(pred_t[b, f], true_t[b, f])
                    count += 1
            eval_res['lpips'] = lpips_val / max(count, 1)

    if 'mae_per_frame' in metrics:
        T = pred.shape[1]
        mae_per_frame = []
        for t in range(T):
            if channel_names is None:
                mae_t = np.mean(np.abs(pred[:, t] - true[:, t]))
            else:
                mae_sum = 0.
                for i in range(c_group):
                    mae_sum += np.mean(np.abs(pred[:, t, i*c_width:(i+1)*c_width] - true[:, t, i*c_width:(i+1)*c_width]))
                mae_t = mae_sum / c_group
            mae_per_frame.append(mae_t)
        eval_res['mae_per_frame'] = np.array(mae_per_frame)

    if return_log:
        for k, v in eval_res.items():
            eval_str = f"{k}:{v}" if len(eval_log) == 0 else f", {k}:{v}"
            eval_log += eval_str

    return eval_res, eval_log

