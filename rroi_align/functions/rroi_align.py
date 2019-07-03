import torch
from torch.autograd import Function
from .._ext import rroi_align
import pdb

class RRoiAlignFunction(Function):
    def __init__(ctx, pooled_height, pooled_width, spatial_scale):
        ctx.pooled_width = pooled_width
        ctx.pooled_height = pooled_height
        ctx.spatial_scale = spatial_scale
        ctx.feature_size = None

    def forward(ctx, features, rois): 
        ctx.feature_size = features.size()           
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_().float()
        # ctx.argmax = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_().int()
        ctx.idx_x = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_().float()       # 都是float类型的变量
        ctx.idx_y = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_().float()
        ctx.rois = rois
        if not features.is_cuda:
            _features = features.permute(0, 2, 3, 1)
            roi_pooling.roi_pooling_forward(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                            _features, rois, output)
        else:
            rroi_align.rroi_align_forward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                                 features, rois, output, ctx.idx_x, ctx.idx_y)

        return output

    def backward(ctx, grad_output):
        assert(ctx.feature_size is not None and grad_output.is_cuda)
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_().float()

        rroi_align.rroi_align_backward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                              grad_output, ctx.rois, grad_input, ctx.idx_x, ctx.idx_y)

        return grad_input, None
