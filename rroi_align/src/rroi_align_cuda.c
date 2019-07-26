#include <THC/THC.h>
#include <math.h>
#include "rroi_align_kernel.h"

extern THCState *state;

int rroi_align_forward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output, 
                        THCudaTensor * idx_x, THCudaTensor * idx_y)
{
    // Grab the input tensor
    float * data_flat = THCudaTensor_data(state, features);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * output_flat = THCudaTensor_data(state, output);
    float * idx_x_flat = THCudaTensor_data(state, idx_x);                   // 每个rroi bin的中心索引
    float * idx_y_flat = THCudaTensor_data(state, idx_y);
    // int * argmax_flat = THCudaIntTensor_data(state, argmax);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 6)
    {
        return 0;
    }

    // data height
    int data_height = THCudaTensor_size(state, features, 2);
    // data width
    int data_width = THCudaTensor_size(state, features, 3);
    // Number of channels
    int num_channels = THCudaTensor_size(state, features, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);

    RROIAlignForwardLaucher(
        data_flat, spatial_scale, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois_flat,
        output_flat, idx_x_flat, idx_y_flat, stream);

    return 1;
}



// 反向传播
int rroi_align_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad, 
                        THCudaTensor * idx_x, THCudaTensor * idx_y)
{
    // Grab the input tensor
    float * top_grad_flat = THCudaTensor_data(state, top_grad);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * bottom_grad_flat = THCudaTensor_data(state, bottom_grad);
    float * idx_x_flat = THCudaTensor_data(state, idx_x);
    float * idx_y_flat = THCudaTensor_data(state, idx_y);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 6)
    {
        return 0;
    }

    // batch size
    int batch_size = THCudaTensor_size(state, bottom_grad, 0);

    // data height
    int data_height = THCudaTensor_size(state, bottom_grad, 2);
    // data width
    int data_width = THCudaTensor_size(state, bottom_grad, 3);
    // Number of channels
    int num_channels = THCudaTensor_size(state, bottom_grad, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);
    RROIAlignBackwardLaucher(
        top_grad_flat, spatial_scale, batch_size, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois_flat, bottom_grad_flat, 
        idx_x_flat, idx_y_flat, stream);

    return 1;
}
