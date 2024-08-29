#include "primitives.h"
#include <stdio.h>
#include <assert.h>

//void forward_conv(Tensor *input, Tensor **conv, int n_conv, Tensor *output, int stride)
//{
//    //assert(conv->w == conv->h);
//    //assert(input->w==input->h);
//    //assert(output->w==output->h);
//    //int offs = conv->w/2;
//    //assert((input->w-offs)/stride>=output->w-offs*2);
//    //
//    //for(int y=offs;y<output->h-offs;y++)
//    //    for(int x=offs;x<output->w-offs;x++)
//    //    {
//    //        for (int cx = 0; cx < conv->w; cx++) 
//    //            for (int cy = 0; cy < conv->h; cy++) 
//    //                tel(output, 0, x, y)+= tel(input, 0, x*stride-offs+cx, y*stride-offs+cy)*tel(conv, 0, cx, cy);  
//    //    }
//}
void backward_conv(Tensor* conv_grad, Tensor *input, Tensor *output_grad, int stride)
{
    int offs = conv_grad->w/2;
    assert((input->w-offs)/stride>=output_grad->w-offs*2);

    for(int y=offs;y<output_grad->h-offs;y++)
        for(int x=offs;x<output_grad->w-offs;x++)
        {
            for (int cx = 0; cx < conv_grad->w; cx++) 
                for (int cy = 0; cy < conv_grad->h; cy++) 
                    tel(conv_grad, 0, cx, cy)+=tel(input, 0, x*stride-offs+cx, y*stride-offs+cy)*tel(output_grad, 0, x, y);
        }
}

void backward_conv_input(Tensor *input_grad, Tensor *conv, Tensor *output_grad)
{
    int offs = conv->w/2;
    for(int y=offs;y<input_grad->h-offs;y++)
        for(int x=offs;x<input_grad->w-offs;x++)
        {
            for (int cx = 0; cx < conv->w; cx++) 
                for (int cy = 0; cy < conv->h; cy++) 
                    tel(input_grad, 0, x-offs+cx, y-offs+cy)+=tel(conv, 0, cx, cy)*tel(output_grad, 0, x, y);
        }
}
