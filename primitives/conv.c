#include "tensor.h"
#include <assert.h>

void forward_conv(Tensor *input, Tensor *conv, Tensor *output)
{
    assert(conv->w == conv->h);
    int offs = conv->w/2;
    
    for(int y=offs;y<input->h-offs;y++)
        for(int x=offs;x<input->w-offs;x++)
        {
            for (int cx = 0; cx < conv->w; cx++) 
                for (int cy = 0; cy < conv->h; cy++) 
                    tel(output, 0, x, y)+= tel(input, 0, x-offs+cx, y-offs+cy)*tel(conv, 0, cx, cy);  
        }
}
void backward_conv(Tensor* conv_grad, Tensor *input, Tensor *output_grad)
{
    int offs = conv_grad->w/2;
    for(int y=offs;y<input->h-offs;y++)
        for(int x=offs;x<input->w-offs;x++)
        {
            for (int cx = 0; cx < conv_grad->w; cx++) 
                for (int cy = 0; cy < conv_grad->h; cy++) 
                    tel(conv_grad, 0, cx, cy)+=tel(input, 0, x-offs+cx, y-offs+cy)*tel(output_grad, 0, x, y);
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
