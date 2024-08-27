#include "primitives.h"
#include <assert.h>

float mse_loss(Tensor *input, Tensor *target, Tensor *input_grad)
{
    float acc = 0;
    int size = tsize(input);
    for(int c=0;c<input->c;c++)
        for(int y=0;y<input->h;y++)
            for(int x=0;x<input->w;x++)
            {
                float error=tel(input,c,x,y)-tel(target,c,x,y);
                tel(input_grad,c,x,y)=2*error/size;
                acc+=error*error;
            }
    return acc/size;
}

