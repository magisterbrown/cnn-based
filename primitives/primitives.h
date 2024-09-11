#include <stddef.h>

typedef struct {
    size_t c,w,h;
    float data[];
} Tensor;
#define tel(ten, c, x, y) ten->data[(c)*ten->w*ten->h + (y)*ten->w + (x)]
#define tsize(ten) (ten)->w*(ten)->h*(ten)->c
#define tcreate(blueprint) memcpy(calloc(sizeof(Tensor)+tsize(&blueprint)*sizeof(float), 1), &blueprint, sizeof(Tensor))
#define tfill(ten, filler) for(int fillp=0;fillp<tsize(ten);fillp++) {ten->data[fillp]=(filler);}

float mse_loss(Tensor *input, Tensor *target, Tensor *input_grad);
//void forward_conv(Tensor *input, Tensor **conv, int n_conv, Tensor *output, int stride);
void backward_conv(Tensor* conv_grad, Tensor *input, Tensor *output_grad, int stride);
void backward_conv_input(Tensor *input_grad, Tensor *conv, Tensor *output_grad);
