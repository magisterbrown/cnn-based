#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <byteswap.h>
#include <time.h>

typedef struct {
    size_t c,w,h;
    float data[];
} Tensor;
#define tel(ten, c, x, y) ten->data[(c)*ten->w*ten->h + (y)*ten->w + (x)]
#define tsize(ten) (ten)->w*(ten)->h*(ten)->c
#define tcreate(blueprint) memcpy(calloc(sizeof(Tensor)+tsize(&blueprint)*sizeof(float), 1), &blueprint, sizeof(Tensor))

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"
#define max(a,b) a>b?a:b
#define min(a,b) a<b?a:b
static inline void write_image(const char* path, size_t width, size_t height, Tensor *img)
{
    uint8_t *buffer = malloc(width*height*sizeof(int)); 
    float minv = 10000;
    float maxv = -10000;
    for(int i=0;i<tsize(img);i++)
    {
        minv = min(minv, img->data[i]); 
        maxv = max(maxv, img->data[i]); 
    }
    for(int y=0;y<height;y++)
        for(int x=0;x<width;x++)
            buffer[y*width+x] = 255*(tel(img, 0, img->w*x/width, img->h*y/height)-minv)/maxv;
    stbi_write_jpg(path, width, height, 1, buffer, 100);
    printf("Image %s min: %.3f max %.3f\n", path, minv, maxv);
    free(buffer);
}

#include <assert.h>
#include <omp.h>
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
void backward_conv_filter(Tensor* conv_grad, Tensor *input, Tensor *output_grad)
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

void update(Tensor *ten, Tensor *grad)
{
    for(int i=0;i<tsize(ten);i++)
        ten->data[i]+=grad->data[i]*0.0001;
}

int main(void) {

    FILE *mnist = fopen("./data/train-images-idx3-ubyte", "rb");
    int header, rows, cols;
    fseek(mnist, sizeof(int)*2, SEEK_SET);
    fread(&rows, sizeof(int), 1, mnist);
    fread(&cols, sizeof(int), 1, mnist);

    rows = bswap_32(rows);
    cols = bswap_32(cols);


    fseek(mnist, rows*cols*12, SEEK_CUR);

    Tensor *im1 = tcreate(((Tensor){1,rows,cols}));
    uint8_t *buff = malloc(rows*cols);
    fread(buff, rows*cols, 1, mnist);


    for(int i=0;i<tsize(im1);i++) 
        im1->data[i] = (float)buff[i]/255;
    fread(buff, rows*cols, 1, mnist);
    fclose(mnist);

    printf("Rows %d\n", rows);
    printf("Coumns %d\n", cols);

    Tensor *conv = tcreate(((Tensor){1,3,3}));
    Tensor *lrconv = tcreate(((Tensor){1,3,3}));
    Tensor *grad_conv= tcreate(((Tensor){1,3,3}));
    memcpy(conv->data, &((float []){-1,-2,-1,0,0,0,1,2,1}), sizeof(float)*9);
    memcpy(lrconv->data, &((float []){-0.13,0.12,0.01,0.06,-0.01,0.02,-0.1,-0.11,-0.08}), sizeof(float)*9);

    Tensor *result = tcreate(((Tensor){1,rows,cols}));
    Tensor *lresult= tcreate(((Tensor){1,rows,cols}));
    Tensor *loss_grad = tcreate(((Tensor){1,rows,cols}));

    
    forward_conv(im1, conv, result);
    
    for(int i=0;i<10;i++)
    {
        forward_conv(im1, lrconv, lresult);
        float mse = mse_loss(lresult, result, loss_grad);
        printf("Loss %.3f\n", mse);
        backward_conv_filter(grad_conv, im1, loss_grad);
        update(lrconv, grad_conv);
        memset(lresult->data, 0, tsize(lresult));
    }
    

    write_image("data/conv_grad.jpg", 280, 280, grad_conv);
    write_image("data/grad.jpg", 280, 280, loss_grad);
    write_image("data/conv.jpg", 280, 280, result);
    write_image("data/lrconv.jpg", 280, 280, lresult);
    return 0;
}
