#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <byteswap.h>
#include <time.h>

#include "primitives/primitives.h"

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

typedef struct {
    int stride;
    int inpt;
    int dim;
    Tensor **convs1; 
    Tensor *interm;
    Tensor **convs2;
    Tensor **downsample;
    Tensor *output;
} Resblock;

#include <assert.h>
void forward_conv(Tensor *input, Tensor **conv, int n_conv, Tensor *output, int stride)
{
    assert(n_conv == output->c);
    for(int i=0;i<n_conv;i++)
    {
        assert(conv[i]->w == conv[i]->h);
    }
    assert(input->w==input->h);
    assert(output->w==output->h);

    int offs = conv[0]->w/2;
    assert((input->w-offs)/stride>=output->w-offs*2);
}
void resblock(Tensor *input, Resblock *block)
{
    forward_conv(input, block->convs1, block->dim, block->interm, block->stride);    
    for(int i=0;i<tsize(block->interm);i++)
        block->interm->data[i] = block->interm->data[i] ? block->interm->data[i]>0 : 0;
    //forward_conv(input, block->convs1, block->dim, output, 1);    
    //for(int i=0;i<tsize(output);i++)
    //    output->data[i] = (output->data[i] + input->data[i])/2;
    
}

#include <math.h>
float normal_dist()
{
    return sqrt(-2*log((float)rand()/RAND_MAX))*cos(2*M_PI*((float)rand()/RAND_MAX));
}

void update(Tensor *ten, Tensor *grad)
{
    for(int i=0;i<tsize(ten);i++)
        ten->data[i]-=grad->data[i]*0.1;
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

    Resblock r1 = {
        .stride = 1,
        .inpt = 1,
        .dim = 8,
    };
    Tensor *convs1[r1.dim];
    Tensor *convs2[r1.dim];
    Tensor *downsample[r1.dim];
    for(int i=0;i<r1.dim;i++) 
    {
        convs1[i] = tcreate(((Tensor){r1.inpt,3,3}));
        tfill(convs1[i], normal_dist()/9); 
        convs2[i] = tcreate(((Tensor){r1.dim,3,3}));
        tfill(convs2[i], normal_dist()/9); 
        downsample[i] = tcreate(((Tensor){r1.inpt,1,1}));
    }
    r1.convs1 = convs1;
    r1.interm = tcreate(((Tensor){r1.dim,rows,cols}));
    r1.convs2 = convs2;
    r1.downsample = downsample;
    r1.output = tcreate(((Tensor){r1.dim,rows,cols}));
    resblock(im1, &r1); 
    printf("Rows %d\n", rows);
    printf("Coumns %d\n", cols);

    Tensor *conv = tcreate(((Tensor){1,3,3}));
    Tensor *lrconv = tcreate(((Tensor){1,3,3}));
    Tensor *grad_conv= tcreate(((Tensor){1,3,3}));
    memcpy(conv->data, &((float []){-1,-2,-1,0,0,0,1,2,1}), sizeof(float)*9);
    tfill(lrconv, normal_dist()/9); 
    
    int stride = 1;
    Tensor *result = tcreate(((Tensor){1,rows/stride,cols/stride}));
    //forward_conv(im1, conv, result, stride);
    
    

    return 0;
}
