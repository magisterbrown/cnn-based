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
static inline void write_image(const char* path, size_t width, size_t height, Tensor *img, int chanel)
{
    uint8_t *buffer = malloc(width*height*sizeof(int)); 
    float minv = 10000;
    float maxv = -10000;
    float avg = 0;
    float avgsq = 0;
    for(int i=0;i<tsize(img);i++)
    {
        minv = min(minv, img->data[i]); 
        maxv = max(maxv, img->data[i]); 
        avg+=(img->data[i]-avg)/(i+1);
        avgsq+=(img->data[i]*img->data[i]-avgsq)/(i+1);
    }
    float std = avgsq - avg*avg;
    for(int y=0;y<height;y++)
        for(int x=0;x<width;x++)
            buffer[y*width+x] = 255*(tel(img, chanel, img->w*x/width, img->h*y/height)-minv)/maxv;
    stbi_write_jpg(path, width, height, 1, buffer, 100);
    printf("Image %s min: %.3f max: %.3f avg: %.3f std: %.3f\n", path, minv, maxv, avg, std);
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
// TODO: optinal reset
void forward_conv(Tensor *input, Tensor **conv, int n_conv, Tensor *output, int stride)
{
    assert(n_conv == output->c);
    for(int i=0;i<n_conv;i++)
    {
        assert(conv[i]->w == conv[i]->h);
    }
    assert(input->w==input->h);
    assert(output->w==output->h);
    tfill(output, 0);

    int offs = conv[0]->w/2;
    //assert((input->w-offs)/stride>=output->w-offs*2);
    assert(input->w/stride==output->w);
    for(int cho = 0; cho < output->c; cho++)
        for(int y=offs;y<output->h-offs;y++)
            for(int x=offs;x<output->w-offs;x++)
            {
                for(int chi = 0; chi < conv[cho]->c; chi++)
                    for (int cx = 0; cx < conv[cho]->w; cx++)
                        for (int cy = 0; cy < conv[cho]->h; cy++)
                            tel(output, cho, x, y) += tel(input, chi, x*stride-offs+cx, y*stride-offs+cy)*tel(conv[cho], chi, cx, cy);
            }
}
#include <math.h>
float normal_dist()
{
    return sqrt(-2*log((float)rand()/RAND_MAX))*cos(2*M_PI*((float)rand()/RAND_MAX));
}
void init_resblock(Resblock *block, int dim, int inpt, int stride, int side) 
{
    int rows = side;
    int cols = side;
    Tensor **convs1 = malloc(sizeof(Tensor *)*dim);
    Tensor **convs2 = malloc(sizeof(Tensor *)*dim);
    Tensor **downsample = malloc(sizeof(Tensor *)*dim);
    for(int i=0;i<dim;i++) 
    {
        convs1[i] = tcreate(((Tensor){inpt,3,3}));
        tfill(convs1[i], normal_dist()/9); 
        convs2[i] = tcreate(((Tensor){dim,3,3}));
        tfill(convs2[i], normal_dist()/9); 
        downsample[i] = tcreate(((Tensor){inpt,3,3}));
        tfill(downsample[i], normal_dist()/9); 
    }
    block->stride = stride;
    block->inpt = inpt;
    block->dim = dim;
    block->convs1 = convs1;
    block->interm = tcreate(((Tensor){dim,rows/stride,cols/stride}));
    block->convs2 = convs2;
    block->downsample = downsample;
    block->output = tcreate(((Tensor){dim,rows/stride,cols/stride}));

}
void resblock(Tensor *input, Resblock *block)
{
    // Conv1
    forward_conv(input, block->convs1, block->dim, block->interm, block->stride);    
    forward_conv(input, block->downsample, block->dim, block->output, block->stride);    
    // ReLu 
    for(int i=0;i<tsize(block->interm);i++)
        block->interm->data[i] = block->interm->data[i] ? block->interm->data[i]>0 : 0;
    //// Conv2
    forward_conv(block->interm, block->convs1, block->dim, block->output, 1);    
    ////Acc
    for(int i=0;i<tsize(block->output);i++)
        block->output->data[i]/=2;
    
}


void update(Tensor *ten, Tensor *grad)
{
    for(int i=0;i<tsize(ten);i++)
        ten->data[i]-=grad->data[i]*0.1;
}

void avg_pooler(Tensor *input, Tensor *pooled)
{
    assert(input->c == pooled->c);
    int sz = input->w*input->h;
    for(int c=0;c<input->c;c++)
        for(int y=0;y<input->h;y++)
            for(int x=0;x<input->w;x++)
                tel(pooled, c, 0, 0) += tel(input, c, x, y)/sz;
}

void linearize(Tensor *pooled, float *matrix, float *result, int nres)
{
    for(int i=0;i<nres;i++)    
        for(int j=0;j<pooled->c/nres;j++)    
            result[i]+=matrix[i*j]*tel(pooled, i*j, 0, 0);

}

#define UPSAMPLE 64
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
    //fread(buff, rows*cols, 1, mnist);

    Tensor *upsampled = tcreate(((Tensor){UPSAMPLE,rows,cols}));
    Tensor *avgpool = tcreate(((Tensor){256,1,1}));
    Tensor *upconvs[UPSAMPLE];
    for(int i=0;i<UPSAMPLE;i++)
    {
        upconvs[i] = tcreate(((Tensor){1, 3,3}));
        tfill(upconvs[i], normal_dist()); 
    }

    Resblock blocks[5];
    init_resblock(&blocks[0], 64, 64, 1, 28);
    init_resblock(&blocks[1], 128, 64, 2, 28);
    init_resblock(&blocks[2], 128, 128, 1, 14);
    init_resblock(&blocks[3], 256, 128, 2, 14);
    init_resblock(&blocks[4], 256, 256, 1, 7);

    float linear[256*10];
    for(int i=0;i<256*10;i++)
        linear[i] = normal_dist();
    float output[10];
    memset(&output, 0, 10);

    //Bootstrap
    //Tensor *conv = upconvs[0]; 
    //Runnn
    for(int lp=0;lp<20;lp++)
    {
        fread(buff, rows*cols, 1, mnist);
        for(int i=0;i<tsize(im1);i++) 
            im1->data[i] = (float)buff[i]/255;

        forward_conv(im1, upconvs, UPSAMPLE, upsampled, 1);    
        for(int i=0;i<UPSAMPLE;i++)
        {
            float avg = 0;
            float avgsq = 0;
            int ct = 1;
            for(int y=0;y<upsampled->h;y++)
                for(int x=0;x<upsampled->w;x++)
                {
                    float el = tel(upsampled, i, x, y);
                    avg += (el-avg)/ct;
                    avgsq += (el*el-avgsq)/ct;
                    ct++;
                }
            float std = avgsq-avg*avg;
            printf("%d. Layer std: %f avg: %f \n", i, std, avg);
            for(int j=0;j<tsize(upconvs[i]);j++)
                upconvs[i]->data[j]/=sqrt(std); 
        }
    }
    fclose(mnist);

    resblock(upsampled, &blocks[0]); 
    //resblock(blocks[0].output, &blocks[1]); 
    //resblock(blocks[1].output, &blocks[2]); 
    //resblock(blocks[2].output, &blocks[3]); 
    //resblock(blocks[3].output, &blocks[4]); 
    //avg_pooler(blocks[4].output, avgpool);
    //linearize(avgpool, linear, output, 10);
    //for(int i=0;i<10;i++)
    //    printf("Digit: %d prob: %.3f; ", i, output[i]);
    //printf("\n");
    write_image("data/layers/ups.jpg", 280, 280, upsampled, 0);
    //write_image("data/layers/b0.jpg", 280, 280, blocks[0].output, 0);
    //write_image("data/layers/b1.jpg", 280, 280, blocks[1].output, 0);
    //write_image("data/layers/b2.jpg", 280, 280, blocks[2].output, 0);
    //write_image("data/layers/b3.jpg", 280, 280, blocks[3].output, 0);
    //write_image("data/layers/b4.jpg", 280, 280, blocks[4].output, 0);
    printf("Rows %d\n", rows);
    printf("Coumns %d\n", cols);

    //forward_conv(im1, conv, result, stride);
    
    

    return 0;
}
