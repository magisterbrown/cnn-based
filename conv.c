#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <byteswap.h>

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
    free(buffer);
}

#include <assert.h>
void forward_conv(Tensor *input, Tensor *conv, Tensor *output)
{
    assert(conv->w == conv->h);
    int offs = conv->w/2;
    for(int x=offs;x<input->w-offs;x++)
        for(int y=offs;y<input->h-offs;y++)
        {
            for (int cx = 0; cx < conv->w; cx++) 
                for (int cy = 0; cy < conv->h; cy++) 
                {
                    tel(output, 0, x, y) += tel(input, 0, x-offs+cx, y-offs+cy)*tel(conv, 0, cx, cy);  
                }
        }
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
    memcpy(conv->data, &((float []){-1,-2,-1,0,0,0,1,2,1}), sizeof(float)*9);
    Tensor *result = tcreate(((Tensor){1,rows,cols}));
    forward_conv(im1, conv, result);

    write_image("data/res.jpg", 280, 280, im1);
    write_image("data/conv.jpg", 280, 280, result);
    return 0;
}
