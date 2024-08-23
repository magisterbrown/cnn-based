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


#include <stdarg.h>
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

int main(void) {

    FILE *mnist = fopen("./data/train-images-idx3-ubyte", "rb");
    int header, rows, cols;
    fseek(mnist, sizeof(int)*2, SEEK_SET);
    fread(&rows, sizeof(int), 1, mnist);
    fread(&cols, sizeof(int), 1, mnist);

    rows = bswap_32(rows);
    cols = bswap_32(cols);

    Tensor *im1 = tcreate(((Tensor){1,rows,cols}));
    Tensor *im2 = tcreate(((Tensor){1,rows,cols}));
    uint8_t *buff = malloc(rows*cols);
    fseek(mnist, rows*cols*12, SEEK_CUR);
    fread(buff, rows*cols, 1, mnist);
    for(int i=0;i<tsize(im1);i++) 
        im1->data[i] = (float)buff[i]/255;
    fread(buff, rows*cols, 1, mnist);
    for(int i=0;i<tsize(im2);i++) 
        im2->data[i] = (float)buff[i]/255;
    free(buff);
    fclose(mnist);

    printf("Rows %d\n", rows);
    printf("Coumns %d\n", cols);

    write_image("data/res.jpg", 280, 280, im1);
    return 0;
}
