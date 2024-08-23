#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <byteswap.h>


typedef struct {
    size_t c,w,h;
    float data[];
} Tensor;

#define tel(ten, c, x, y) ten->data[(c)*ten->w*ten->h+(y)*ten->w+(x)]
#define tsize(ten) (ten)->w*(ten)->h*(ten)->c
#define tcreate(blueprint) memcpy(calloc(sizeof(Tensor)+tsize(&blueprint)*sizeof(float), 1), &blueprint, sizeof(Tensor))


#include <stdarg.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"
#define max(a,b) a>b?a:b
#define min(a,b) a<b?a:b
#define write_images(path, ...) write_images_va(path, sizeof((void*[]){__VA_ARGS__})/sizeof(void *),((Tensor*[]){__VA_ARGS__}))

static inline void write_images_va(const char* path, int n, Tensor *imgs[]) 
{
    size_t width = 0;
    size_t height = 0;

    for(int i=0;i<n;i++){
        height+=imgs[i]->h;
        width = max(width, imgs[i]->w);
    }
    uint8_t *buffer = malloc(width*height); 
    int cheight = 0;
    for(int i=0;i<n;i++){
        int minv = 10000;
        int maxv = -10000;
        for(int j=0;j<tsize(imgs[i]);j++)
        {
            minv = min(minv, imgs[i]->data[j]); 
            maxv = max(maxv, imgs[i]->data[j]); 
        }
        for(int j=0;j<imgs[i]->h;j++)
        {
            for(int k=0;k<imgs[i]->w;k++)
            {
               buffer[cheight*width+k] = (uint8_t)(255*(tel(imgs[i],1, k, j)-minv)/maxv); 
            }
            cheight++;
        }
    }
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
    fseek(mnist, rows*cols*0, SEEK_CUR);
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
    
    write_images("data/res.jpg",im1);
    ////int a = (&(Tensor){1,2,3})->c;
    //printf("%d", a);
    return 0;
}
