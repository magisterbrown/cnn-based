#include <stddef.h>

typedef struct {
    size_t c,w,h;
    float data[];
} Tensor;
#define tel(ten, c, x, y) ten->data[(c)*ten->w*ten->h + (y)*ten->w + (x)]
#define tsize(ten) (ten)->w*(ten)->h*(ten)->c
#define tcreate(blueprint) memcpy(calloc(sizeof(Tensor)+tsize(&blueprint)*sizeof(float), 1), &blueprint, sizeof(Tensor))
