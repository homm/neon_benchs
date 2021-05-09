#ifndef __image_h__
#define __image_h__

#include <stdlib.h>
#include <stdint.h>

#include "lodepng.h"

typedef struct {
    uint8_t r, g, b, a;
} pixel32;

typedef struct {
    uint16_t r, g, b, a;
} pixel64;

typedef struct {
    uint32_t r, g, b, a;
} pixel128;


struct image32 {
    uint32_t xsize;
    uint32_t ysize;
    uint32_t next_element;
    uint32_t next_pixel;
    uint32_t next_line;
    uint8_t* data;
};

void
image32_alloc(struct image32* image, uint32_t xsize, uint32_t ysize);

void
image32_free(struct image32* image);

unsigned
png32_decode(struct image32* image, const char* filename);

unsigned
png32_encode(struct image32* image, const char* filename);

uint32_t
all_bits_mask(uint32_t x);

#endif