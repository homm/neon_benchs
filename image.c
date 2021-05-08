#include "image.h"


void
image32_alloc(struct image32* image, uint32_t xsize, uint32_t ysize) {
    image->xsize = 0;
    image->ysize = 0;
    image->data = calloc(xsize * ysize, 4);
    if ( ! image->data) {
        return;
    }
    image->xsize = xsize;
    image->ysize = ysize;
    image->next_element = 1;
    image->next_pixel = 4;
    image->next_line = image->next_pixel * image->xsize;
}

void
image32_free(struct image32* image) {
    free(image->data);
    image->data = NULL;
    image->xsize = 0;
    image->ysize = 0;
}

unsigned
png32_decode(struct image32* image, const char* filename) {
    unsigned err;
    err = lodepng_decode32_file(&image->data, &image->xsize, &image->ysize, filename);
    if (err) {
        free(image->data);
        return err;
    }
    image->next_element = 1;
    image->next_pixel = 4;
    image->next_line = image->next_pixel * image->xsize;
    return 0;
}

unsigned
png32_encode(struct image32* image, const char* filename) {
    return lodepng_encode32_file(filename, image->data, image->xsize, image->ysize);
}
