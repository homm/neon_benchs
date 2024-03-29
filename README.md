# ARM NEON benchmarks

This is simple and dumb benchmarks written to explore ARM CPUs performance.

It implements simple alpha blending for premultiplied pixels format using
different techniques.

```c
extern void
opSourceOver_premul(
    uint8_t* restrict Rrgba, const uint8_t* restrict Srgba,
    const uint8_t* restrict Drgba, size_t len)
{
    size_t i = 0;
    for (; i < len*4; i += 4) {
        uint8_t Sa = 255 - Srgba[i + 3];
        Rrgba[i + 0] = DIV255(Srgba[i + 0] * 255 + Drgba[i + 0] * Sa);
        Rrgba[i + 1] = DIV255(Srgba[i + 1] * 255 + Drgba[i + 1] * Sa);
        Rrgba[i + 2] = DIV255(Srgba[i + 2] * 255 + Drgba[i + 2] * Sa);
        Rrgba[i + 3] = DIV255(Srgba[i + 3] * 255 + Drgba[i + 3] * Sa);
    }
}
```


## Runing

```sh
$ make
```
Run default benchmarks, which include not vectorized and auto vectorized variants.

```sh
$ make arm
```
Run benchmarks for ARM target, which include NEON optimized versions.

```sh
$ make x86
```
Run benchmarks for x86 target, which include SSE and AVX optimized versions.

```sh
$ CC='clang' make arm
```
Use different C compiler. Default compiler is `cc`.

```sh
$ CFLAGS='-O3' make arm
```
Use different compiler parameters. Default is `-Wall -O2`.


## Meaning

Each test runs 4 times and each run contains of 100 000 calls of
`opSourceOver_premul` function, which gives 100MPx picture in total.
The results are given in seconds, so, for example, if result is `0.160 s`,
the speed of conversion is 100 / 0.16 = 625 MPx/s.
