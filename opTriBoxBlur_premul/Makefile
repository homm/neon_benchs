.DEFAULT_GOAL = agnostic
CFLAGS ?= -Wall -O2
EXEC ?= run.64
COMMON ?= main.c ./lodepng.o ./image.o
CC ?= gcc

_ver:
	@$(CC) --version | awk /./
	@echo

lodepng.o:
	$(CC) $(CFLAGS) -c -o ./lodepng.o ../lodepng.c

image.o: lodepng.o
	$(CC) $(CFLAGS) -c -o ./image.o ../image.c

default: image.o
	$(CC) $(CFLAGS) -o $(EXEC) $(COMMON) impl.native.c && ./$(EXEC)
	@echo

novect: image.o
	$(CC) $(CFLAGS) -o $(EXEC) $(COMMON) impl.native.c -fno-tree-vectorize && ./$(EXEC)
	@echo

autovect: image.o
	$(CC) $(CFLAGS) -o $(EXEC) $(COMMON) impl.native.c -ftree-vectorize && ./$(EXEC)
	@echo

neon: image.o
	$(CC) $(CFLAGS) -o $(EXEC) $(COMMON) impl.neon.c && ./$(EXEC)
	@echo

neon_preload: image.o
	$(CC) $(CFLAGS) -o $(EXEC) $(COMMON) impl.neon_preload.c && ./$(EXEC)
	@echo

neon_asm: image.o
	$(CC) $(CFLAGS) -o $(EXEC) $(COMMON) impl.neon_asm.s && ./$(EXEC)
	@echo

sse: image.o
	$(CC) $(CFLAGS) -o $(EXEC) $(COMMON) impl.sse.c -msse4 && ./$(EXEC)
	@echo

avx: image.o
	$(CC) $(CFLAGS) -o $(EXEC) $(COMMON) impl.avx.c -mavx2 && ./$(EXEC)
	@echo

sse_2: image.o  # Alternative implementation which is not correct
	$(CC) $(CFLAGS) -o $(EXEC) $(COMMON) impl.sse_2.c -msse4 && ./$(EXEC)
	@echo

avx_2: image.o  # Alternative implementation which is not correct
	$(CC) $(CFLAGS) -o $(EXEC) $(COMMON) impl.avx_2.c -mavx2 && ./$(EXEC)
	@echo

agnostic: _ver default novect autovect

arm: _ver default novect autovect neon neon_preload neon_asm

x86: _ver default novect autovect sse avx
