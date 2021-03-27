.DEFAULT_GOAL = agnostic
CFLAGS ?= -Wall -O2
EXEC ?= run.64
MAIN ?= main.c
CC ?= gcc

_ver:
	@$(CC) --version | awk /./
	@echo

default:
	$(CC) $(CFLAGS) -o $(EXEC) $(MAIN) impl.native.c && ./$(EXEC)
	@echo

novect:
	$(CC) $(CFLAGS) -o $(EXEC) $(MAIN) impl.native.c -fno-tree-vectorize && ./$(EXEC)
	@echo

autovect:
	$(CC) $(CFLAGS) -o $(EXEC) $(MAIN) impl.native.c -ftree-vectorize && ./$(EXEC)
	@echo

neon:
	$(CC) $(CFLAGS) -o $(EXEC) $(MAIN) impl.neon.c && ./$(EXEC)
	@echo

neon_preload:
	$(CC) $(CFLAGS) -o $(EXEC) $(MAIN) impl.neon_preload.c && ./$(EXEC)
	@echo

neon_asm:
	$(CC) $(CFLAGS) -o $(EXEC) $(MAIN) impl.neon_asm.s && ./$(EXEC)
	@echo

sse:
	$(CC) $(CFLAGS) -o $(EXEC) $(MAIN) impl.sse.c -msse4 && ./$(EXEC)
	@echo

avx:
	$(CC) $(CFLAGS) -o $(EXEC) $(MAIN) impl.avx.c -mavx2 && ./$(EXEC)
	@echo

sse_2:  # Alternative implementation which is not correct
	$(CC) $(CFLAGS) -o $(EXEC) $(MAIN) impl.sse_2.c -msse4 && ./$(EXEC)
	@echo

avx_2:  # Alternative implementation which is not correct
	$(CC) $(CFLAGS) -o $(EXEC) $(MAIN) impl.avx_2.c -mavx2 && ./$(EXEC)
	@echo

agnostic: _ver default novect autovect

arm: _ver default novect autovect neon neon_preload neon_asm

x86: _ver default novect autovect sse avx
