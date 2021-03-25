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

vect:
	$(CC) $(CFLAGS) -o $(EXEC) $(MAIN) impl.native.c -ftree-vectorize && ./$(EXEC)
	@echo

neon:
	$(CC) $(CFLAGS) -o $(EXEC) $(MAIN) impl.neon.c && ./$(EXEC)
	@echo

preload:
	$(CC) $(CFLAGS) -o $(EXEC) $(MAIN) impl.preload.c && ./$(EXEC)
	@echo

asm:
	$(CC) $(CFLAGS) -o $(EXEC) $(MAIN) impl.asm.s && ./$(EXEC)
	@echo

all: _ver default novect vect neon preload asm
	