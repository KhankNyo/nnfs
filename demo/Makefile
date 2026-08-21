

CCF=-Wall -Wextra -ggdb -I".."
LDF=-lm

.PHONY: all clean

all:xor.exe

xor.exe:*.c ../*.h
	gcc $(CCF) -o $@ xor.c $(LDF)

clean:
	rm xor.exe
