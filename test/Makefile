

CCF=-Wall -Wextra -ggdb -I".."
LDF=-lm

.PHONY: all clean

all:main.exe

main.exe:*.c ../*.h
	gcc $(CCF) -o $@ test.c $(LDF)

clean:
	rm main.exe
