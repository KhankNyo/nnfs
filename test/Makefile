

CCF=-Wall -Wextra -ggdb -I".."

.PHONY: all clean

all:main.exe

main.exe:*.c ../*.h
	gcc $(CCF) -o $@ test.c

clean:
	rm main.exe
