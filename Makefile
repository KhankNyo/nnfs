

.PHONY: all clean

all:main.exe

main.exe:*.c *.h
	gcc -Wall -Wextra -ggdb -o $@ main.c

clean:
	rm main.exe
