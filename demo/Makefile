

CCF=-Wall -Wextra -ggdb -I".."
LDF=-lm

.PHONY: all clean

all:xor.exe number.exe

xor.exe:xor.c ../*.h
	gcc $(CCF) -o $@ xor.c $(LDF)

number.exe:number.c ../*.h
	gcc $(CCF) -o $@ number.c $(LDF)

clean:
	rm *.exe
