

CCF=-Wall -Wextra -ggdb -I".." -O3 -mavx2 -mfma
LDF=-lm

.PHONY: all clean

all:xor.exe number.exe number-i.exe ixor.exe

xor.exe:xor.c ../*.h
	gcc $(CCF) -o $@ xor.c $(LDF)

ixor.exe:ixor.c ../*.h
	gcc $(CCF) -o $@ ixor.c $(LDF)

number.exe:number.c ../*.h
	gcc $(CCF) -o $@ number.c $(LDF)

number-i.exe:number-i.c ../*.h
	gcc $(CCF) -o $@ number-i.c $(LDF)

clean:
	rm *.exe
