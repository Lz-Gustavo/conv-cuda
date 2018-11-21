LIBS = -std=c++11
CC = g++

filter: main.o
	$(CC) main.o -o filter $(LIBS)

main.o: main.cpp conv.h exceptions.h stb_image.h tiny_jpeg.h
	$(CC) -c main.cpp

clean:
	rm *.o
	rm filter