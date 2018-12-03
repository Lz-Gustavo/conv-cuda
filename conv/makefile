LIBS = -std=c++11
CC = g++

filter: main.o
	$(CC) main.o -o filter $(LIBS)

main.o: main.cpp conv.h exceptions.h stb_image.h tiny_jpeg.h
	$(CC) -c main.cpp

simulation: simulation.cpp
	$(CC) simulation.cpp `pkg-config --cflags --libs glib-2.0` -o simulation $(LIBS) -w

clean:
	rm *.o
	rm filter
	rm simulation