CC = gcc
LIBS = -lpthread

serial: example2-serial.c
	$(CC) example2-serial.c `pkg-config --cflags --libs glib-2.0` -o example2-serial

parallel: example2-parallel.c
	$(CC) example2-parallel.c `pkg-config --cflags --libs glib-2.0` -o example2-parallel $(LIBS) -w

cuda: example2.cu
	nvcc example2.cu `pkg-config --cflags --libs glib-2.0` -o example2-cuda

clean:
	rm example2-serial
	rm example2-parallel
	rm example2-cuda