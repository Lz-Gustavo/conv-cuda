#ifndef CONV_H
#define CONV_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "exceptions.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define USE_MGL_NAMESPACE

namespace conv {

	class Image2D {
	private:
		unsigned char* image;
		int size_y;
		int size_x;
		int nrChannels;
		unsigned short** image_copy;

	public:
		Image2D(std::string filename) {

			try {

				image = stbi_load(filename.c_str(), &size_x, &size_y, &nrChannels, 1);			
				if (!image)
					throw (d_ecp);

				image_copy = new unsigned short*[size_y];
				for (int i = 0; i < size_y; ++i) {

					image_copy[i] = new unsigned short[size_x];
					for (int j = 0; j < size_x; ++j) {
						
						// image_copy[i][j] = image[i+j];
						// image_copy[i][j] ? : 0xff;

						image_copy[i][j] = getPixel(i, j);
					}
				}
			} catch (std::exception& e) {

				std::cout << e.what() << std::endl; 
			}
		}
		~Image2D() {
			stbi_image_free(image);

			//delete each cell in image row
			for (int i = 0; i < size_x; ++i) {
				delete[] image_copy[i];
			}
			delete[] image_copy;
		}

		int getWidth() {
			return size_x;
		}
		int getHeight() {
			return size_y;
		}

		unsigned short getPixel(int i, int j) {

			int offset = 2 * (j * size_y + i);
			
			// image pixels are usually stored in big-endian format
			//return image[offset]*256 + image[offset+1];

			//TODO: verify this pixel value return expression
			return image[offset];
		}

		void ApplyMask(std::vector<std::vector<int>*> *kernel) {

			try {

				int mask_summ = 0;
				//check the element summ before use
				for (int i = 0; i < kernel->size(); ++i) {
					for (int j = 0; j < kernel->at(i)->size(); ++j) {
						
						mask_summ += kernel->at(i)->at(j);
					}
				}
				if (mask_summ == 0) {
					throw(v_ecp);
				}
				
				for (int i = 1; i < (size_y - 1) ; ++i) {
					for (int j = 1; j < (size_x - 1); ++j) {
						
					image_copy[i][j] = (
						
						(getPixel(i-1, j-1) * kernel->at(0)->at(0)) +  (getPixel(i-1, j) * kernel->at(0)->at(1)) + (getPixel(i-1, j+1) * kernel->at(0)->at(2))
						+ (getPixel(i, j-1) * kernel->at(1)->at(0)) + (getPixel(i, j) * kernel->at(1)->at(1)) + (getPixel(i, j+1) * kernel->at(1)->at(2))
						+ (getPixel(i+1, j-1) * kernel->at(2)->at(0)) + (getPixel(i+1, j) * kernel->at(2)->at(1)) + (getPixel(i+1, j+1) * kernel->at(2)->at(2))
						
					) / mask_summ; }
				}
			} catch (std::exception &e) {

				std::cout << e.what() << std::endl;
			}
		}

		void showImage() {

			std::cout << "Image pixel map: " << std::endl;
			for (int i = 0; i < size_y; ++i) {
				for (int j = 0; j < size_x; ++j) {

					std::cout << image_copy[i][j] << " ";
				}
				std::cout << std::endl;
			}
		}
	};

	class Kernel {
	private:
		std::vector<std::vector<int>*> *mask;
		int size_x;
		int size_y;

	public:
		Kernel(std::string filename) {

			std::fstream file;	
			std::string line;
			std::stringstream byte, aux_space;
			
			file.open(filename, std::ios::in);
			getline(file, line);
			byte << line;
			byte >> size_x;
			byte >> size_y;
			byte.clear();
			
			//std::cout << "size x: " << size_x << std::endl;
			//std::cout << "size y: " << size_y << std::endl;

			mask = new std::vector<std::vector<int>*>();
			while (getline(file, line)) {
				
				byte << line;
				int aux_number;
				std::vector<int> *row = new std::vector<int>();
				for (int i = 0; i < size_x; ++i) {
					byte >> aux_number;
					row->push_back(aux_number);
				}
				mask->push_back(row);
				//delete &row;
				byte.clear();
			}
			file.close();
			//showMask();
		}
		~Kernel() {
			
			for (int i = 0; i < mask->size(); ++i) {
				delete mask->at(i);
			}
			delete mask;
		}

		std::vector<std::vector<int>*>* getMask() {
			return mask;
		}

		void showMask() {

			std::cout << "Kernel mask: " << std::endl;
			for (int i = 0; i < mask->size(); ++i) {
				for (int j = 0; j < mask->at(i)->size(); ++j) {

					std::cout << mask->at(i)->at(j) << " ";
				}
				std::cout << std::endl;
			}
		}
	};
}

#endif