#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <iostream>
#include <exception>

class ValueException : public std::exception {

	virtual const char* what() const throw() {
		return "Elements summ cant be zero.";
	}

} v_ecp;

class ResourceException : public std::exception {

	virtual const char* what() const throw() {
		return "Image not found or Insufficient memory.";
	}

} d_ecp;

#endif