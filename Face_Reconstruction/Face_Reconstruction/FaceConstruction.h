#pragma once
#include <windows.h>
#include <string>

class FaceConstruction
{
public:
	FaceConstruction();
	~FaceConstruction();
	int Reconstruct();
	std::string OpenImageFile(const char* filter = "All Files (*.*)\0*.*\0", HWND owner = NULL);

private:

};