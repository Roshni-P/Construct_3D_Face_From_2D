#pragma once
#ifndef TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#endif

#include <windows.h>
#include <string>

using namespace std;
class GLFWwindow;

class FaceConstruction
{
public:
	FaceConstruction();
	~FaceConstruction();
	int Reconstruct();
	string OpenImageFile(const char* filter = "All Files (*.*)\0*.*\0", HWND owner = NULL);
	int Create3DFace(string objfilepath);

private:

};