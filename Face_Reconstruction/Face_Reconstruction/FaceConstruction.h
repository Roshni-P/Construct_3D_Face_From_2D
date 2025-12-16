#pragma once
#ifndef TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#endif

#include <windows.h>
#include <string>
#include <vector>
#include <memory>
#include "Face_DataStructures.h"

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
	void SetShader(GLuint shaderID);
	int CreateDisplayWindow();
	GLFWwindowInstance CreateGLFWWindow();
	int RenderMesh(GLuint shaderID, std::vector<Vertex> vertexBuffer, std::vector<unsigned int> indices);
	int AddTexture();

	int winWidth;
	int winHeight;
	GLFWwindowInstance window;
	GLuint textureID;

};