#pragma once
#ifndef TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#endif

#include <windows.h>
#include <string>
#include "glew.h"
#include <vector>


using namespace std;
class GLFWwindow;

// Flattened structure of vertices, texture coords
struct Vertex
{
	float x, y, z;
	float u, v;
};

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
	int RenderMesh(GLuint shaderID, std::vector<Vertex> vertexBuffer, std::vector<unsigned int> indices);

	int winWidth;
	int winHeight;
	GLFWwindow* window;

};