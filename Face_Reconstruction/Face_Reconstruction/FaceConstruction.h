#pragma once

#include "Face_DataStructures.h"

using namespace std;
class GLFWwindow;

class FaceConstruction
{
public:
	FaceConstruction();
	~FaceConstruction();
	string OpenImageFile(const char* filter = "All Files (*.*)\0*.*\0", HWND owner = NULL);

	int Reconstruct();
	int Create3DFace(string objfilepath);

private:
	//Display - main thread
	int CreateDisplayWindow();
	GLFWwindowInstance CreateGLFWWindow();

	int DetectFacialLandmarks(const cv::Mat& img, std::vector<std::vector<cv::Point2f>>& facialLandmarks, std::vector<cv::Rect>& faces);
	int LoadFaceModel(const cv::Mat& img, string objfilepath, std::vector<std::vector<cv::Point2f>>& facialLandmarks, std::vector<cv::Rect>& faces);

	void SetShader(GLuint shaderID);
	int AddTexture();

	int RenderMesh(GLuint shaderID, std::vector<Vertex> vertexBuffer, std::vector<unsigned int> indices);

	int winWidth;
	int winHeight;
	GLFWwindowInstance window;
	GLuint textureID;

};