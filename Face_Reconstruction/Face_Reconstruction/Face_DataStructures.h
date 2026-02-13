#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face/facemarkLBF.hpp>
#include "Eigen/Core"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/fitting/linear_shape_fitting.hpp"
#include "eos/fitting/orthographic_camera_estimation_linear.hpp"
#include "eos/render/texture_extraction.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

#include <memory>
#include "glew.h"
#include "glfw3.h"

#include "gl\GL.h"
#include "glut.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define NOMINMAX
#include <windows.h>



class GLFWwindow;

// Flattened structure of vertices, texture coords
struct Vertex
{
	float x, y, z;
	float u, v;
};

struct GLFWwindowDeleter
{
	void operator()(GLFWwindow* window) const noexcept {
		if (window)
		{
			glfwDestroyWindow(window);
		}
	};
};

using GLFWwindowInstance = std::unique_ptr<GLFWwindow, GLFWwindowDeleter>;