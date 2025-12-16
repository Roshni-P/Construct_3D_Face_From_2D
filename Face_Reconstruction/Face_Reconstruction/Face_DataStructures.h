#pragma once

#include <memory>
#include "glew.h"
#include "glfw3.h"

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