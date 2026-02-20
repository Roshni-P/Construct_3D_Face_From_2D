#include <iostream>
#include "FaceConstruction.h"
#include "eos/core/Image.hpp"
#include "eos/core/image/opencv_interop.hpp"
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/read_pts_landmarks.hpp"
#include "eos/core/write_obj.hpp"
#include "eos/render/render.hpp"

#ifndef TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#endif
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

using namespace cv;
using namespace cv::face;

using namespace eos;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using Eigen::Vector2f;
using Eigen::Vector4f;
using cv::Mat;
using std::cout;
using std::endl;
using std::string;
using std::vector;

// Simple shaders
const char* vertexShaderSource = R"glsl(
    #version 330 core
	layout (location = 0) in vec3 aPos;
	layout (location = 1) in vec2 aTexCoord;

	out vec2 TexCoord; 

	uniform mat4 model;
	uniform mat4 view;
	uniform mat4 projection;

	void main()
	{
		gl_Position = projection * view * model * vec4(aPos, 1.0);
		TexCoord = aTexCoord;
	}
)glsl";

const char* fragmentShaderSource = R"glsl(
    #version 330 core
	in vec2 TexCoord;
    out vec4 FragColor;

	uniform sampler2D ourTexture;

    void main() {
        FragColor = texture(ourTexture, TexCoord);
    }
)glsl";

FaceConstruction::FaceConstruction()
	:winWidth(1366), winHeight(768), window(nullptr)
{
}

FaceConstruction::~FaceConstruction()
{
}

/*	
* Desc:	This function opens the File Open dialog
*		using which we can select the image file.
* Return: Filename
*/
string FaceConstruction::OpenImageFile(const char* filter, HWND owner)
{
	string fileNameStr;
	try
	{
		OPENFILENAMEA ofn;
		//LPWSTR fileName = NULL;
		char filename[MAX_PATH];
		ZeroMemory(&filename, sizeof(filename));
		ZeroMemory(&ofn, sizeof(ofn));

		ofn.lStructSize = sizeof(OPENFILENAMEA);
		ofn.hwndOwner = owner;
		ofn.lpstrFilter = filter;
		ofn.lpstrFile = filename;
		ofn.nMaxFile = MAX_PATH;
		ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
		ofn.lpstrDefExt = "";

		if (GetOpenFileNameA(&ofn))
			fileNameStr = filename;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Caught exception! " << e.what() << std::endl;
		throw e;
	}
	catch (...)
	{
		std::cerr << "Caught an unknown exception type" << std::endl;
		throw;
	}

	return fileNameStr;
}

/*
* Desc:	This function opens a 2D image file, detects 
*		facial landmarks using haarcascade, loads Surrey 
*		Face model. The face is then rendered onto 3D
*		
* Return: Error/Success Code
*		  0 means Success
*/
int FaceConstruction::Reconstruct()
{
	try
	{
		//Select Image File
		string strImgFile = OpenImageFile();
		if (strImgFile.empty())
			return EXIT_FAILURE;

		//Obtain 2D Image
		Mat img = imread(strImgFile, IMREAD_UNCHANGED);
		if (img.empty())
		{
			cout << "Error loading image!" << endl;
			return -1;
		}

		//Detect Facial Landmarks
		vector< vector<Point2f> > facialLandmarks;
		vector<Rect> faces;
		DetectFacialLandmarks(img, facialLandmarks, faces);

		//Load Surrey Face Model
		string objfilepath;
		LoadFaceModel(img, objfilepath, facialLandmarks, faces);
		//Load face fitting model file

		Create3DFace(objfilepath);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Caught exception! " << e.what() << std::endl;
		return 1;
	}
	catch (...)
	{
		std::cerr << "Caught an unknown exception type" << std::endl;
		return 1;
	}

	return EXIT_SUCCESS;
}

int FaceConstruction::Create3DFace(string objfilepath)
{
	try
	{
		// Parsing the obj file
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shape;
		std::vector<tinyobj::material_t> materials;
		std::string warn;
		std::string err;

		bool ret = tinyobj::LoadObj(&attrib, &shape, &materials, &warn, &err, objfilepath.c_str(), OUT_DIR);

		if (!warn.empty()) {
			std::cout << "WARN: " << warn << std::endl;
		}
		if (!err.empty()) {
			std::cerr << "ERR: " << err << std::endl;
		}

		std::vector<unsigned int> indices;
		std::vector<Vertex> vertexBuffer;

		for (const auto& sh : shape) {
			for (const auto& index : sh.mesh.indices) {
				Vertex v;

				//Position - x,y,z
				v.x = (attrib.vertices[3 * index.vertex_index + 0]); //x
				v.y = (attrib.vertices[3 * index.vertex_index + 1]); //y
				v.z = (attrib.vertices[3 * index.vertex_index + 2]); //z

				//Texture Co-ord
				v.u = (attrib.texcoords[2 * index.texcoord_index + 0]); //u
				v.v = (attrib.texcoords[2 * index.texcoord_index + 1]); //v
				vertexBuffer.push_back(v);
				indices.push_back(indices.size());
			}
		}

		if (!glfwInit()) {
			std::cerr << "Failed to initialize GLFW\n";
			return -1;
		}
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

		// We want a normal decorated window
		glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		int errCode = CreateDisplayWindow();
		if (errCode != 0)
			return errCode;

		// Make the window's context current
		glfwMakeContextCurrent(window.get());

		glewExperimental = GL_TRUE;
		if (glewInit() != GLEW_OK) {
			std::cerr << "Failed to initialize GLEW\n";
			return -1;
		}

		AddTexture();

		// Build shaders
		GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
		glCompileShader(vertexShader);

		GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
		glCompileShader(fragmentShader);

		GLuint shaderProgram = glCreateProgram();

		glAttachShader(shaderProgram, vertexShader);
		glAttachShader(shaderProgram, fragmentShader);
		glLinkProgram(shaderProgram);
		// check for linking errors
		int success;
		char infoLog[512];
		glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
		}

		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);

		RenderMesh(shaderProgram, vertexBuffer, indices);

		glDeleteProgram(shaderProgram);

		//Destroy the window before terminating glfw
		glfwDestroyWindow(window.get());
		glfwTerminate();
	}
	catch (const std::exception& e)
	{
		std::cerr << "Caught exception! " << e.what() << std::endl;
		return 1;
	}
	catch (...)
	{
		std::cerr << "Caught an unknown exception type" << std::endl;
		return 1;
	}

	return 0;
}

int FaceConstruction::SetShader(GLuint shaderID)
{
	try
	{
		// Model
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::scale(model, glm::vec3(0.01f));  // shrink face

		// View (move camera back)
		glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, -3.0f));

		// Projection
		glm::mat4 projection = glm::perspective(
			glm::radians(45.0f),
			(float)winWidth / (float)winHeight,
			0.1f,
			100.0f
		);

		// Send to shader
		glUniformMatrix4fv(glGetUniformLocation(shaderID, "model"), 1, GL_FALSE, glm::value_ptr(model));
		glUniformMatrix4fv(glGetUniformLocation(shaderID, "view"), 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(glGetUniformLocation(shaderID, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
	}
	catch (const std::exception& e)
	{
		std::cerr << "Caught exception! " << e.what() << std::endl;
		return 1;
	}
	catch (...)
	{
		std::cerr << "Caught an unknown exception type" << std::endl;
		return 1;
	}

	return 0;
}

int FaceConstruction::CreateDisplayWindow()
{
	try
	{
		// Create window (not fullscreen)
		window = CreateGLFWWindow();

		// Make it maximized - fills the whole screen but keeps title bar + buttons
		glfwMaximizeWindow(window.get());

		glfwMakeContextCurrent(window.get());
		if (!window) {
			std::cerr << "Failed to create GLFW window\n";
			glfwTerminate();
			return -1;
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << "Caught exception! " << e.what() << std::endl;
		return 1;
	}
	catch (...)
	{
		std::cerr << "Caught an unknown exception type" << std::endl;
		return 1;
	}

	return 0;
}

GLFWwindowInstance FaceConstruction::CreateGLFWWindow()
{
	GLFWwindow* win = glfwCreateWindow(winWidth, winHeight, "3D Face", NULL, NULL);
	if (!win)
	{
		glfwTerminate();
		throw std::runtime_error("Failed to create GLFW window!");
	}

	return GLFWwindowInstance(win);
}


int FaceConstruction::RenderMesh(GLuint shaderID, std::vector<Vertex> vertexBuffer, std::vector<unsigned int> indices)
{
	try
	{
		GLuint VAO, VBO, EBO;
		// Setup OpenGL buffers
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glGenBuffers(1, &EBO);

		glBindVertexArray(VAO);

		// VBO: store vertices
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, vertexBuffer.size() * sizeof(Vertex), vertexBuffer.data(), GL_STATIC_DRAW);

		// EBO: store indices
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

		// Position attribute
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
		glEnableVertexAttribArray(0);

		// TexCoord attribute
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);

		glBindVertexArray(0);

		while (!glfwWindowShouldClose(window.get())) {
			if (glfwGetKey(window.get(), GLFW_KEY_ESCAPE) == GLFW_PRESS)
				glfwSetWindowShouldClose(window.get(),
					true);

			// Colour the background in light gray
			glClearColor(0.827f, 0.827f, 0.827f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT);

			// Use shader
			glUseProgram(shaderID);

			// Activate and bind texture unit 0
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textureID);

			// Tell shader sampler to use texture unit 0
			glUniform1i(glGetUniformLocation(shaderID, "ourTexture"), 0);

			glBindVertexArray(VAO);
			SetShader(shaderID);

			// Fill the triangles with color
			glUniform3f(glGetUniformLocation(shaderID, "color"), 1.0f, 0.8f, 0.8f);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, 0);

			glUseProgram(shaderID);
			glUniform3f(glGetUniformLocation(shaderID, "color"), 0.0f, 0.0f, 0.0f); // border color

			// Draw the border of the triangles
			glLineWidth(1.5f);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, 0);

			// restore
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

			glfwSwapBuffers(window.get());
			glfwPollEvents();
		}

		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &VBO);
		glDeleteBuffers(1, &EBO);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Caught exception! " << e.what() << std::endl;
		return 1;
	}
	catch (...)
	{
		std::cerr << "Caught an unknown exception type" << std::endl;
		return 1;
	}

	return 0;
}

int FaceConstruction::AddTexture()
{
	try
	{
		glGenTextures(1, &textureID);
		glBindTexture(GL_TEXTURE_2D, textureID);

		// Set filtering & wrapping
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		stbi_set_flip_vertically_on_load(true);

		// Load image (e.g., with stb_image)
		int width, height, nrChannels;
		unsigned char* data = stbi_load(OUT_DIR "Face.texture.png", &width, &height, &nrChannels, 0);

		if (data) {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
				nrChannels == 4 ? GL_RGBA : GL_RGB,
				GL_UNSIGNED_BYTE, data);
			glGenerateMipmap(GL_TEXTURE_2D);
		}
		else {
			std::cerr << "Failed to load texture\n";
		}

		stbi_image_free(data);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Caught exception! " << e.what() << std::endl;
		return 1;
	}
	catch (...)
	{
		std::cerr << "Caught an unknown exception type" << std::endl;
		return 1;
	}

	return 0;
}

// Detect facial landmarks using ML model file. This loads data
// into OpenCV's Facemark API
int FaceConstruction::DetectFacialLandmarks(const cv::Mat& img, vector<vector<Point2f>>& facialLandmarks,
	vector<Rect>& faces)
{
	try
	{
		// Detects front facing human face
		CascadeClassifier faceCascade;
		faceCascade.load(DATA_DIR "haarcascade_frontalface_alt.xml");

		// Detects facial landmarks using pre-trained model
		Ptr<Facemark> facemark = FacemarkLBF::create();
		facemark->loadModel(DATA_DIR "lbfmodel.yaml");
		cout << "Loaded model" << endl;

		resize(img, img, Size(460, 460), 0, 0, INTER_LINEAR_EXACT);

		Mat gray;
		if (img.channels() > 1)
		{
			cvtColor(img, gray, COLOR_BGR2GRAY);
		}
		else
		{
			// This is gray scale image
			gray = img.clone();
		}
		equalizeHist(gray, gray);

		faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));

		if (facemark->fit(img, faces, facialLandmarks))
		{
			for (size_t i = 0; i < faces.size(); i++)
			{
				// This draws a rectangle around the detected face
				cv::rectangle(img, faces[i], Scalar(255, 0, 0));
			}
			for (unsigned long i = 0; i < faces.size(); i++)
			{
				// This draws dots at the facial landmarks
				for (unsigned long k = 0; k < facialLandmarks[i].size(); k++)
					cv::circle(img, facialLandmarks[i][k], 1, cv::Scalar(0, 0, 255));
			}
			imshow("Detected_shape", img);
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << "Caught exception! " << e.what() << std::endl;
		return 1;
	}
	catch (...)
	{
		std::cerr << "Caught an unknown exception type" << std::endl;
		return 1;
	}

	return 0;
}

/*
* This functions fits facial landmarks onto Surrey 3DMM
*/
int FaceConstruction::LoadFaceModel(const cv::Mat& img, string objfilepath, vector<vector<Point2f>>& facialLandmarks,
	vector<Rect>& faces)
{
	try
	{
		morphablemodel::MorphableModel morphable_model;
		try
		{
			morphable_model = morphablemodel::load_model(DATA_DIR "sfm_shape_3448.bin");
		}
		catch (const std::runtime_error& e)
		{
			cout << "Error loading the Morphable Model: " << e.what() << endl;
			return EXIT_FAILURE;
		}

		// Load Landmark mappings file
		core::LandmarkMapper landmark_mapper;
		try
		{
			landmark_mapper = core::LandmarkMapper(DATA_DIR "ibug_to_sfm.txt");
		}
		catch (const std::exception& e)
		{
			cout << "Error loading the landmark mappings: " << e.what() << endl;
			return EXIT_FAILURE;
		}

		// These will be the final 2D and 3D points used for the fitting:
		vector<Vector4f> model_points; // the points in the 3D shape model
		vector<int> vertex_indices;    // their vertex indices
		vector<Vector2f> image_points; // the corresponding 2D landmark points

		// Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
		for (unsigned long i = 0; i < faces.size(); i++)
		{
			for (int k = 0; k < facialLandmarks[i].size(); ++k)
			{
				const auto converted_name = landmark_mapper.convert(std::to_string(k + 1));
				if (!converted_name)
				{ // no mapping defined for the current landmark
					continue;
				}
				const int vertex_idx = std::stoi(converted_name.value());
				const auto vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
				model_points.emplace_back(vertex.homogeneous());
				vertex_indices.emplace_back(vertex_idx);
				image_points.emplace_back(facialLandmarks[i][k].x, facialLandmarks[i][k].y);
			}
		}

		// Estimate the camera (pose) from the 2D - 3D point correspondences
		fitting::ScaledOrthoProjectionParameters pose =
			fitting::estimate_orthographic_projection_linear(image_points, model_points, true, img.rows);
		fitting::RenderingParameters rendering_params(pose, img.cols, img.rows);

		// The 3D head pose can be recovered as follows - the function returns an Eigen::Vector3f with yaw, pitch,
		// and roll angles:
		const float yaw_angle = rendering_params.get_yaw_pitch_roll()[0];

		// Estimate the shape coefficients by fitting the shape to the landmarks:
		const Eigen::Matrix<float, 3, 4> affine_from_ortho =
			fitting::get_3x4_affine_camera_matrix(rendering_params, img.cols, img.rows);
		const vector<float> fitted_coeffs = fitting::fit_shape_to_landmarks_linear(
			morphable_model.get_shape_model(), affine_from_ortho, image_points, vertex_indices);

		// Obtain the full mesh with the estimated coefficients:
		const core::Mesh mesh = morphable_model.draw_sample(fitted_coeffs, vector<float>());

		// Extract the texture from the image using given mesh and camera parameters:
		const core::Image4u texturemap =
			render::extract_texture(mesh, rendering_params.get_modelview(), rendering_params.get_projection(),
				render::ProjectionType::Orthographic, core::from_mat_with_alpha(img));

		// Save the mesh as textured obj to the output path
		std::filesystem::path outputfile = OUT_DIR "Face.obj";
		core::write_textured_obj(mesh, outputfile.string());
		objfilepath = outputfile.string();

		// And save the texture map:
		outputfile.replace_extension(".texture.png");
		cv::imwrite(outputfile.string(), core::to_mat(texturemap));

		cout << "Finished fitting and wrote result mesh and texture to files with basename "
			<< outputfile << "." << endl;
	}
	catch (const std::exception& e)
	{
		std::cerr << "Caught exception! " << e.what() << std::endl;
		return 1;
	}
	catch (...)
	{
		std::cerr << "Caught an unknown exception type" << std::endl;
		return 1;
	}

	return 0;
}