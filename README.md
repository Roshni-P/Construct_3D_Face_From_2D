# Construct 3D Face Mesh From a 2D image
PLEASE NOTE: ** This application uses a low-resolution PCA model (sfm_shape_3448.bin). For highly unique or complicated, non-average facial shapes, it might not capture fine-grained details compared to more complex models. Hence, kindly bear in mind that it does not support all facial images. Tiny white patches can be seen in a few outputs.
All input images shown here have been taken from https://stocksnap.io and https://unsplash.com**

## Description
The project is a basic C++ console based app that allows a user to select a 2D image and construct a 3D face mesh using OpenCV, EOS and OpenGL. The user should select an image file that contains a face. The application identifies facial landmarks using haarcascade. These are shown as bright red dots on the image. (This feature is not displayed currently.)

The ouput object wave format file is read and converted to a 3D mesh comprising of tiny triangles.

Seen below, is a sample 3D face mesh, generated internally during the model creation:

<img width="50%" alt="image" src="https://github.com/user-attachments/assets/abfc685e-d163-4f83-9f88-9f397ea355fa" />

## Output
The 3D face models of sample images, are shown below:

<img width="25%" alt="image" src="https://github.com/user-attachments/assets/af7aab9a-1f37-4083-bacf-04865dc5d640" />
<img width="25%" alt="image" src="https://github.com/user-attachments/assets/b10d3867-f639-4df5-bae8-cc9739d29003" />
<br>
<img width="25%" alt="image" src="https://github.com/user-attachments/assets/6c1e1c24-b290-4540-b5ef-fe758ae11e9b" />
<img width="25%" alt="image" src="https://github.com/user-attachments/assets/1dd84d69-2592-4281-9bd4-61124c18abf3" />


## Installation
The application is purely Windows based now. It will be upgraded to support Linux platform, in the future.

### Prerequisites
- C++17 compiler
- OpenCV ≥ 4.12

### Clone the repository
https://github.com/Roshni-P/Construct_3D_Face_From_2D.git
cd Construct_3D_Face_From_2D

### Build
Using Visual Studio
1.	Open <repo-name>.sln
2.	Select Release | x64
3.	Build → Build Solution

### Dependencies:
1.	Install OpenCV from https://opencv.org/releases/ and opencv_contrib. Ensure that same versions are downloaded.
    In the project Solution, add paths to the ‘include’ folders.
2.	Install EOS from https://github.com/patrikhuber/eos.git
3.	To use OpenGL, download GLUT header file, .lib and .dll files.
    You can make use of https://www.opengl.org/resources/libraries/glut/glut_downloads.php
    Add these files to the project Solution.
4.	Get .lib files from GLEW and GLFW from https://www.glfw.org/download.html
5.	Install tinyobjloader to read the mesh obj file, from https://github.com/tinyobjloader/tinyobjloader
    One option is to simply copy the header file into your project and to make sure that TINYOBJLOADER_IMPLEMENTATION is defined exactly once.


