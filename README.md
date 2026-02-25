# Construct 3D Face Mesh From a 2D image
PLEASE NOTE: ** This application uses a low-resolution PCA model (sfm_shape_3448.bin). For highly unique or complicated, non-average facial shapes, it might not capture fine-grained details compared to more complex models. Hence, kindly bear in mind that it does not support all facial images. Tiny white patches can be seen in a few outputs.
All input images shown here have been taken from https://stocksnap.io and https://unsplash.com**

The project is a basic C++ console based app that allows a user to select a 2D image and construct a 3D face mesh using OpenCV, EOS and OpenGL. The user should select an image file that contains a face. The application identifies facial landmarks using haarcascade. These are shown as bright red dots on the image. (This feature is not displayed currently.)

The ouput object wave format file is read and converted to a 3D mesh comprising of tiny triangles.

Seen below, is a sample 3D face mesh, generated internally during the model creation:

<img width="386" height="543" alt="image" src="https://github.com/user-attachments/assets/abfc685e-d163-4f83-9f88-9f397ea355fa" />

Some final output, 3D models of faces are shown below:

![dwayne-joe-6-Rwiq8VbKs-unsplash](https://github.com/user-attachments/assets/af7aab9a-1f37-4083-bacf-04865dc5d640) <img width="441" height="535" alt="image" src="https://github.com/user-attachments/assets/b10d3867-f639-4df5-bae8-cc9739d29003" />

![StockSnap_8SM5T92EN3](https://github.com/user-attachments/assets/27e3a6c5-b002-41b6-af49-8228d154ef00) <img width="692" height="844" alt="image" src="https://github.com/user-attachments/assets/1dd84d69-2592-4281-9bd4-61124c18abf3" />



