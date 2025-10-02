#pragma once
//Include GLEW
#include <GL/glew.h>
//Include GLFW
#include <GLFW/glfw3.h> 



//Include the standard C++ headers  
#include <stdlib.h>
#include <stdio.h>

class Application {
	public:
		Application();
		~Application();


		void initialization();
		void createShaders();
		void createModels();
		void run();

	private:
		GLFWwindow* window;
		GLuint shaderProgram;
		GLuint vertexShader;
		GLuint fragmentShader;
		GLuint VAO;
		GLuint VBO;
		int width, height;

};