#pragma once
//Include GLEW
#include <GL/glew.h>
//Include GLFW
#include <GLFW/glfw3.h> 

//Include the standard C++ headers  
#include <stdlib.h>
#include <stdio.h>

//Aditional classes
#include "VertexShader.h"
#include "FragmentShader.h"
#include "Model.h"
#include "../general/Callbacks.h"
#include "../Models/sphere.h"

class Application {
	public:

		void initialization();
		void createShaders();
		void createModels();

		void printVersionInfo();
		void updateViewport();

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