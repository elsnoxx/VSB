#pragma once
//Include GLEW
#include <GL/glew.h>
//Include GLFW
#include <GLFW/glfw3.h> 

//Include the standard C++ headers  
#include <stdlib.h>
#include <stdio.h>

// my class
#include "./Scene/ScreenManager.h"
#include "./Input/InputManager.h"
#include "Config.h"
#include "Callbacks.h"
#include "./ModelObject/ModelManager.h"



class Application {
	public:
		InputManager input;

		void initialization();

		void printVersionInfo();
		void updateViewport(double x, double y);
		
		void switchScene(int index);
		void toggleFOV();

		void run();

	private:
		GLFWwindow* window;
		int width, height;

		// scene
		ScreenManager screenManager;
		
};