#pragma once
//Include GLEW
#include <GL/glew.h>
//Include GLFW
#include <GLFW/glfw3.h> 

//Include the standard C++ headers  
#include <stdlib.h>
#include <stdio.h>

// my class
#include "./Shader/VertexShader.h"
#include "./Shader/FragmentShader.h"
#include "./Shader/ShaderProgram.h"
#include "Model.h"
#include "DrawableObject.h"
#include "./Transform/Transform.h"
#include "./Transform/Rotation.h"
#include "./Transform/Scale.h"
#include "./Scene/Scene.h"

class Application {
	public:
		int currentSceneIndex = 0;

		void initialization();
		void createShaders();
		void createModels();

		void createDrawableObjects();

		void printVersionInfo();
		void updateViewport();

		void run();

		void switchScene(int index);
		void createScenes();

	private:
		GLFWwindow* window;
		int width, height;

		// objektovì
		ShaderProgram* shaderProgram;
		Model* model;
		DrawableObject* drawableObject;

		// scene
		std::vector<Scene*> scenes;
		

		
};