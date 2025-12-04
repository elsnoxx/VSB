#pragma once
//Include GLEW
#include <GL/glew.h>

//Include GLFW
#include <GLFW/glfw3.h>

//Include GLM  
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>

#include "../Observer/Subject.h"
#include "../Observer/ObservableSubjects.h"
#include "../Config.h"

class Camera : public Subject
{
private:
	float alpha;// horisontal rotational angle
	float fi;// vertical rotation angle

	glm::vec3 eye; //camera location
	glm::vec3 target;//view direction vector
	glm::vec3 up = Config::upVector;// up vec

	float fov = Config::UsedFOV;

	float screenAspectRatio = 4.0f / 3.0f;
	glm::mat4 viewMatrix = 0; //view matrix
	glm::mat4 projectionMatrix = 0;// projection matrics

public:
	Camera(const glm::vec3& eye);


	// Matrix getter
	glm::mat4 getViewMatrix();
	glm::mat4 getProjectionMatrix();
	glm::vec3 getPosition();
	glm::vec3 getTarget();
	void updateScreenSize(int width, int height);

	// update angle based on mouse orientation
	void updateOrientation(glm::vec2 mouseOffset, float deltaTime);

	// Movement
	void forward(float deltaTime);
	void backward(float deltaTime);
	void left(float deltaTime);
	void right(float deltaTime);
};

