#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "../Observer/Subject.h"
#include <glm/fwd.hpp>

class Camera : public Subject
{
private:
	float alpha;// horisontal rotational angle
	float fi;// vertical rotation angle

	glm::vec3 eye; //camera location
	glm::vec3 target;//view direction vector
	glm::vec3 up = glm::vec3(0.f, 1.f, 0.f);// up vec

	float fov = glm::radians(60.0f);

	float movementSpeed = 6.f;


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
