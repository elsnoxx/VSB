#pragma once

// glm
#include <glm/vec2.hpp> // glm::vec2
#include <glm/vec3.hpp> // glm::vec3
#include <glm/glm.hpp>  // glm::mat4

// standard c++ libraries
#include <cstddef>
#include <string>

class Config {
public:
	//File paths for shaders
	static const std::string VertexShadersPath;
	static const std::string FragmentShadersPath;


	//Application settings
	static const int WindowHeight;
	static const int WindowWidth;
	static const char* Title;


	// Movement settings
	static const float MouseSensitivity;
	static const float MovementSpeed;
	static const float TWO_PI;
	static const float PI;
	static const float MinAspect;
	static const float MaxAspect;
	static const float MinFOV;
	static const float MaxFOV;
	static const float UsedFOV;
	static const glm::vec3 upVector;
	static const glm::vec3 defaultCameraPosition;

	// Camera settings
	const float fovDegrees[3];

	static float Shininess;
};

