#include "Config.h"

// File paths for shaders
const std::string Config::VertexShadersPath = "Shader/ShaderSources/vertex/";
const std::string Config::FragmentShadersPath = "Shader/ShaderSources/fragment/";


// Application settings
const int Config::WindowHeight = 600;
const int Config::WindowWidth = 800;
const char* Config::Title = "ZPG";


//Movement settings
const float Config::MouseSensitivity = 10.0f;
const float Config::MovementSpeed = 1.0f;
const float Config::TWO_PI = glm::radians(360.0f);
const float Config::PI = glm::radians(180.0f);
const float Config::MinAspect = 1.0f;
const float Config::MaxAspect = 2.0f;
const float Config::MinFOV = 50.0f;
const float Config::MaxFOV = 80.0f;
const float Config::UsedFOV = glm::radians(60.0f);
const glm::vec3 Config::upVector = glm::vec3(0.f, 1.f, 0.f);
const glm::vec3 Config::defaultCameraPosition = glm::vec3(0.0f, 0.0f, 1.0f);