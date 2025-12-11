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

// Camera class
// - Stores camera position/orientation (eye, target, up).
// - Computes view and projection matrices.
// - Provides methods to update orientation from mouse input and to move the camera.
// - Notifies observers (Subject) when camera parameters change.
class Camera : public Subject
{
private:
    // Horizontal (yaw) and vertical (pitch) rotation angles (radians)
    float alpha; // horizontal rotational angle
    float fi;    // vertical rotation angle

    // World-space camera position and orientation vectors
    glm::vec3 eye;    // camera location (world-space)
    glm::vec3 target; // view direction (unit vector)
    glm::vec3 up = Config::upVector; // world up vector (default from Config)

    // Field of view (radians)
    float fov = Config::UsedFOV;

    // Screen aspect ratio (width/height)
    float screenAspectRatio = 4.0f / 3.0f;

    // Cached matrices
    glm::mat4 viewMatrix = 0;       // view matrix (lookAt)
    glm::mat4 projectionMatrix = 0; // projection matrix (perspective)

public:
    // Construct camera at given world position (eye). Initial direction comes from Config::defaultCameraPosition.
    Camera(const glm::vec3& eye);

    // Matrix getters (view-projection)
    // - getViewMatrix recomputes view matrix from eye, target and up.
    // - getProjectionMatrix computes perspective projection using current FOV and aspect ratio.
    glm::mat4 getViewMatrix();
    glm::mat4 getProjectionMatrix();

    // Accessors for position and current target direction
    glm::vec3 getPosition();
    glm::vec3 getTarget();

    // Update viewport/resolution so projection matrix can be adjusted.
    // width/height are in pixels.
    void updateScreenSize(int width, int height);

    // Explicitly set field of view (in radians).
    void setFOV(float radians);

    // Update camera orientation using mouse movement.
    // - mouseOffset: 2D mouse delta (x,y).
    // - deltaTime: frame time to scale sensitivity.
    // This updates internal yaw (alpha) and pitch (fi) and recomputes the target vector.
    void updateOrientation(glm::vec2 mouseOffset, float deltaTime);

    // Movement helpers: move camera in local forward/back/left/right directions.
    // deltaTime scales movement by frame time and Config::MovementSpeed.
    void forward(float deltaTime);
    void backward(float deltaTime);
    void left(float deltaTime);
    void right(float deltaTime);
};