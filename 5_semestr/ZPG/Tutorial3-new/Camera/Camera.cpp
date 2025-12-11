#include "Camera.h"

// Camera implementation
// The Camera stores its eye (position), target (direction vector),
// up vector and computes view/projection matrices. It also provides
// movement helpers (forward/back/left/right) and notifies observers
// on changes.

Camera::Camera(const glm::vec3& eye)
    : eye(eye), alpha(0.0f), fi(0.0f) {
    // Initialize the target vector and compute the initial view matrix.
    // 'Config::defaultCameraPosition' defines the initial direction.
    target = Config::defaultCameraPosition;
    viewMatrix = glm::lookAt(eye, eye + target, up);
}

glm::mat4 Camera::getViewMatrix() {
    // Recompute and return the view matrix based on current eye/target/up.
    // Uses glm::lookAt(eye, eye+target, up) where target is a direction vector.
    viewMatrix = glm::lookAt(eye, eye + target, up);
    return viewMatrix;
}

glm::mat4 Camera::getProjectionMatrix() {
    // Compute and return the perspective projection matrix.
    // Field of view (fov) and aspect ratio are used with near/far planes.
    projectionMatrix = glm::perspective(fov, screenAspectRatio, 0.1f, 70.0f);
    return projectionMatrix;
}

glm::vec3 Camera::getPosition() {
    // Return camera world position.
    return eye;
}

glm::vec3 Camera::getTarget() {
    // Return the camera direction (target vector).
    return target;
}

void Camera::updateOrientation(glm::vec2 mouseOffset, float deltaTime) {
    // Update camera orientation from mouse movement.
    // mouseOffset.x/y represent delta in mouse movement; deltaTime scales sensitivity.
    // alpha: horizontal angle (yaw), fi: vertical angle (pitch).

    // Update angles based on mouse movement
    alpha += mouseOffset.x * deltaTime;
    fi -= mouseOffset.y * deltaTime;

    // Keep alpha in [0, 2*PI) to avoid overflow and for stable cyclic rotation
    alpha = std::fmod(alpha, Config::TWO_PI);
    if (alpha < 0.0f) alpha += Config::TWO_PI;

    // Clamp vertical rotation to avoid gimbal flip (almost +/-89 degrees)
    fi = glm::clamp(fi, -glm::radians(89.0f), glm::radians(89.0f));

    // Calculate new target (direction) vector using spherical coordinates
    glm::vec3 direction;
    direction.x = cos(fi) * sin(alpha);
    direction.y = sin(fi);
    direction.z = -cos(fi) * cos(alpha);

    // Normalize to keep target as a unit direction vector
    target = glm::normalize(direction);

    // Notify observers that camera changed (used by shader/camera listeners)
    notify(ObservableSubjects::SCamera);
}

void Camera::updateScreenSize(int width, int height) {
    // Update aspect ratio and adjust field-of-view (FOV) based on aspect.
    // This keeps a reasonable view across different window sizes.

    screenAspectRatio = width / (float)height;

    // Interpolate FOV between configured min/max depending on aspect ratio.
    float t = glm::clamp((screenAspectRatio - Config::MinAspect) / (Config::MaxAspect - Config::MinAspect), 0.0f, 1.0f);
    float minFov = glm::radians(Config::MinFOV);
    float maxFov = glm::radians(Config::MaxFOV);

    fov = glm::mix(minFov, maxFov, t);

    // Inform observers (for example to update camera uniforms)
    notify(ObservableSubjects::SCamera);
}

void Camera::forward(float deltaTime) {
    // Move camera forward along the target direction (local forward).
    this->eye += glm::normalize(glm::vec3(this->target)) * Config::MovementSpeed * deltaTime;
    notify(ObservableSubjects::SCamera);
}

void Camera::left(float deltaTime) {
    // Move camera left: compute right vector from target x up and negate it.
    this->eye -= glm::normalize(glm::cross(glm::vec3(this->target), glm::vec3(this->up))) * Config::MovementSpeed * deltaTime;
    notify(ObservableSubjects::SCamera);
}

void Camera::backward(float deltaTime) {
    // Move camera backward (opposite of forward).
    this->eye -= glm::normalize(glm::vec3(this->target)) * Config::MovementSpeed * deltaTime;
    notify(ObservableSubjects::SCamera);
}

void Camera::right(float deltaTime) {
    // Move camera right: cross(target, up) gives the right vector.
    this->eye += glm::normalize(glm::cross(glm::vec3(this->target), glm::vec3(this->up))) * Config::MovementSpeed * deltaTime;
    notify(ObservableSubjects::SCamera);
}

void Camera::setFOV(float radians) {
    // Set the field of view explicitly (in radians) and notify observers.
    fov = radians;
    notify(ObservableSubjects::SCamera);
}