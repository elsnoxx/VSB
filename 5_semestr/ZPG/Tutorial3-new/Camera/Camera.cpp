#include "Camera.h"


Camera::Camera(const glm::vec3& eye)
    : eye(eye), alpha(0.0f), fi(0.0f) {
    // init target vector
    target = Config::defaultCameraPosition;
    viewMatrix = glm::lookAt(eye, eye + target, up);

}

glm::mat4 Camera::getViewMatrix() {
    // Update view matrix
    viewMatrix = glm::lookAt(eye, eye + target, up);
    return viewMatrix;
}

glm::mat4 Camera::getProjectionMatrix() {
    // Projection perspective: FOV=60°, near plane =0.1, far plane=100
    projectionMatrix = glm::perspective(fov, screenAspectRatio, 0.1f, 70.0f);
    return projectionMatrix;
}

glm::vec3 Camera::getPosition() {
    return eye;
}

glm::vec3 Camera::getTarget() {
    return target;
}

void Camera::updateOrientation(glm::vec2 mouseOffset, float deltaTime) {
    // Update angles based on mouse movement
    alpha += mouseOffset.x * deltaTime;
    fi += mouseOffset.y * deltaTime;

    // Horizontal cyclic rotation
    alpha = std::fmod(alpha, Config::TWO_PI);
    if (alpha < 0.0f) alpha += Config::TWO_PI;

    // Vertical rotation limit
    fi = glm::clamp(fi, -glm::radians(89.0f), glm::radians(89.0f));

    // Calculate new target vector
    glm::vec3 direction;
    direction.x = cos(fi) * sin(alpha);
    direction.y = sin(fi);
    direction.z = -cos(fi) * cos(alpha);

    target = glm::normalize(direction);

    // Notify observers
    notify(ObservableSubjects::SCamera);
}

void Camera::updateScreenSize(int width, int height) {
    screenAspectRatio = width / (float)height;


    float t = glm::clamp((screenAspectRatio - Config::MinAspect) / (Config::MaxAspect - Config::MinAspect), 0.0f, 1.0f);
    float minFov = glm::radians(Config::MinFOV);
    float maxFov = glm::radians(Config::MaxFOV);

    fov = glm::mix(minFov, maxFov, t);

    notify(ObservableSubjects::SCamera);
}

void Camera::forward(float deltaTime) {
    this->eye += glm::normalize(glm::vec3(this->target)) * Config::MovementSpeed * deltaTime;
    notify(ObservableSubjects::SCamera);
}

void Camera::left(float deltaTime) {
    this->eye -= glm::normalize(glm::cross(glm::vec3(this->target), glm::vec3(this->up))) * Config::MovementSpeed * deltaTime;
    notify(ObservableSubjects::SCamera);
}

void Camera::backward(float deltaTime) {
    this->eye -= glm::normalize(glm::vec3(this->target)) * Config::MovementSpeed * deltaTime;
    notify(ObservableSubjects::SCamera);
}

void Camera::right(float deltaTime) {
    this->eye += glm::normalize(glm::cross(glm::vec3(this->target), glm::vec3(this->up))) * Config::MovementSpeed * deltaTime;
    notify(ObservableSubjects::SCamera);
}
