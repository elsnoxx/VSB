#include "Camera.h"

Camera::Camera(glm::vec3 startPos)
    : position(startPos), yaw(0.f), pitch(0.f)
{
}

void Camera::notifyCameraChanged() {
    notify(*this);
}

glm::vec3 Camera::getForward() const {
    return glm::normalize(glm::vec3(
        cos(pitch) * sin(yaw),
        sin(pitch),
        cos(pitch) * cos(yaw)
    ));
}

void Camera::moveForward() { position += getForward() * speed; notifyCameraChanged(); }
void Camera::moveBackward() { position -= getForward() * speed; notifyCameraChanged(); }
void Camera::moveLeft() { position -= glm::normalize(glm::cross(getForward(), glm::vec3(0, 1, 0))) * speed; notifyCameraChanged(); }
void Camera::moveRight() { position += glm::normalize(glm::cross(getForward(), glm::vec3(0, 1, 0))) * speed; notifyCameraChanged(); }

void Camera::rotate(float dx, float dy) {
    yaw += dx * sensitivity;
    pitch -= dy * sensitivity;
    pitch = glm::clamp(pitch, -1.5f, 1.5f);
    notifyCameraChanged();
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(position, position + getForward(), glm::vec3(0, 1, 0));
}

glm::mat4 Camera::getProjectionMatrix(float aspect) const {
    return glm::perspective(glm::radians(60.f), aspect, 0.1f, 100.f);
}
