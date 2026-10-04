#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include "../Shader/ShaderProgram.h"
#include <algorithm>

Camera::Camera()
    : eye(0.0f, 0.0f, 3.0f),
      up(0.0f, 1.0f, 0.0f),
      fov(60.0f),
      aspect(4.0f/3.0f),
      zNear(0.1f),
      zFar(100.0f),
      yaw(0.0f),
      pitch(glm::half_pi<float>()/2.0f), // nějaký počáteční úhel
      firstMouse(true),
      lastX(0.0),
      lastY(0.0),
      sensitivity(0.005f)
{
    updateTargetFromAngles();
}

void Camera::setPosition(const glm::vec3& pos) { eye = pos; notifyObservers(); }
const glm::vec3& Camera::getPosition() const { return eye; }

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(eye, eye + target, up);
}

glm::mat4 Camera::getProjectionMatrix() const {
    return glm::perspective(glm::radians(fov), aspect, zNear, zFar);
}

void Camera::setFOV(float degrees) { fov = degrees; notifyObservers(); }
void Camera::setAspect(float a) { aspect = a; notifyObservers(); }
void Camera::setNearFar(float n, float f) { zNear = n; zFar = f; notifyObservers(); }

void Camera::processMouseMovement(double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos; lastY = ypos; firstMouse = false;
    }
    double xoffset = xpos - lastX;
    double yoffset = lastY - ypos; // invert Y
    lastX = xpos; lastY = ypos;

    yaw   += (float)(xoffset * sensitivity);
    pitch += (float)(yoffset * sensitivity);

    // omezíme pitch (alpha) tak, aby kamera nepřeklopila
    const float eps = 0.001f;
    if (pitch < eps) pitch = eps;
    if (pitch > glm::pi<float>() - eps) pitch = glm::pi<float>() - eps;

    // normalizace yaw
    if (yaw > glm::two_pi<float>()) yaw -= glm::two_pi<float>();
    if (yaw < -glm::two_pi<float>()) yaw += glm::two_pi<float>();

    updateTargetFromAngles();
    notifyObservers();
}

void Camera::updateTargetFromAngles() {
    // target.x = sin(alpha) * cos(fi);
    // target.z = sin(alpha) * sin(fi);
    // target.y = cos(alpha);
    float alpha = pitch; // latitude
    float fi = yaw;      // longitude

    float sinA = sinf(alpha);
    target.x = sinA * cosf(fi);
    target.z = sinA * sinf(fi);
    target.y = cosf(alpha);

    target = glm::normalize(target);
}

void Camera::addObserver(ShaderProgram* shader) {
    if (!shader) return;
    observers.push_back(shader);
}

void Camera::removeObserver(ShaderProgram* shader) {
    observers.erase(std::remove(observers.begin(), observers.end(), shader), observers.end());
}

void Camera::notifyObservers() {
    for (auto* shader : observers) {
        if (shader) shader->onCameraChanged(this);
    }
}