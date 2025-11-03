#pragma once
#include <vector>
#include <memory>
#include <glm/glm.hpp>

// dopředná deklarace
class ShaderProgram;

class Camera {
public:
    Camera();

    // pozice kamery
    void setPosition(const glm::vec3& pos);
    const glm::vec3& getPosition() const;

    // view / projection matice
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;

    // nastavení projekce
    void setFOV(float degrees);
    void setAspect(float aspect);
    void setNearFar(float nearPlane, float farPlane);

    // myš: přijme aktuální kurzor
    void processMouseMovement(double xpos, double ypos);

    // Observer management
    void addObserver(ShaderProgram* shader);
    void removeObserver(ShaderProgram* shader);
    void notifyObservers();

private:
    glm::vec3 eye;
    glm::vec3 target;
    glm::vec3 up;

    float fov;
    float aspect;
    float zNear, zFar;

    float yaw;
    float pitch;

    bool firstMouse;
    double lastX;
    double lastY;
    float sensitivity;

    std::vector<ShaderProgram*> observers;

    void updateTargetFromAngles();
};