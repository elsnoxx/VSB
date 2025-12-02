#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "../Observer/Subject.h"
#include <glm/fwd.hpp>

class Camera : public Subject {
public:
    Camera(glm::vec3 startPos);

    void moveForward();
    void moveBackward();
    void moveLeft();
    void moveRight();

    void rotate(float dx, float dy);

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix(float aspect) const;

private:
    glm::vec3 position;
    float yaw;
    float pitch;

    float speed = 0.05f;
    float sensitivity = 0.002f;

    glm::vec3 getForward() const;

    void notifyCameraChanged();
};
