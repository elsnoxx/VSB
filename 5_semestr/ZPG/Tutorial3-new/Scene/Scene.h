#pragma once
#include <vector>
#include "../ModelObject/DrawableObject.h"
#include "../Shader/ShaderProgram.h"
#include "../Camera/Camera.h"
#include "../Input/InputManager.h"

class Scene {
public:
    Camera* getCamera();

    Scene() = default;

    void addObject(DrawableObject* obj);
    void update(float dt, InputManager& input);
    void draw();

    void bindCameraAndLightToUsedShaders();


private:
    std::vector<DrawableObject*> objects;

protected:
	Camera* camera = new Camera(glm::vec3(0.f, 1.f, 5.f));
};