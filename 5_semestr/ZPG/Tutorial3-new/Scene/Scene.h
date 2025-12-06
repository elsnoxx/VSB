#pragma once
#include <vector>
#include "../ModelObject/DrawableObject.h"
#include "../Shader/ShaderProgram.h"
#include "../Camera/Camera.h"
#include "../Input/InputManager.h"
#include "../Light/Light.h"

class Scene {
public:
    Camera* getCamera();

    Scene() = default;

    void addObject(DrawableObject* obj);
    void update(float dt, InputManager& input);
    void draw();

    void addLight(const Light& l) { lights.push_back(l); }

    void bindCameraAndLightToUsedShaders();

    void spawnTree(const glm::vec3& pos);

private:
    std::vector<DrawableObject*> objects;
    std::vector<Light> lights;

protected:
	Camera* camera = new Camera(glm::vec3(0.f, 1.f, 5.f));
};