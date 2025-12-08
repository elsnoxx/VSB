#pragma once
#include <vector>
#include "../ModelObject/DrawableObject.h"
#include "../Shader/ShaderProgram.h"
#include "../Camera/Camera.h"
#include "../Input/InputManager.h"
#include "../Light/Light.h"
#include "../Light/LightManager.h"
#include "../Light/HeadLight.h"

class Scene {
public:
    Camera* getCamera();

    Scene();
    ~Scene();

    void addObject(DrawableObject* obj);
    void update(float dt, InputManager& input);
    void draw();

    // přidá světlo do per-scene manageru (vrací předané pointer, nebo nullptr)
    Light* addLight(Light* light);

    void bindCameraAndLightToUsedShaders();

    // přepne headlight (pokud existuje)
    void switchHeadLight();

private:
    std::vector<DrawableObject*> objects;

    // per-scene manager (Scene vlastní manager)
    LightManager* lightManager = nullptr;

    // headlight je vlastněné Scene (uložený i v manageru)
    HeadLight* headLight = nullptr;

protected:
    Camera* camera = nullptr;
};