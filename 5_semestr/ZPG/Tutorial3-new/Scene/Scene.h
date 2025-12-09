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

    int pickAtCursor(double x, double y, glm::vec3* outWorld = nullptr);
    void plantObjectAtWorldPos(const glm::vec3& worldPos, ModelType type = ModelType::Tree, ShaderType shader = ShaderType::Phong);

    // selection / deletion
    int getSelectedIndex() const { return selectedIndex; }
    void setSelectedIndex(int idx) { selectedIndex = idx; }
    // remove object by scene-local index (0..objects.size()-1). returns true if removed.
    bool removeObjectAt(int idx);

private:
    std::vector<DrawableObject*> objects;

    // per-scene manager (Scene vlastní manager)
    LightManager* lightManager = nullptr;

    // headlight je vlastněné Scene (uložený i v manageru)
    HeadLight* headLight = nullptr;

    int selectedIndex = -1;

protected:
    Camera* camera = nullptr;
};