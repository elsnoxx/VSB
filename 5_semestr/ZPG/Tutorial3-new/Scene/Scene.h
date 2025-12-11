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

    // add a light to the per-scene manager (returns the passed pointer or nullptr)
    Light* addLight(Light* light);

    void bindCameraAndLightToUsedShaders();

    // toggle headlight (if present)
    void switchHeadLight();

    int pickAtCursor(double x, double y, glm::vec3* outWorld = nullptr);
    void plantObjectAtWorldPos(const glm::vec3& worldPos, ModelType type = ModelType::Tree, ShaderType shader = ShaderType::Phong);
    void buildBezierFromControlPoints(float speed = 0.25f, bool loop = true);

    void addControlPoint(const glm::vec3& p);
    void clearControlPoints();
    const std::vector<glm::vec3>& getControlPoints() const;


    // selection / deletion
    int getSelectedIndex() const { return selectedIndex; }
    void setSelectedIndex(int idx) { selectedIndex = idx; }
    // remove object by scene-local index (0..objects.size()-1). returns true if removed.
    bool removeObjectAt(int idx);
    // Reset scene camera to initial state (as created)
    void reset();

private:
    std::vector<DrawableObject*> objects;

    // per-scene manager (Scene owns the manager)
    LightManager* lightManager = nullptr;

    // headlight is owned by the Scene (also stored in the manager)
    HeadLight* headLight = nullptr;
    std::vector<glm::vec3> controlPoints;
    int selectedIndex = -1;
    glm::vec3 initialCameraEye;
    glm::vec3 initialCameraTarget;

protected:
    Camera* camera = nullptr;
};