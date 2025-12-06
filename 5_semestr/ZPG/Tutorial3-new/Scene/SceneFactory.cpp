#include "SceneFactory.h"
#include "../ModelObject/ModelManager.h"


std::vector<Scene*> SceneFactory::createAllScenes() {
    return {
        // Tutorial 2
       /* createScene1(),
        createScene2(),
        createScene3(),
        createScene4(),
        createScene5(),
        
        // Tutorial 3
        createSceneSphereLights(),
        createSceneDifferentModes(),

        createSceneTinyObjects(),
        createSceneFormula1(),
        */

        
        createForestScene()
    };
}

Scene* SceneFactory::createForestScene() {
    Scene* scene = new Scene();

    // modely
    Model* treeModel = ModelManager::instance().get(ModelType::Tree);
    Model* bushModel = ModelManager::instance().get(ModelType::Bushes);
    Model* plainModel = ModelManager::instance().get(ModelType::Plain);

    // přidej terén (velký plane)
    if (plainModel) {
        DrawableObject* ground = new DrawableObject(plainModel, ShaderType::Basic);
        Transform tg;
        tg.addTransform(std::make_shared<Scale>(glm::vec3(50.0f, 1.0f, 50.0f)));
        tg.addTransform(std::make_shared<Translation>(glm::vec3(0.0f, -0.5f, 0.0f)));
        ground->setTransform(tg);
        scene->addObject(ground);
    }

    // náhodný generátor
    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distPos(-40.0f, 40.0f);
    std::uniform_real_distribution<float> distTreeScale(0.8f, 1.6f);
    std::uniform_real_distribution<float> distBushScale(0.3f, 0.9f);
    std::uniform_real_distribution<float> distRot(0.0f, 360.0f);

    auto placeObjects = [&](Model* model, int count, bool isTree) {
        if (!model) return;
        const float minDist = isTree ? 2.0f : 1.0f;
        std::vector<glm::vec2> placed;
        int attempts = 0;
        for (int i = 0; i < count && attempts < count * 50; ++attempts) {
            glm::vec2 p(distPos(rng), distPos(rng));
            bool ok = true;
            for (auto& q : placed) {
                if (glm::length(p - q) < minDist) { ok = false; break; }
            }
            if (!ok) continue;
            placed.push_back(p);
            ++i;

            DrawableObject* obj = new DrawableObject(model, ShaderType::Phong); // vyber shader podle potřeby
            Transform t;
            float scale = isTree ? distTreeScale(rng) : distBushScale(rng);
            t.addTransform(std::make_shared<Scale>(glm::vec3(scale)));
            t.addTransform(std::make_shared<Translation>(glm::vec3(p.x, 0.0f, p.y)));
            // rotace kolem Y
            float angle = distRot(rng);
            t.addTransform(std::make_shared<Rotation>([angle]() { return angle; }, glm::vec3(0, 1, 0)));
            obj->setTransform(t);
            scene->addObject(obj);
        }
        };

    // 50 stromů a 50 keřů
    placeObjects(treeModel, 50, true);
    placeObjects(bushModel, 50, false);

    // přidej jedno sluneční/centrální světlo nahoře (posílat do shaderů)
    Light sun;
    sun.position = glm::vec3(0.0f, 30.0f, 0.0f); // vysoko nad scénou
    sun.color = glm::vec3(1.0f, 0.95f, 0.9f);
    sun.intensity = 1.5f;
    scene->addLight(sun);

    return scene;
}


Scene* SceneFactory::createSceneSphereLights() {
    Scene* scene = new Scene();

    Model* sphereModel = ModelManager::instance().get(ModelType::Sphere);
    float offset = 2.5f;

    std::vector<glm::vec3> positions = {
        { offset, 0, 0 },
        {-offset, 0, 0 },
        { 0,  offset, 0 },
        { 0, -offset, 0 }
    };

    for (size_t i = 0; i < positions.size(); ++i) {
        DrawableObject* obj = new DrawableObject(sphereModel, ShaderType::Phong);

        Transform t;
        t.addTransform(std::make_shared<Scale>(glm::vec3(0.8f)));
        t.addTransform(std::make_shared<Translation>(positions[i]));
        obj->setTransform(t);
        scene->addObject(obj);
    }

    // JEDNO centrální světlo (world space) — uprav Z podle polohy kamery
    Light center;
    center.position = glm::vec3(0.0f, 0.0f, 0.0f);
    center.color = glm::vec3(1.0f, 1.0f, 1.0f);
    center.intensity = 6.0f;
    scene->addLight(center);

    return scene;
}

Scene* SceneFactory::createSceneDifferentModes() {
    Scene* scene = new Scene();

    Model* sphereModel = ModelManager::instance().get(ModelType::Sphere);
    float offset = 2.5f;

    std::vector<glm::vec3> positions = {
        { offset, 0, 0 },
        {-offset, 0, 0 },
        { 0,  offset, 0 },
        { 0, -offset, 0 }
    };

    // různé shadery pro jednotlivé koule
    std::vector<ShaderType> shaders = {
        ShaderType::Phong,
        ShaderType::Lambert,
        ShaderType::Basic,
        ShaderType::Textured
    };

    for (size_t i = 0; i < positions.size(); ++i) {
        ShaderType st = shaders[i % shaders.size()];
        DrawableObject* obj = new DrawableObject(sphereModel, st);

        Transform t;
        t.addTransform(std::make_shared<Scale>(glm::vec3(0.8f)));
        t.addTransform(std::make_shared<Translation>(positions[i]));
        obj->setTransform(t);

        scene->addObject(obj);
    }

    // jedno centrální světlo uprostřed (world space)
    Light center;
    center.position = glm::vec3(0.0f, 0.0f, 1.5f); // nastav podle kamery
    center.color = glm::vec3(1.0f);
    center.intensity = 6.0f;
    scene->addLight(center);

    return scene;
}

Scene* SceneFactory::createSceneFormula1() {
    Scene* scene = new Scene();

    // načtení shared OBJ modelu (tinyobjloader)
    Model* cubeModel = ModelManager::instance().get(ModelType::Formula1);

    // vytvoříme jeden objekt s Phong shaderem
    DrawableObject* obj = new DrawableObject(cubeModel, ShaderType::Phong);

    Transform t;
    t.addTransform(std::make_shared<Scale>(glm::vec3(0.7f)));
    t.addTransform(std::make_shared<Translation>(glm::vec3(-2.0f, 0.0f, 0.0f)));
    
    obj->setTransform(t);
    scene->addObject(obj);

    return scene;
}

Scene* SceneFactory::createSceneTinyObjects() {
    Scene* scene = new Scene();

    // načtení shared OBJ modelu (tinyobjloader)
    Model* cubeModel = ModelManager::instance().get(ModelType::Cube);

    // vytvoříme jeden objekt s Phong shaderem
    DrawableObject* obj = new DrawableObject(cubeModel, ShaderType::Phong);

    Transform t;
    t.addTransform(std::make_shared<Scale>(glm::vec3(0.7f)));
    t.addTransform(std::make_shared<Translation>(glm::vec3(-2.0f, 0.0f, 0.0f)));
    // jednoduchá rotace bez závislosti na neexistujícím 'i'
    t.addTransform(std::make_shared<Rotation>([]() {
        return (float)glfwGetTime() * 20.0f;
        }, glm::vec3(0, 1, 0)));

    obj->setTransform(t);
    scene->addObject(obj);

    return scene;
}



Scene* SceneFactory::createScene1() {
    Scene* scene = new Scene();

    Model* sphereModel = ModelManager::instance().get(ModelType::Sphere);
    DrawableObject* obj = new DrawableObject(sphereModel, ShaderType::Basic);

    Transform t;
    t.addTransform(std::make_shared<Translation>(glm::vec3(0, 0, 0)));
    obj->setTransform(t);

    scene->addObject(obj);
    return scene;
}

Scene* SceneFactory::createScene2() {
    Scene* scene = new Scene();

    Model* treeModel = ModelManager::instance().get(ModelType::Tree);
    DrawableObject* obj = new DrawableObject(treeModel, ShaderType::Basic);

    Transform t;
    t.addTransform(std::make_shared<Scale>(glm::vec3(0.15f)));
    t.addTransform(std::make_shared<Translation>(glm::vec3(-0.5f, -1.5f, 0.0f)));

    obj->setTransform(t);
    scene->addObject(obj);

    return scene;
}

Scene* SceneFactory::createScene3() {
    Scene* scene = new Scene();

    float triangle[] = {
         0.0f,  0.5f, 0.0f, 1, 0, 0,
         0.5f, -0.5f, 0.0f, 0, 1, 0,
        -0.5f, -0.5f, 0.0f, 0, 0, 1
    };

    Model* model = new Model(triangle, sizeof(triangle), sizeof(triangle) / (6 * sizeof(float)));
    DrawableObject* obj = new DrawableObject(model, ShaderType::Basic);

    Transform t;
    t.addTransform(std::make_shared<Translation>(glm::vec3(-0.25f, 0.17f, 0.0f)));

    t.addTransform(std::make_shared<Rotation>([]() {
        return (float)glfwGetTime() * 50.0f;
        }, glm::vec3(0, 0, 1)));

    obj->setTransform(t);
    scene->addObject(obj);

    return scene;
}

Scene* SceneFactory::createScene4() {
    Scene* scene = new Scene();

    Model* sphereModel = ModelManager::instance().get(ModelType::Sphere);
    float offset = 3.0f;

    std::vector<glm::vec3> positions = {
        {offset,0,0}, {-offset,0,0}, {0,offset,0}, {0,-offset,0}
    };

    for (auto& pos : positions) {
        DrawableObject* obj = new DrawableObject(sphereModel, ShaderType::Basic);

        Transform t;
        t.addTransform(std::make_shared<Scale>(glm::vec3(0.2f)));
        t.addTransform(std::make_shared<Translation>(pos));

        obj->setTransform(t);
        scene->addObject(obj);
    }

    return scene;
}

Scene* SceneFactory::createScene5() {
    Scene* scene = new Scene();


    return scene;
}
