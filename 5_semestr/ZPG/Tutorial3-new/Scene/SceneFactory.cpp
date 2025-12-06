#include "SceneFactory.h"
#include "../ModelObject/ModelManager.h"


std::vector<Scene*> SceneFactory::createAllScenes() {
    return {
        // Tutorial 2
       /* createScene1(),
        createScene2(),
        createScene3(),
        createScene4(),
        createScene5(),*/

        // Tutorial 3
        createSceneSphereLights()
    };
}

Scene* SceneFactory::createSceneSphereLights() {
    Scene* scene = new Scene();

    // sdílený model pro všechny koule
    Model* sphereModel = ModelManager::instance().get(ModelType::Sphere);
    float offset = 3.0f;

    // pozice pro 4 koule
    std::vector<glm::vec3> positions = {
        { offset, 0, 0 },
        {-offset, 0, 0 },
        { 0,  offset, 0 },
        { 0, -offset, 0 }
    };

    // čtyři různé shader typy (musí odpovídat vašemu ShaderType enum)
    std::vector<ShaderType> shaders = {
        ShaderType::Basic,
        ShaderType::Phong,
        ShaderType::Lambert,
        ShaderType::Textured
    };

    // vytvoříme 4 objekty, každý s jiným shaderem
    for (size_t i = 0; i < positions.size() && i < shaders.size(); ++i) {
        DrawableObject* obj = new DrawableObject(sphereModel, shaders[i]);

        Transform t;
        t.addTransform(std::make_shared<Scale>(glm::vec3(0.2f)));
        t.addTransform(std::make_shared<Translation>(positions[i]));

        obj->setTransform(t);
        scene->addObject(obj);
    }

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
