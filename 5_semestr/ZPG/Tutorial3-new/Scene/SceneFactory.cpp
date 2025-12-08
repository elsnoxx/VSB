#include "SceneFactory.h"
#include "../ModelObject/ModelManager.h"
#include "../Light/PointLight.h"


std::vector<Scene*> SceneFactory::createAllScenes() {
    return {
        // Tutorial 2
        //createScene1(),
        //createScene2(),
        //createScene3(),
        //createScene4(),
        //createScene5(),

        // Tutorial 3
        //createSceneSphereLights(),
        //createSceneDifferentModes(),
        //createSceneSolarSystem(),

        //createSceneTinyObjects(),
        //createSceneFormula1(),



        //createForestScene(),

        //Tutorial 5
        createSceneShrekFamily()
        
    };
}

Scene* SceneFactory::createSceneShrekFamily() {
    Scene* scene = new Scene();

    // získat modely z ModelManageru (uprav jména ModelType podle tvého enumu)
    Model* shrekModel = ModelManager::instance().get(ModelType::Shrek);
    Model* fionaModel = ModelManager::instance().get(ModelType::Fiona);
    Model* toiletModel = ModelManager::instance().get(ModelType::Toilet);
    Model* plainModel = ModelManager::instance().get(ModelType::Plain);

    // zkontroluj, že modely obsahují texture coords (UV) - jinak shader Textured nic nezobrazí
    if (shrekModel) {
        DrawableObject* shrek = new DrawableObject(shrekModel, ShaderType::Textured);
        Transform ts;
        ts.addTransform(std::make_shared<Scale>(glm::vec3(0.8f)));
        ts.addTransform(std::make_shared<Translation>(glm::vec3(-2.0f, 0.0f, 0.0f)));
        shrek->setTransform(ts);
        scene->addObject(shrek);
    }

    if (fionaModel) {
        DrawableObject* fiona = new DrawableObject(fionaModel, ShaderType::Textured);
        Transform tf;
        tf.addTransform(std::make_shared<Scale>(glm::vec3(0.8f)));
        tf.addTransform(std::make_shared<Translation>(glm::vec3(2.0f, 0.0f, 0.0f)));
        fiona->setTransform(tf);
        scene->addObject(fiona);
    }

    if (toiletModel) {
        DrawableObject* toilet = new DrawableObject(toiletModel, ShaderType::Textured);
        Transform tt;
        tt.addTransform(std::make_shared<Scale>(glm::vec3(0.7f)));
        tt.addTransform(std::make_shared<Translation>(glm::vec3(0.0f, 0.0f, -1.5f)));
        toilet->setTransform(tt);
        scene->addObject(toilet);
    }

    if (plainModel) {
        DrawableObject* ground = new DrawableObject(plainModel, ShaderType::Basic);
        Transform tg;
        tg.addTransform(std::make_shared<Scale>(glm::vec3(50.0f, 1.0f, 50.0f)));
        tg.addTransform(std::make_shared<Translation>(glm::vec3(0.0f, 0.0f, 0.0f)));
        ground->setTransform(tg);
        scene->addObject(ground);
    }

    // přidej jednoduché světlo, aby textury byly viditelné
    PointLight* sunLight = new PointLight(glm::vec3(0.0f, 30.0f, 0.0f), glm::vec3(1.0f, 0.95f, 0.9f), 1.5f);
    scene->addLight(sunLight);

    return scene;
}

Scene* SceneFactory::createSceneSolarSystem() {
    Scene* scene = new Scene();

    Model* sphereModel = ModelManager::instance().get(ModelType::Sphere);
    if (!sphereModel) return scene;

    // SUN
    DrawableObject* sun = new DrawableObject(sphereModel, ShaderType::Phong);
    {
        Transform ts;
        ts.addTransform(std::make_shared<Scale>(glm::vec3(2.5f))); // větší
        sun->setTransform(ts);
    }
    scene->addObject(sun);

    // světlo u Slunce
    PointLight* sunLight = new PointLight(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.95f, 0.9f), 3.0f);
    scene->addLight(sunLight);


    // parametry oběhů (v sekundách)
    const float earthOrbitPeriod = 20.0f;
    const float earthSelfRotatePeriod = 2.0f;
    const float moonOrbitPeriod = 5.0f;
    const float earthOrbitRadius = 6.0f;
    const float moonOrbitRadius = 1.6f;

    // lambda pro úhel orbity Země (v stupních) - sdílená pro Zemi i Měsíc (aby Měsíc následoval Zemi)
    auto earthOrbitAngle = [earthOrbitPeriod]() -> float {
        return static_cast<float>(glfwGetTime()) * (360.0f / earthOrbitPeriod);
    };

    // EARTH (orbituje kolem Slunce)
    DrawableObject* earth = new DrawableObject(sphereModel, ShaderType::Phong);
    {
        Transform te;
        // 1) orbitální rotace kolem Y (posune místní osu, následná translace je kolem středu světa)
        te.addTransform(std::make_shared<Rotation>(earthOrbitAngle, glm::vec3(0,1,0)));
        // 2) translace od středu (umístí Zemi na orbitu)
        te.addTransform(std::make_shared<Translation>(glm::vec3(earthOrbitRadius, 0.0f, 0.0f)));
        // 3) vlastní rotace Země kolem své osy (vizuální efekt)
        te.addTransform(std::make_shared<Rotation>([]() {
            return static_cast<float>(glfwGetTime()) * (360.0f / 2.0f);
        }, glm::vec3(0,1,0)));
        // 4) měřítko Země
        te.addTransform(std::make_shared<Scale>(glm::vec3(0.6f)));
        earth->setTransform(te);
    }
    scene->addObject(earth);

    // MOON (orbita kolem Země; konstrukce zajistí, že následuje zemní orbitu)
    DrawableObject* moon = new DrawableObject(sphereModel, ShaderType::Phong);
    {
        Transform tm;
        // 1) stejná orbitální rotace kolem Slunce jako Země (díky tomu je Měsíc relativně k Zemi)
        tm.addTransform(std::make_shared<Rotation>(earthOrbitAngle, glm::vec3(0,1,0)));
        // 2) translace na pozici Země (earthOrbitRadius od středu)
        tm.addTransform(std::make_shared<Translation>(glm::vec3(earthOrbitRadius, 0.0f, 0.0f)));
        // 3) rotace Měsíce kolem Země (lokální orbitální rotace)
        tm.addTransform(std::make_shared<Rotation>([]() {
            return static_cast<float>(glfwGetTime()) * (360.0f / 5.0f);
        }, glm::vec3(0,1,0)));
        // 4) translace na vzdálenost od Země
        tm.addTransform(std::make_shared<Translation>(glm::vec3(moonOrbitRadius, 0.0f, 0.0f)));
        // 5) měřítko Měsíce
        tm.addTransform(std::make_shared<Scale>(glm::vec3(0.2f)));
        moon->setTransform(tm);
    }
    scene->addObject(moon);

    // volitelně: malá Zemská atmosféra / indikátor (jiný materiál by bylo potřeba)
    // volitelně: přidejte více planet, upravte periody/radii, nebo přidejte orbitální stopy

    return scene;
}


Scene* SceneFactory::createForestScene() {
    Scene* scene = new Scene();

    // modely
    Model* treeModel = ModelManager::instance().get(ModelType::Tree);
    Model* bushModel = ModelManager::instance().get(ModelType::Bushes);
    Model* plainModel = ModelManager::instance().get(ModelType::Plain);
    Model* sphereModel = ModelManager::instance().get(ModelType::Sphere);


    // přidej terén (velký plane)
    if (plainModel) {
        DrawableObject* ground = new DrawableObject(plainModel, ShaderType::Basic);
        Transform tg;
        tg.addTransform(std::make_shared<Scale>(glm::vec3(50.0f, 1.0f, 50.0f)));
        tg.addTransform(std::make_shared<Translation>(glm::vec3(0.0f, 0.0f, 0.0f)));
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

            DrawableObject* obj = new DrawableObject(model, ShaderType::Phong);
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

    // 50 tree a 50 bushes
    placeObjects(treeModel, 50, true);
    placeObjects(bushModel, 50, false);

    auto addFireflies = [&](int count) {
        if (!sphereModel) return;

        std::uniform_real_distribution<float> distHeight(1.0f, 5.0f);
        std::uniform_real_distribution<float> distRange(-30.0f, 30.0f);

        for (int i = 0; i < count; i++) {
            glm::vec3 pos(distRange(rng), distHeight(rng), distRange(rng));

            // vizuální glow koule
            DrawableObject* firefly = new DrawableObject(sphereModel, ShaderType::Phong);
            Transform tf;
            tf.addTransform(std::make_shared<Scale>(glm::vec3(0.1f)));  // malinká koule
            tf.addTransform(std::make_shared<Translation>(pos));
            firefly->setTransform(tf);
            scene->addObject(firefly);

            // světlo -> dělá glow efekt do okolí
            PointLight* fl = new PointLight(
                pos,
                glm::vec3(1.0f, 0.9f, 0.3f),  // barva světlušky
                1.0f, 0.35f, 0.44f            // rychle klesající světlo
            );

            fl->intensity = 2.0f;             // slabé ale jasné světlo
            scene->addLight(fl);
        }
        };


    PointLight* sunLight = new PointLight(glm::vec3(0.0f, 30.0f, 0.0f), glm::vec3(1.0f, 0.95f, 0.9f), 1.5f);
    scene->addLight(sunLight);
    addFireflies(20);


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

    PointLight* center = new PointLight(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), 6.0f);
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
    PointLight* center = new PointLight(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), 6.0f);
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
