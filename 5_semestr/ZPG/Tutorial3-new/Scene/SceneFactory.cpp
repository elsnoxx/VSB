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
         createSceneShrekFamily(),
        
    };
}

Scene* SceneFactory::createSceneShrekFamily() {
    Scene* scene = new Scene();

    // zkontroluj, že modely obsahují texture coords (UV) - jinak shader Textured nic nezobrazí
    DrawableObject* shrek = new DrawableObject(ModelType::Shrek, ShaderType::Textured, TextureType::Shrek);
    Transform ts;
    ts.addTransform(std::make_shared<Scale>(glm::vec3(0.8f)));
    ts.addTransform(std::make_shared<Translation>(glm::vec3(-2.0f, 0.0f, 0.0f)));
    shrek->setTransform(ts);
    scene->addObject(shrek);

    DrawableObject* fiona = new DrawableObject(ModelType::Fiona, ShaderType::Textured, TextureType::Fiona);
    Transform tf;
    tf.addTransform(std::make_shared<Scale>(glm::vec3(0.8f)));
    tf.addTransform(std::make_shared<Translation>(glm::vec3(2.0f, 0.0f, 0.0f)));
    fiona->setTransform(tf);
    scene->addObject(fiona);

    DrawableObject* toilet = new DrawableObject(ModelType::Toilet, ShaderType::Textured, TextureType::Toilet);
    Transform tt;
    tt.addTransform(std::make_shared<Scale>(glm::vec3(0.7f)));
    tt.addTransform(std::make_shared<Translation>(glm::vec3(0.0f, 0.0f, -1.5f)));
    toilet->setTransform(tt);
    scene->addObject(toilet);

    DrawableObject* ground = new DrawableObject(ModelType::Plain, ShaderType::Basic);
    Transform tg;
    tg.addTransform(std::make_shared<Scale>(glm::vec3(50.0f, 1.0f, 50.0f)));
    tg.addTransform(std::make_shared<Translation>(glm::vec3(0.0f, 0.0f, 0.0f)));
    ground->setTransform(tg);
    scene->addObject(ground);


    // přidej jednoduché světlo, aby textury byly viditelné
    PointLight* sunLight = new PointLight(glm::vec3(0.0f, 30.0f, 0.0f), glm::vec3(1.0f, 0.95f, 0.9f), 1.5f);
    scene->addLight(sunLight);

    return scene;
}

Scene* SceneFactory::createSceneSolarSystem() {
    Scene* scene = new Scene();

    // ===== SUN =====
    DrawableObject* sun = new DrawableObject(ModelType::Sun, ShaderType::Textured, TextureType::Sun);
    {
        Transform t;
        t.addTransform(std::make_shared<Scale>(glm::vec3(0.5f)));
        sun->setTransform(t);
    }
    scene->addObject(sun);

    // Real light
    scene->addLight(new PointLight(glm::vec3(0, 0, 0), glm::vec3(1.0f, 0.95f, 0.8f), 10.0f));

    // ===== PLANETS =====
    struct PlanetDef {
        ModelType type;
        TextureType texture;
        float orbitRadius;
        float orbitPeriod;
        float rotate;
        float scale;
    };

    /*std::vector<PlanetDef> planets = {
        { ModelType::Mercury, TextureType::Mars,    2.0f,   12.0f, 10.0f, 0.15f },
        { ModelType::Venus,   TextureType::Mars,    3.0f,   20.0f, 30.0f, 0.20f },
        { ModelType::Earth,   TextureType::Mars,    4.0f,   24.0f,  2.0f, 0.22f },
        { ModelType::Mars,    TextureType::Mars,    5.0f,   32.0f,  3.0f, 0.18f },
        { ModelType::Jupiter, TextureType::Mars,    7.0f,   60.0f, 20.0f, 0.30f },
        { ModelType::Saturn,  TextureType::Mars,    9.0f,   80.0f, 18.0f, 0.55f },
        { ModelType::Uranus,  TextureType::Mars,    11.0f,  110.0f, 25.0f, 0.45f },
        { ModelType::Neptune, TextureType::Mars,    13.0f,  140.0f, 28.0f, 0.40f },
        { ModelType::Pluto,   TextureType::Mars,    15.0f,  200.0f, 35.0f, 0.12f }
    };*/
    std::vector<PlanetDef> planets = {
        { ModelType::Saturn,  TextureType::Mars,    9.0f,   80.0f, 18.0f, 0.55f },
        { ModelType::Uranus,  TextureType::Mars,    11.0f,  110.0f, 25.0f, 0.45f },
        { ModelType::Neptune, TextureType::Mars,    13.0f,  140.0f, 28.0f, 0.40f },
        { ModelType::Pluto,   TextureType::Mars,    15.0f,  200.0f, 35.0f, 0.12f },
        
    }; 

    auto orbitAngle = [](float p) {
        return [p]() { return glfwGetTime() * (360.0 / p); };
        };

    for (auto& p : planets) {
        DrawableObject* obj = new DrawableObject(p.type, ShaderType::Textured, p.texture);

        Transform t;
        t.addTransform(std::make_shared<Rotation>(orbitAngle(p.orbitPeriod), glm::vec3(0, 1, 0)));
        t.addTransform(std::make_shared<Translation>(glm::vec3(p.orbitRadius, 0, 0)));
        t.addTransform(std::make_shared<Rotation>([p]() { return glfwGetTime() * (360.0 / p.rotate); }, glm::vec3(0, 1, 0)));
        t.addTransform(std::make_shared<Scale>(glm::vec3(p.scale)));

        obj->setTransform(t);
        scene->addObject(obj);
    }

    return scene;
}



Scene* SceneFactory::createForestScene() {
    Scene* scene = new Scene();

    // vytváříme hlavní objekty přímo přes ModelType
    DrawableObject* shrek = new DrawableObject(ModelType::Shrek, ShaderType::Textured, TextureType::Shrek);
    {
        Transform ts;
        ts.addTransform(std::make_shared<Scale>(glm::vec3(0.8f)));
        ts.addTransform(std::make_shared<Translation>(glm::vec3(-2.0f, 0.0f, 0.0f)));
        shrek->setTransform(ts);
        scene->addObject(shrek);
    }

    DrawableObject* fiona = new DrawableObject(ModelType::Fiona, ShaderType::Textured, TextureType::Fiona);
    {
        Transform tf;
        tf.addTransform(std::make_shared<Scale>(glm::vec3(0.8f)));
        tf.addTransform(std::make_shared<Translation>(glm::vec3(2.0f, 0.0f, 0.0f)));
        fiona->setTransform(tf);
        scene->addObject(fiona);
    }

    DrawableObject* toilet = new DrawableObject(ModelType::Toilet, ShaderType::Textured, TextureType::Toilet);
    {
        Transform tt;
        tt.addTransform(std::make_shared<Scale>(glm::vec3(0.7f)));
        tt.addTransform(std::make_shared<Translation>(glm::vec3(0.0f, 0.0f, -1.5f)));
        toilet->setTransform(tt);
        scene->addObject(toilet);
    }

    DrawableObject* ground = new DrawableObject(ModelType::Plain, ShaderType::Textured, TextureType::Teren);
    {
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

    // placeObjects nyní přijímá ModelType místo Model* a kontroluje dostupnost modelu v ModelManageru
    auto placeObjects = [&](ModelType modelType, int count, bool isTree) {
        // zkontroluj, že model existuje v manageru (lazy-loaded)
        if (!ModelManager::instance().get(modelType)) return;

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

            // vytvoříme objekt přes ModelType - DrawableObject si v konstruktoru získá model z ModelManageru
            DrawableObject* obj = new DrawableObject(modelType, ShaderType::Phong);
            Transform t;
            float scale = isTree ? distTreeScale(rng) : distBushScale(rng);
            t.addTransform(std::make_shared<Scale>(glm::vec3(scale)));
            t.addTransform(std::make_shared<Translation>(glm::vec3(p.x, 0.0f, p.y)));
            float angle = distRot(rng);
            t.addTransform(std::make_shared<Rotation>([angle]() { return angle; }, glm::vec3(0, 1, 0)));
            obj->setTransform(t);
            scene->addObject(obj);
        }
        };

    // 50 tree a 50 bushes (použijeme enumy místo surových pointerů)
    placeObjects(ModelType::Tree, 50, true);
    placeObjects(ModelType::Bushes, 50, false);

    auto addFireflies = [&](int count) {
        // zkontroluj dostupnost koule (sphere)
        if (!ModelManager::instance().get(ModelType::Sphere)) return;

        std::uniform_real_distribution<float> distHeight(1.0f, 5.0f);
        std::uniform_real_distribution<float> distRange(-30.0f, 30.0f);

        for (int i = 0; i < count; i++) {
            glm::vec3 pos(distRange(rng), distHeight(rng), distRange(rng));

            // vizuální glow koule
            DrawableObject* firefly = new DrawableObject(ModelType::Sphere, ShaderType::Phong);
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

    float offset = 2.5f;

    std::vector<glm::vec3> positions = {
        { offset, 0, 0 },
        {-offset, 0, 0 },
        { 0,  offset, 0 },
        { 0, -offset, 0 }
    };

    for (size_t i = 0; i < positions.size(); ++i) {
        DrawableObject* obj = new DrawableObject(ModelType::Sphere, ShaderType::Phong);

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
        DrawableObject* obj = new DrawableObject(ModelType::Sphere, st);

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


    // vytvoříme jeden objekt s Phong shaderem
    DrawableObject* obj = new DrawableObject(ModelType::Formula1, ShaderType::Phong);

    Transform t;
    t.addTransform(std::make_shared<Scale>(glm::vec3(0.7f)));
    t.addTransform(std::make_shared<Translation>(glm::vec3(-2.0f, 0.0f, 0.0f)));
    
    obj->setTransform(t);
    scene->addObject(obj);

    return scene;
}

Scene* SceneFactory::createSceneTinyObjects() {
    Scene* scene = new Scene();


    // vytvoříme jeden objekt s Phong shaderem
    DrawableObject* obj = new DrawableObject(ModelType::Cube, ShaderType::Phong);

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

    DrawableObject* obj = new DrawableObject(ModelType::Sphere, ShaderType::Basic);

    Transform t;
    t.addTransform(std::make_shared<Translation>(glm::vec3(0, 0, 0)));
    obj->setTransform(t);

    scene->addObject(obj);
    return scene;
}

Scene* SceneFactory::createScene2() {
    Scene* scene = new Scene();

    DrawableObject* obj = new DrawableObject(ModelType::Tree, ShaderType::Basic);

    Transform t;
    t.addTransform(std::make_shared<Scale>(glm::vec3(0.15f)));
    t.addTransform(std::make_shared<Translation>(glm::vec3(-0.5f, -1.5f, 0.0f)));

    obj->setTransform(t);
    scene->addObject(obj);

    return scene;
}

Scene* SceneFactory::createScene3() {
    Scene* scene = new Scene();

    DrawableObject* obj = new DrawableObject(ModelType::Triangle, ShaderType::Basic);

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

    float offset = 3.0f;

    std::vector<glm::vec3> positions = {
        {offset,0,0}, {-offset,0,0}, {0,offset,0}, {0,-offset,0}
    };

    for (auto& pos : positions) {
        DrawableObject* obj = new DrawableObject(ModelType::Sphere, ShaderType::Basic);

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
