#include "SceneFactory.h"
#include "../ModelObject/ModelManager.h"
#include "../Light/PointLight.h"
#include "../Transform/Bezier.h"
#include "../Light/DirectionalLight.h"


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

        //createSceneFormula1(),

        createForestScene(),
        
    };
}

Scene* SceneFactory::createSceneSolarSystem()
{
    Scene* scene = new Scene();

    // =========================
    //  SUN (root object)
    // =========================
    auto sunNode = std::make_shared<TransformNode>();
    sunNode->addTransform(std::make_shared<Scale>(glm::vec3(1.0f)));

    DrawableObject* sun = new DrawableObject(ModelType::Sphere, ShaderType::Textured, TextureType::Sun);
    sun->setTransformNode(sunNode);   // use shared_ptr API
    scene->addObject(sun);

    scene->addLight(new PointLight(glm::vec3(0, 0, 0), glm::vec3(1.0f, 0.95f, 0.8f), 25.0f));


    // =========================
    //  EARTH
    // =========================
    auto earthNode = std::make_shared<TransformNode>();

    // Orbit Earth around Sun
    earthNode->addTransform(std::make_shared<Rotation>([]() { return glfwGetTime() * 8.0f; }, glm::vec3(0, 1, 0)));
    earthNode->addTransform(std::make_shared<Translation>(glm::vec3(2.5f, 0, 0)));
    earthNode->addTransform(std::make_shared<Rotation>([]() { return glfwGetTime() * 40.0f; }, glm::vec3(0, 1, 0)));
    earthNode->addTransform(std::make_shared<Scale>(glm::vec3(0.10f)));

    sunNode->addChild(earthNode); // Earth is child of Sun

    DrawableObject* earth = new DrawableObject(ModelType::Sphere, ShaderType::Textured, TextureType::Earth);
    earth->setTransformNode(earthNode);
    scene->addObject(earth);


    // =========================
    //  MOON (child of Earth)
    // =========================
    auto moonNode = std::make_shared<TransformNode>();
    moonNode->addTransform(std::make_shared<Rotation>([]() { return glfwGetTime() * 25.0f; }, glm::vec3(0, 1, 0)));
    moonNode->addTransform(std::make_shared<Translation>(glm::vec3(0.6f, 0, 0)));
    moonNode->addTransform(std::make_shared<Scale>(glm::vec3(0.03f)));

    earthNode->addChild(moonNode); // Attach moon to earth

    DrawableObject* moon = new DrawableObject(ModelType::Sphere, ShaderType::Textured, TextureType::Moon);
    moon->setTransformNode(moonNode);
    scene->addObject(moon);


    // =========================
    //  OTHER PLANETS (simple)
    // =========================
    struct PlanetDef { TextureType texture; float orbitRadius, orbitSpeed, selfRotate, scale; };

    std::vector<PlanetDef> planets = {
        { TextureType::Mercury, 1.5f, 15.0f, 20.0f, 0.05f },
        { TextureType::Venus,   2.0f, 10.0f, 15.0f, 0.09f },
        { TextureType::Mars,    3.2f,  7.0f, 18.0f, 0.07f },
        { TextureType::Jupiter, 5.0f,  4.0f, 25.0f, 0.25f },
        { TextureType::Saturn,  6.0f,  3.0f, 22.0f, 0.22f },
        { TextureType::Uranus,  7.5f,  2.0f, 20.0f, 0.18f },
        { TextureType::Neptune, 8.7f,  1.7f, 18.0f, 0.17f },
        { TextureType::Pluto,  10.0f,  1.0f, 13.0f, 0.03f }
    };

    for (auto& p : planets)
    {
        auto node = std::make_shared<TransformNode>();
        node->addTransform(std::make_shared<Rotation>([p]() { return glfwGetTime() * p.orbitSpeed; }, glm::vec3(0, 1, 0)));
        node->addTransform(std::make_shared<Translation>(glm::vec3(p.orbitRadius, 0, 0)));
        node->addTransform(std::make_shared<Rotation>([p]() { return glfwGetTime() * p.selfRotate; }, glm::vec3(0, 1, 0)));
        node->addTransform(std::make_shared<Scale>(glm::vec3(p.scale)));

        sunNode->addChild(node);

        DrawableObject* planet = new DrawableObject(ModelType::Sphere, ShaderType::Textured, p.texture);
        planet->setTransformNode(node);
        scene->addObject(planet);
    }

    return scene;
}





Scene* SceneFactory::createForestScene() {
    Scene* scene = new Scene();

    // vytváříme hlavní objekty přímo přes ModelType
    DrawableObject* shrek = new DrawableObject(ModelType::Shrek, ShaderType::Phong, TextureType::Shrek);
    {
        Transform ts;
        ts.addTransform(std::make_shared<Scale>(glm::vec3(0.8f)));
        ts.addTransform(std::make_shared<Translation>(glm::vec3(-2.0f, 0.0f, 0.0f)));
        shrek->setTransform(ts);
        scene->addObject(shrek);
    }

    DrawableObject* fiona = new DrawableObject(ModelType::Fiona, ShaderType::Phong, TextureType::Fiona);
    {
        Transform tf;
        tf.addTransform(std::make_shared<Scale>(glm::vec3(0.8f)));
        tf.addTransform(std::make_shared<Translation>(glm::vec3(2.0f, 0.0f, 0.0f)));
        fiona->setTransform(tf);
        scene->addObject(fiona);
    }

    DrawableObject* toilet = new DrawableObject(ModelType::Toilet, ShaderType::Phong, TextureType::Toilet);
    {
        Transform tt;
        tt.addTransform(std::make_shared<Scale>(glm::vec3(0.7f)));
        tt.addTransform(std::make_shared<Translation>(glm::vec3(0.0f, 0.0f, -1.5f)));
        toilet->setTransform(tt);
        scene->addObject(toilet);
    }

    DrawableObject* ground = new DrawableObject(ModelType::Plain, ShaderType::Phong, TextureType::Teren);
    {
        Transform tg;
        tg.addTransform(std::make_shared<Scale>(glm::vec3(50.0f, 1.0f, 50.0f)));
        tg.addTransform(std::make_shared<Translation>(glm::vec3(0.0f, 0.0f, 0.0f)));
        ground->setTransform(tg);
        scene->addObject(ground);
    }

    DrawableObject* login = new DrawableObject(ModelType::Login, ShaderType::Phong, TextureType::WoodenFence);
    {
        static float angle = 0.0f;
        Transform tl;
        tl.addTransform(std::make_shared<Scale>(glm::vec3(1.0f)));
        tl.addTransform(std::make_shared<Rotation>(
            []() {
                static float rot = 0.0f;
                rot += 1.0f;
                return rot;
            },
            glm::vec3(0, 1, 0)
        ));

        tl.addTransform(std::make_shared<Translation>(glm::vec3(0.0f, 3.0f, 0.0f)));

        login->setTransform(tl);
        scene->addObject(login);
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
            DrawableObject* obj = new DrawableObject(modelType, ShaderType::Phong, TextureType::Grass);
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
        std::uniform_real_distribution<float> distRadius(0.2f, 2.0f);
        std::uniform_real_distribution<float> distSpeed(0.5f, 2.5f);
        std::uniform_real_distribution<float> distPhase(0.0f, 6.28318530718f);

        for (int i = 0; i < count; i++) {
            glm::vec3 basePos(distRange(rng), distHeight(rng), distRange(rng));

            // vizuální glow koule - menší a s jednoduchou barevnou texturou
            DrawableObject* firefly = new DrawableObject(ModelType::Sphere, ShaderType::Phong, TextureType::Yellow);
            {
                // náhodné parametry pohybu
                float radius = distRadius(rng);
                float speed = distSpeed(rng);
                float phase = distPhase(rng);
                float bobAmp = 0.25f + (radius * 0.1f);

                Transform tf;
                // velmi malá koule
                tf.addTransform(std::make_shared<Scale>(glm::vec3(0.03f)));

                firefly->setTransform(tf);
                scene->addObject(firefly);
            }

            // světlo -> menší dosah, nižší intensity (rychle klesající)
            PointLight* fl = new PointLight(
                basePos,
                glm::vec3(1.0f, 0.95f, 0.6f),  // barva světlušky
                1.0f,   // constant
                1.5f,   // linear (větší hodnota = rychlejší útlum)
                2.5f    // quadratic (větší = ještě rychlejší útlum)
            );

            fl->intensity = 0.8f;      // celkově slabší světlo
            scene->addLight(fl);
        }
        };

    // 1) Directional light (sun) - rovnoměrné světlo směrem dolů
    DirectionalLight* sunDirectional = new DirectionalLight(
        glm::normalize(glm::vec3(-0.3f, -1.0f, -0.2f)),   // direction
        glm::vec3(1.0f, 0.95f, 0.9f)                      // color
    );
    sunDirectional->intensity = 0.9f; // nastav intenzitu zvlášť
    scene->addLight(sunDirectional);

    // 2) Main point light (dálkový "sun glow") - krátký dosah, vyšší pozice
    PointLight* sunPoint = new PointLight(
        glm::vec3(0.0f, 25.0f, 0.0f),  // position
        glm::vec3(1.0f, 0.95f, 0.9f),  // color
        1.0f,  // constant
        0.022f, // linear
        0.0019f // quadratic (pomalejší útlum než fireflies)
    );
    sunPoint->intensity = 1.2f;
    scene->addLight(sunPoint);

    // příklad: pokud SpotLight má signaturu (pos, dir, innerCos, outerCos, color, constant, linear, quadratic)
    SpotLight* searchLight = new SpotLight(
        glm::vec3(10.0f, 8.0f, 10.0f),                       // position
        glm::normalize(glm::vec3(-1.0f, -0.6f, -1.0f)),      // direction
        glm::vec3(1.0f, 0.0f, 0.0f),                         // color = red
        12.5f,                                               // inner cut angle in degrees
        20.0f                                                // outer cut angle in degrees
    );
    searchLight->intensity = 2.0f; // nastavit sílu (pokud Light má public member intensity)
    scene->addLight(searchLight);


    // 5) Fireflies (point lights s malým dosahem) - už tam máte funkci, voláme ji
    addFireflies(20);

    // optionally: menší ukázkové bodové světlo v blízkosti postavy
    glm::vec3 lanternPos(2.0f, 1.0f, 0.5f);

    // 1) samotné světlo (green)
    PointLight* lantern = new PointLight(lanternPos, glm::vec3(0.0f, 1.0f, 0.0f), 1.0f, 0.14f, 0.07f);
    lantern->intensity = 1.0f;
    scene->addLight(lantern);

    // 2) vizuální marker pro světlo (malá koule s barevnou texturou)
    DrawableObject* lanternVis = new DrawableObject(ModelType::Sphere, ShaderType::Phong, TextureType::Green);
    {
        Transform lt;
        lt.addTransform(std::make_shared<Scale>(glm::vec3(0.06f)));   // velikost markeru
        lt.addTransform(std::make_shared<Translation>(lanternPos));   // umístění přesně na světlo
        lanternVis->setTransform(lt);
        scene->addObject(lanternVis);
    }

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

    DrawableObject* ground = new DrawableObject(ModelType::Plain, ShaderType::Textured, TextureType::Teren);
    {
        Transform tg;
        tg.addTransform(std::make_shared<Scale>(glm::vec3(50.0f, 1.0f, 50.0f)));
        tg.addTransform(std::make_shared<Translation>(glm::vec3(0.0f, 0.0f, 0.0f)));
        ground->setTransform(tg);
        scene->addObject(ground);
    }

    PointLight* center = new PointLight(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), 6.0f);
    scene->addLight(center);

    //// pohybující se model Formula1
    //DrawableObject* obj = new DrawableObject(ModelType::Formula1, ShaderType::Phong);

    //Transform t;
    //t.addTransform(std::make_shared<Scale>(glm::vec3(0.05f)));

    //// nové, plynulejší kontrolní body
    //std::vector<glm::vec3> ctrl = {
    //    { -4.0f, 0.0f, -1.0f },
    //    { -1.5f, 0.0f,  1.5f },
    //    {  1.5f, 0.0f,  1.5f },
    //    {  4.0f, 0.0f, -1.0f }
    //};

    //// smoother curve speed & loop
    //t.addTransform(std::make_shared<Bezier>(ctrl, 0.18f, true));

    //obj->setTransform(t);
    //scene->addObject(obj);

    //// vizualizace křivky - více vzorků = hladší křivka
    //const int samples = 80;
    //for (int i = 0; i <= samples; ++i) {
    //    float s = i / float(samples);
    //    glm::vec3 p = Bezier::evalCubic(ctrl[0], ctrl[1], ctrl[2], ctrl[3], s);

    //    DrawableObject* marker = new DrawableObject(ModelType::Sphere, ShaderType::Phong);
    //    Transform mt;
    //    mt.addTransform(std::make_shared<Translation>(p));
    //    mt.addTransform(std::make_shared<Scale>(glm::vec3(0.05f)));
    //    marker->setTransform(mt);
    //    scene->addObject(marker);
    //}

    return scene;
}


Scene* SceneFactory::createScene1() {
    Scene* scene = new Scene();

    DrawableObject* obj = new DrawableObject(ModelType::Sphere, ShaderType::Basic);

    Transform t;
    t.addTransform(std::make_shared<Translation>(glm::vec3(0, 0, 0)));
    obj->setTransform(t);
	obj->addTexture(TextureManager::instance().get(TextureType::Earth));

    scene->addObject(obj);
    return scene;
}

Scene* SceneFactory::createScene2() {
    Scene* scene = new Scene();

    // objekt s wooden fence
    DrawableObject* fence = new DrawableObject(ModelType::Login, ShaderType::Textured);
    Transform tf1;
    tf1.addTransform(std::make_shared<Scale>(glm::vec3(0.15f)));
    tf1.addTransform(std::make_shared<Translation>(glm::vec3(-0.8f, -1.5f, 0.0f)));
    fence->setTransform(tf1);
    fence->addTexture(TextureManager::instance().get(TextureType::WoodenFence));
    fence->addTexture(TextureManager::instance().get(TextureType::Grass));
    scene->addObject(fence);


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
