#include "SceneFactory.h"
#include "../ModelObject/ModelManager.h"
#include "../Light/PointLight.h"
#include "../Transform/Bezier.h"
#include "../Light/DirectionalLight.h"
#include "../Material/MaterialType.h"


std::vector<Scene*> SceneFactory::createAllScenes() {
    return {
        // Tutorial 2
        //createScene1(),
        //createScene2(),
        //createScene3(),
        //createScene4(),

        // Tutorial 3
        createSceneSphereLights(),

        createSceneDifferentModes(),

        createSceneSolarSystem(),

        createSceneFormula1(),

        createForestScene(),
        
    };
}

Scene* SceneFactory::createSceneSolarSystem()
{
    Scene* scene = new Scene();

    // =========================
    //  SUN (root object)
    // =========================
    // menší viditelná koule pro Slunce, vykreslená jako unlit/textured (světélko zůstane z PointLight)
    auto sunNode = std::make_shared<TransformNode>();
    sunNode->addTransform(std::make_shared<Scale>(glm::vec3(1.0f)));

    DrawableObject* sun = new DrawableObject(ModelType::Sphere, ShaderType::Phong, TextureType::Sun);
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

    DrawableObject* earth = new DrawableObject(ModelType::Earth, ShaderType::Phong, TextureType::Earth);
    earth->setTransformNode(earthNode);
    scene->addObject(earth);


    // =========================
    //  MOON (child of Earth)
    // =========================
    auto moonNode = std::make_shared<TransformNode>();
    // Kompenzujeme parent scale (earth scale = 0.10f), takže posun i scale zvětšíme ~10×
    moonNode->addTransform(std::make_shared<Rotation>([]() { return glfwGetTime() * 25.0f; }, glm::vec3(0, 1, 0)));
    moonNode->addTransform(std::make_shared<Translation>(glm::vec3(6.0f, 0, 0))); // bylo 0.6f
    moonNode->addTransform(std::make_shared<Scale>(glm::vec3(0.3f)));             // bylo 0.03f

    earthNode->addChild(moonNode); // Attach moon to earth

    // Prefer dedicated Moon model when available, otherwise use Sphere + Moon texture
    DrawableObject* moon = nullptr;
    if (ModelManager::instance().get(ModelType::Moon)) {
        moon = new DrawableObject(ModelType::Moon, ShaderType::Phong, TextureType::Moon);
    }
    else {
        moon = new DrawableObject(ModelType::Sphere, ShaderType::Phong, TextureType::Moon);
    }
    moon->setTransformNode(moonNode);
    scene->addObject(moon);


    // =========================
    //  OTHER PLANETS (simple)
    // =========================
    struct PlanetDef { ModelType model; TextureType texture; float orbitRadius, orbitSpeed, selfRotate, scale; };

    std::vector<PlanetDef> planets = {
        { ModelType::Mercury,   TextureType::Mercury, 1.5f, 15.0f, 20.0f, 0.05f },
        { ModelType::Venus,     TextureType::Venus,   2.0f, 10.0f, 15.0f, 0.09f },
        { ModelType::Mars,      TextureType::Mars,    3.2f,  7.0f, 18.0f, 0.07f },
        { ModelType::Jupiter,   TextureType::Jupiter, 5.0f,  4.0f, 25.0f, 0.05f },
        { ModelType::Sphere,    TextureType::Saturn,  6.0f,  3.0f, 22.0f, 0.25f },
        { ModelType::Sphere,    TextureType::Uranus, 7.5f,  2.0f, 20.0f, 0.18f },
        { ModelType::Sphere,    TextureType::Neptune, 8.7f,  1.7f, 18.0f, 0.17f },
        { ModelType::Sphere,    TextureType::Pluto,10.0f,  1.0f, 13.0f, 0.03f }
    };

    for (auto& p : planets)
    {
        auto node = std::make_shared<TransformNode>();
        node->addTransform(std::make_shared<Rotation>([p]() { return glfwGetTime() * p.orbitSpeed; }, glm::vec3(0, 1, 0)));
        node->addTransform(std::make_shared<Translation>(glm::vec3(p.orbitRadius, 0, 0)));
        node->addTransform(std::make_shared<Rotation>([p]() { return glfwGetTime() * p.selfRotate; }, glm::vec3(0, 1, 0)));
        node->addTransform(std::make_shared<Scale>(glm::vec3(p.scale)));

        sunNode->addChild(node);

        DrawableObject* planet = new DrawableObject(p.model, ShaderType::Phong, p.texture);
        planet->setTransformNode(node);
        scene->addObject(planet);
    }

    return scene;
}





Scene* SceneFactory::createForestScene() {
    Scene* scene = new Scene();

    // create main objects directly via ModelType
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


    // random generator
    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distPos(-40.0f, 40.0f);
    std::uniform_real_distribution<float> distTreeScale(0.8f, 1.6f);
    std::uniform_real_distribution<float> distBushScale(0.3f, 0.9f);
    std::uniform_real_distribution<float> distRot(0.0f, 360.0f);

    // placeObjects now accepts ModelType instead of Model* and checks model availability in ModelManager
    auto placeObjects = [&](ModelType modelType, int count, bool isTree) {
        // check that the model exists in the manager (lazy-loaded)
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

            // create object via ModelType - DrawableObject will obtain the model from ModelManager in its constructor
            DrawableObject* obj = new DrawableObject(modelType, ShaderType::Phong);
			obj->addTexture(TextureManager::instance().get(TextureType::WoodenFence));
			obj->addTexture(TextureManager::instance().get(TextureType::Teren));
            int variant = (rng() & 1); // rychlý bitový výběr (0/1)
            obj->setActiveTextureIndex(variant);
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

    // 50 trees and 50 bushes (use enums instead of raw pointers)
    placeObjects(ModelType::Tree, 50, true);
    placeObjects(ModelType::Bushes, 50, false);

    auto addFireflies = [&](int count) {
        // check availability of the sphere model
        if (!ModelManager::instance().get(ModelType::Sphere)) return;

        std::uniform_real_distribution<float> distHeight(1.0f, 5.0f);
        std::uniform_real_distribution<float> distRange(-30.0f, 30.0f);
        std::uniform_real_distribution<float> distRadius(0.2f, 2.0f);
        std::uniform_real_distribution<float> distSpeed(0.5f, 2.5f);
        std::uniform_real_distribution<float> distPhase(0.0f, 6.28318530718f);

        for (int i = 0; i < count; i++) {
            glm::vec3 basePos(distRange(rng), distHeight(rng), distRange(rng));

            // visual glow sphere - small with a simple colored texture
            DrawableObject* firefly = new DrawableObject(ModelType::Sphere, ShaderType::Phong, TextureType::Yellow);
            {
                // random movement parameters
                float radius = distRadius(rng);
                float speed = distSpeed(rng);
                float phase = distPhase(rng);
                float bobAmp = 0.25f + (radius * 0.1f);

                Transform tf;
                // very small sphere
				tf.addTransform(std::make_shared<Translation>(basePos));
                tf.addTransform(std::make_shared<Scale>(glm::vec3(0.03f)));
				firefly->setMaterial(MaterialType::Emissive);

                firefly->setTransform(tf);
                scene->addObject(firefly);
            }

            // light -> smaller range, lower intensity (quick falloff)
            PointLight* fl = new PointLight(
                basePos,
                glm::vec3(1.0f, 0.95f, 0.6f),  // color of the firefly
                1.0f,   // constant
                1.5f,   // linear (higher value = faster attenuation)
                2.5f    // quadratic (higher = even faster attenuation)
            );

            fl->intensity = 0.8f;      // overall weaker light
            scene->addLight(fl);
        }
        };

    // example: if SpotLight has signature (pos, dir, innerCos, outerCos, color, constant, linear, quadratic)
    SpotLight* searchLight = new SpotLight(
        glm::vec3(10.0f, 8.0f, 10.0f),                       // position
        glm::normalize(glm::vec3(-1.0f, -0.6f, -1.0f)),      // direction
        glm::vec3(1.0f, 0.0f, 0.0f),                         // color = red
        12.5f,                                               // inner cut angle in degrees
        20.0f                                                // outer cut angle in degrees
    );
    searchLight->intensity = 2.0f; // set intensity (if Light has a public member 'intensity')
    scene->addLight(searchLight);


    // 5) Fireflies (point lights with small range) - there is already a helper function, call it
    addFireflies(5);

    // optionally: small example point light near the character
    glm::vec3 lanternPos(2.0f, 1.0f, 0.5f);

    // 1) the light itself (green)
    PointLight* lantern = new PointLight(lanternPos, glm::vec3(0.0f, 1.0f, 0.0f), 1.0f, 0.14f, 0.07f);
    lantern->intensity = 1.0f;
    scene->addLight(lantern);

    // 2) visual marker for the light (small sphere with colored texture)
    DrawableObject* lanternVis = new DrawableObject(ModelType::Sphere, ShaderType::Phong, TextureType::Green);
    {
        Transform lt;
        lt.addTransform(std::make_shared<Scale>(glm::vec3(0.06f)));   // marker size
        lt.addTransform(std::make_shared<Translation>(lanternPos));   // position exactly at the light
        lanternVis->setTransform(lt);
        scene->addObject(lanternVis);
    }

    // --- Reflector (spotlight) for Fiona ---
    // Fiona's translation in this scene is (2.0f, 0.0f, 0.0f)
    glm::vec3 fionaPos(2.0f, 0.0f, 0.0f);

    // place the reflector slightly above and in front of Fiona
    glm::vec3 reflectorPos = fionaPos + glm::vec3(0.0f, 3.5f, 0.5f);
    glm::vec3 reflectorDir = glm::normalize(fionaPos - reflectorPos);

    // Create spotlight aimed at Fiona:
    // constructor: (pos, dir, color, innerAngleDeg, outerAngleDeg)
    SpotLight* fionaReflector = new SpotLight(
        reflectorPos,
        reflectorDir,
        glm::vec3(1.0f, 0.0f, 0.0f), // warm white color
        14.0f,  // inner cone angle (deg)
        25.0f   // outer cone angle (deg)
    );
    fionaReflector->intensity = 2.2f; // brighten a bit
    scene->addLight(fionaReflector);

    // optional: small visual marker for the reflector (small sphere)
    DrawableObject* reflectorVis = new DrawableObject(ModelType::Sphere, ShaderType::Phong, TextureType::Yellow);
    {
        Transform tr;
        tr.addTransform(std::make_shared<Scale>(glm::vec3(0.05f)));
        tr.addTransform(std::make_shared<Translation>(reflectorPos));
        reflectorVis->setTransform(tr);
        scene->addObject(reflectorVis);
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
        DrawableObject* obj = new DrawableObject(ModelType::Sphere, ShaderType::Phong, TextureType::Red);

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

    // different shaders for the individual spheres
    std::vector<ShaderType> shaders = {
        ShaderType::Phong,
        ShaderType::Lambert,
        ShaderType::Basic,
        ShaderType::Textured
    };

    for (size_t i = 0; i < positions.size(); ++i) {
        ShaderType st = shaders[i % shaders.size()];
        DrawableObject* obj = new DrawableObject(ModelType::Sphere, st, TextureType::Red);

        Transform t;
        t.addTransform(std::make_shared<Scale>(glm::vec3(0.8f)));
        t.addTransform(std::make_shared<Translation>(positions[i]));
        obj->setTransform(t);

        scene->addObject(obj);
    }

    // a single central point light in the middle (world space)
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

    // moving Formula1 model
    DrawableObject* obj = new DrawableObject(ModelType::Formula1, ShaderType::Phong, TextureType::WoodenFence);
    {
        //obj->addTexture(TextureManager::instance().get(TextureType::WoodenFence));
        obj->setMaterial(MaterialType::Metal);

        Transform t;
        // nejdřív Bezier s orientací, pak scale
        std::vector<glm::vec3> segment = {
            { -40.0f, 0.0f, -10.0f },
            { -10.5f, 0.0f,  10.5f },
            {  10.5f, 0.0f,  10.5f },
            {  40.0f, 0.0f, -10.0f }
        };
        t.addTransform(std::make_shared<Bezier>(segment, 6.0f, true /*loop*/, true /*orient*/, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, -0.25f, 0.0f)));
        t.addTransform(std::make_shared<Scale>(glm::vec3(0.05f)));

        obj->setTransform(t);
        scene->addObject(obj);
    }


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

