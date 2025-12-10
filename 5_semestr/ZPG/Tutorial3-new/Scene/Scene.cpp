#include "Scene.h"
#include "../Light/PointLight.h"
#include "../Light/SpotLight.h"
#include "../Light/DirectionalLight.h"
#include <unordered_set>
#include "../Transform/Scale.h"
#include "../Transform/Translation.h"
#include "../Transform/Bezier.h"

Scene::Scene() {
    camera = new Camera(glm::vec3(0.f, 1.f, 5.f));
    lightManager = new LightManager();

    headLight = new HeadLight(camera);
    headLight->intensity = 5.0f;
	headLight->cutOff = glm::cos(glm::radians(12.5f));
    lightManager->addLight(headLight);
}

Scene::~Scene() {
    delete headLight;
    delete lightManager;
    delete camera;
}

Camera* Scene::getCamera() {
    return camera;
}

void Scene::addObject(DrawableObject* obj) {
    objects.push_back(obj);

    ShaderProgram* shader = obj->getShader();

    if (shader) {
        shader->attachCamera(camera);
    }
}

Light* Scene::addLight(Light* light) {
    if (!light || !lightManager) return nullptr;
    return lightManager->addLight(light);
}

void Scene::draw() {
    bindCameraAndLightToUsedShaders();

    // enable stencil for picking: we will write object IDs into stencil buffer
    glEnable(GL_STENCIL_TEST);
    glStencilMask(0xFF); // enable writing to all bits
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

    unsigned int id = 1;
    for (auto& obj : objects) {
        unsigned int writeId = (id <= 255) ? id : 255;
        obj->setID(writeId);
        glStencilFunc(GL_ALWAYS, writeId, 0xFF);
        obj->draw();
        ++id;
    }

    glStencilMask(0x00);
    glDisable(GL_STENCIL_TEST);
}

int Scene::pickAtCursor(double x, double y, glm::vec3* outWorld) {
    // get framebuffer size and convert mouse y
    GLFWwindow* win = glfwGetCurrentContext();
    if (!win) return -1;
    int fbw, fbh;
    glfwGetFramebufferSize(win, &fbw, &fbh);
    GLint ix = static_cast<GLint>(x);
    GLint iy = static_cast<GLint>(y);
    GLint newy = fbh - iy;

    // read stencil
    GLuint stencilIndex = 0;
    glReadPixels(ix, newy, 1, 1, GL_STENCIL_INDEX, GL_UNSIGNED_INT, &stencilIndex);

    // read depth
    GLfloat depth = 1.0f;
    glReadPixels(ix, newy, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);

    // optional: read color (not required)
    GLubyte color[4] = { 0,0,0,0 };
    glReadPixels(ix, newy, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, color);

    printf("[Picking] cursor %d,%d -> stencil %u depth %f color %02x%02x%02x%02x\n",
        ix, iy, stencilIndex, depth, color[0], color[1], color[2], color[3]);

    if (stencilIndex == 0) {
        selectedIndex = -1;
        return -1;
    }

    // compute world position via unProject
    glm::vec3 screenPos((float)ix, (float)newy, depth);
    glm::mat4 view = camera->getViewMatrix();           // adjust if different name
    glm::mat4 proj = camera->getProjectionMatrix();     // adjust if different name
    glm::vec4 viewport(0.f, 0.f, (float)fbw, (float)fbh);
    glm::vec3 world = glm::unProject(screenPos, view, proj, viewport);

    if (outWorld) *outWorld = world;

    // scene-local index (ids were written as 1..)
    selectedIndex = static_cast<int>(stencilIndex) - 1;
    return selectedIndex;
}

void Scene::plantObjectAtWorldPos(const glm::vec3& worldPos, ModelType type, ShaderType shader) {
    DrawableObject* obj = new DrawableObject(type, shader);
    Transform t;
    t.addTransform(std::make_shared<Scale>(glm::vec3(1.0f)));
    t.addTransform(std::make_shared<Translation>(worldPos));
    obj->setTransform(t);
	obj->addTexture(TextureManager::instance().get(TextureType::WoodenFence));
	obj->addTexture(TextureManager::instance().get(TextureType::Teren));
    this->addObject(obj);
}

bool Scene::removeObjectAt(int idx) {
    if (idx < 0 || idx >= static_cast<int>(objects.size())) return false;
    DrawableObject* obj = objects[idx];
    // erase from vector and delete object
    objects.erase(objects.begin() + idx);
    delete obj;
    // reset selection if needed
    if (selectedIndex == idx) selectedIndex = -1;
    else if (selectedIndex > idx) --selectedIndex; // shift selection index
    return true;
}

void Scene::buildBezierFromControlPoints(float speed, bool loop)
{
    // vytvoří Bezier segmenty z každých 4 bodů (nepřekrývající se)
    size_t count = controlPoints.size();
    if (count < 4) return;

    for (size_t i = 0; i + 3 < count; i += 4) {
        std::vector<glm::vec3> segment = {
            controlPoints[i + 0],
            controlPoints[i + 1],
            controlPoints[i + 2],
            controlPoints[i + 3]
        };
        std:cout << "[Scene] Building Bezier segment from control points: "
                  << "P0(" << segment[0].x << "," << segment[0].y << "," << segment[0].z << "), "
                  << "P1(" << segment[1].x << "," << segment[1].y << "," << segment[1].z << "), "
                  << "P2(" << segment[2].x << "," << segment[2].y << "," << segment[2].z << "), "
			<< "P3(" << segment[3].x << "," << segment[3].y << "," << segment[3].z << ")\n";
        // vytvoříme nový objekt, který pojede po tomto segmentu

        const int samples = 24;
        for (int s = 0; s <= samples; ++s) {
            float tt = s / (float)samples;
            glm::vec3 p = Bezier::evalCubic(segment[0], segment[1], segment[2], segment[3], tt);
            DrawableObject* marker = new DrawableObject(ModelType::Sphere, ShaderType::Phong, TextureType::Sun);
            Transform mt;
            mt.addTransform(std::make_shared<Translation>(p));
            mt.addTransform(std::make_shared<Scale>(glm::vec3(0.04f)));
            marker->setTransform(mt);
            addObject(marker);
        }

        DrawableObject* mover = new DrawableObject(ModelType::Formula1, ShaderType::Phong, TextureType::WoodenFence);
        Transform t;
        // nejprve Bezier (world-space translation/orientace), pak škálování modelu
        // použij zde duration (v sekundách) místo "speed" — např. 6.0f pro plynulý průjezd
        t.addTransform(std::make_shared<Bezier>(segment, 6.0f, true /*loop*/, true /*orient*/, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, -0.25f, 0.0f)));
        t.addTransform(std::make_shared<Scale>(glm::vec3(0.05f))); // případně Scale(1.0f) během ladění
       
        mover->setTransform(t);
        addObject(mover);
    }

    // po zpracování smažeme body (pokud chcete je nechat, odstraňte tento clear)
    controlPoints.erase(controlPoints.begin(), controlPoints.begin() + (count / 4) * 4);
}

void Scene::addControlPoint(const glm::vec3& p)
{
    controlPoints.push_back(p);

    // vytvoříme malý marker pro vizualizaci kliknutí
    DrawableObject* marker = new DrawableObject(ModelType::Sphere, ShaderType::Phong);
    Transform mt;
    mt.addTransform(std::make_shared<Translation>(p));
    mt.addTransform(std::make_shared<Scale>(glm::vec3(0.06f)));
    marker->setTransform(mt);
    addObject(marker);
}

void Scene::clearControlPoints()
{
    controlPoints.clear();
}

// vrátí referenci (pouze čtení)
const std::vector<glm::vec3>& Scene::getControlPoints() const
{
    return controlPoints;
}

void Scene::update(float dt, InputManager& input)
{
    float camSpeed = 5.0f * dt;
    auto cam = getCamera();

    // WSAD pohyb
    if (input.isKeyDown(GLFW_KEY_W)) { 
		std::cout << "Moving forward W\n";
        cam->forward(camSpeed); 
    }
    if (input.isKeyDown(GLFW_KEY_S)) { 
		std::cout << "Moving backward S\n";
        cam->backward(camSpeed); 
    }
    if (input.isKeyDown(GLFW_KEY_A)) { 
		std::cout << "Moving left A\n";
        cam->left(camSpeed); 
    }
    if (input.isKeyDown(GLFW_KEY_D)) {
		std::cout << "Moving right D\n";
        cam->right(camSpeed); 
    }

    if (input.isKeyPressed(GLFW_KEY_F)) {
	    std:cout << "Toggling headlight F\n";
        switchHeadLight();
    }

    // rmouse rotate
    glm::vec2 delta = input.getMouseDeltaAndReset(dt);
    if (delta.x != 0.0f || delta.y != 0.0f) {
        cam->updateOrientation(delta, dt);
    }
}

void Scene::bindCameraAndLightToUsedShaders()
{
    if (!lightManager) return;

    const int MAX_SHADER_LIGHTS = 16;
    int totalLights = lightManager->getLightsAmount();
    int n = std::min(totalLights, MAX_SHADER_LIGHTS);

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> colors;
    std::vector<float> intensities;
    std::vector<glm::vec3> directions;
    std::vector<float> cutOffs;
    std::vector<float> outerCutOffs;
    std::vector<int> types;
    std::vector<int> isOn;

    positions.reserve(n); colors.reserve(n); intensities.reserve(n);
    directions.reserve(n); cutOffs.reserve(n); outerCutOffs.reserve(n); types.reserve(n);

    for (int i = 0; i < n; ++i) {
        Light* L = lightManager->getLight(i);
        if (!L) {
            positions.push_back(glm::vec3(0.0f));
            colors.push_back(glm::vec3(0.0f));
            intensities.push_back(0.0f);
            directions.push_back(glm::vec3(0.0f));
            cutOffs.push_back(0.0f);
            outerCutOffs.push_back(0.0f);
            types.push_back(1);
            continue;
        }

        colors.push_back(L->color);
        isOn.push_back(L->isOn ? 1 : 0);
        // respect on/off
        intensities.push_back(L->isOn ? L->intensity : 0.0f);
        types.push_back(static_cast<int>(L->type));

        if (auto pl = dynamic_cast<PointLight*>(L)) {
            positions.push_back(pl->position);
            directions.push_back(glm::vec3(0.0f));
            cutOffs.push_back(0.0f);
            outerCutOffs.push_back(0.0f);
        }
        else if (auto sl = dynamic_cast<SpotLight*>(L)) {
            positions.push_back(sl->position);
            directions.push_back(glm::normalize(sl->direction));
            cutOffs.push_back(sl->cutOff);
            outerCutOffs.push_back(sl->outerCutOff);
        }
        else if (auto dl = dynamic_cast<DirectionalLight*>(L)) {
            positions.push_back(-dl->direction * 10000.0f);
            directions.push_back(glm::normalize(dl->direction));
            cutOffs.push_back(0.0f);
            outerCutOffs.push_back(0.0f);
        }
        else {
            positions.push_back(glm::vec3(0.0f));
            directions.push_back(glm::vec3(0.0f));
            cutOffs.push_back(0.0f);
            outerCutOffs.push_back(0.0f);
        }
    }

    std::unordered_set<ShaderProgram*> processed;
    for (auto* obj : objects)
    {
        ShaderProgram* shader = obj->getShader();
        if (!shader) continue;
        if (processed.count(shader)) continue;
        processed.insert(shader);

        shader->use();
        shader->setUniform("numLights", n);

        for (int i = 0; i < n; ++i) {
            shader->setUniform(("lightTypes[" + std::to_string(i) + "]").c_str(), types[i]);
            shader->setUniform(("lightPositions[" + std::to_string(i) + "]").c_str(), positions[i]);
            shader->setUniform(("lightDirections[" + std::to_string(i) + "]").c_str(), directions[i]);
            shader->setUniform(("lightColors[" + std::to_string(i) + "]").c_str(), colors[i]);
            shader->setUniform(("lightIntensities[" + std::to_string(i) + "]").c_str(), intensities[i]);
            shader->setUniform(("lightCutOffs[" + std::to_string(i) + "]").c_str(), cutOffs[i]);
            shader->setUniform(("lightOuterCutOffs[" + std::to_string(i) + "]").c_str(), outerCutOffs[i]);
            shader->setUniform(("lightIsOn[" + std::to_string(i) + "]").c_str(), isOn[i]);
        }

        shader->setUniform("viewPosition", camera->getPosition());
        shader->setUniform("shininess", 64.0f);
        shader->setUniform("ambientStrength", 0.15f);

        glUseProgram(0);
    }
}

void Scene::switchHeadLight() {
    if (!headLight) return;

    headLight->isOn = !headLight->isOn;

    if (headLight->isOn) {
        std::cout << "Headlight turned ON " << headLight->isOn << "\n";
    }
    else {
        std::cout << "Headlight turned OFF " << headLight->isOn << "\n";
    }

    headLight->notify(ObservableSubjects::SLight);
}