#pragma once
#include "Application.h"

void Application::printVersionInfo() {
    // version info
    printf("--------------------------------------------------------------------------------\n");
    printf("vendor: %s\n", glGetString(GL_VENDOR));
    printf("renderer: %s\n", glGetString(GL_RENDERER));
    printf("OpenGL version: %s\n", glGetString(GL_VERSION));
    printf("GLSL version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    printf("using GLEW %s\n", glewGetString(GLEW_VERSION));

    int major, minor, revision;
    glfwGetVersion(&major, &minor, &revision);
    printf("using GLFW %i.%i.%i\n", major, minor, revision);
    printf("--------------------------------------------------------------------------------\n");
}

void Application::updateViewport(double x, double y) {
    glfwGetFramebufferSize(window, &width, &height);
    float ratio = width / (float)height;
    glViewport(0, 0, width, height);

    Scene* cur = screenManager.getCurrentScene();
    if (cur) {
        cur->getCamera()->updateScreenSize(width, height);
        cur->bindCameraAndLightToUsedShaders();
        input.onMouseMove(x, y);
    }
}

void Application::switchScene(int index) {
    screenManager.switchTo(index);
}

void Application::updateFOV(float radians) {
    screenManager.changeFOV(radians);
}


void Application::run() {
    glEnable(GL_DEPTH_TEST);
    while (!glfwWindowShouldClose(window)) {
        double current = glfwGetTime();
        static double last = current;
        float dt = float(current - last);
        last = current;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        screenManager.update(dt, input);
        screenManager.draw();

        glfwPollEvents();
        glfwSwapBuffers(window);
    }
}

void Application::handleMouseClick(double x, double y, int button) {
    Scene* cur = screenManager.getCurrentScene();
    if (!cur) return;

    glm::vec3 worldPos;
    int picked = cur->pickAtCursor(x, y, &worldPos);

    if (picked >= 0) {
        printf("[App] picked object index %d at world [%f,%f,%f]\n", picked, worldPos.x, worldPos.y, worldPos.z);
        // example: left click = plant a tree at clicked world position
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            cur->plantObjectAtWorldPos(worldPos, ModelType::Tree, ShaderType::Phong);
        }
        // example: right click = select object
        else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            bool ok = cur->removeObjectAt(picked);
            printf("[App] removeObjectAt(%d) -> %s\n", picked, ok ? "ok" : "failed");
        }
    }
    else {
        printf("[App] clicked empty space\n");
        // if left click on empty space, you can unproject and plant on plane y=0:
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            // if depth read was 1.0 (far), compute ray and intersect y=0 plane
            // reuse cur->pickAtCursor without hit: compute ray manually
            int fbw, fbh; glfwGetFramebufferSize(glfwGetCurrentContext(), &fbw, &fbh);
            glm::vec3 nearP((float)x, (float)(fbh - y), 0.0f);
            glm::vec3 farP((float)x, (float)(fbh - y), 1.0f);
            glm::mat4 view = cur->getCamera()->getViewMatrix();
            glm::mat4 proj = cur->getCamera()->getProjectionMatrix();
            glm::vec4 vp(0, 0, (float)fbw, (float)fbh);
            glm::vec3 n = glm::unProject(nearP, view, proj, vp);
            glm::vec3 f = glm::unProject(farP, view, proj, vp);
            glm::vec3 dir = glm::normalize(f - n);
            if (fabs(dir.y) > 1e-6f) {
                float t = -n.y / dir.y;
                if (t > 0.0f) {
                    glm::vec3 planePos = n + dir * t;
                    cur->plantObjectAtWorldPos(planePos, ModelType::Tree, ShaderType::Phong);
                }
            }
        }
    }
}

void Application::initialization() {
    glfwSetErrorCallback(callbackError);
    if (!glfwInit()) {
        fprintf(stderr, "ERROR: could not start GLFW3\n");
        exit(EXIT_FAILURE);
    }

    // Voliteln�: nastaven� verze OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(Config::WindowWidth, Config::WindowHeight, Config::Title, NULL, NULL);
    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwSetWindowUserPointer(window, this);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    glewInit();

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
 

	// screen init
    screenManager.init();

    // Nastaven� callback�
    glfwSetKeyCallback(window, callbackKey);
    glfwSetCursorPosCallback(window, callbackCursor);
    glfwSetMouseButtonCallback(window, callbackButton);
    glfwSetWindowFocusCallback(window, callbackWindowFocus);
    glfwSetWindowIconifyCallback(window, callbackWindowIconify);
    glfwSetWindowSizeCallback(window, callbackWindowSize);

    printVersionInfo();
    updateViewport(0,0);
}