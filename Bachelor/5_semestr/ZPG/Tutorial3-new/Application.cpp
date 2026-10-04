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

    // Ensure input is initialized for the newly switched scene so the camera
    // doesn't require an extra click or mouse move to start responding.
    // Process pending events and focus window so cursor queries are reliable.
    if (window) {
        glfwPollEvents();
        glfwFocusWindow(window);
    }

    // Sample current cursor position and prime the InputManager's lastMousePos.
    double cx = 0.0, cy = 0.0;
    if (window) {
        glfwGetCursorPos(window, &cx, &cy);
        // Ensure GLFW internal cursor position is set consistently
        glfwSetCursorPos(window, cx, cy);
    }

    // Reset internal input state (clears accumulators) and set the last mouse
    // position to current cursor so next movement produces a proper delta.
    input.resetState();
    input.onMouseMove(cx, cy);

    // Update viewport/projection and shader bindings for the new scene now.
    updateViewport(cx, cy);
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

void Application::initialization() {
    glfwSetErrorCallback(callbackError);
    if (!glfwInit()) {
        fprintf(stderr, "ERROR: could not start GLFW3\n");
        exit(EXIT_FAILURE);
    }

    // Optional: set OpenGL version
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

    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
 

	// screen init
    screenManager.bindInput(&input);
    screenManager.init();

    // Set callbacks
    glfwSetKeyCallback(window, callbackKey);
    glfwSetCursorPosCallback(window, callbackCursor);
    glfwSetMouseButtonCallback(window, callbackButton);
    glfwSetWindowFocusCallback(window, callbackWindowFocus);
    glfwSetWindowIconifyCallback(window, callbackWindowIconify);
    glfwSetWindowSizeCallback(window, callbackWindowSize);
    printVersionInfo();

    // Make sure events are processed and the window is focused before priming
    // input state so the first scene reacts without an extra click.
    glfwPollEvents();
    if (window) glfwFocusWindow(window);

    // Prime input with current cursor position
    double cx = 0.0, cy = 0.0;
    if (window) {
        glfwGetCursorPos(window, &cx, &cy);
        glfwSetCursorPos(window, cx, cy);
    }
    input.resetState();
    input.onMouseMove(cx, cy);
    updateViewport(cx, cy);
}