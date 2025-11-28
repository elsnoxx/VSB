#include "callbacks.h"
#include <stdio.h>

static double lastMouseX = 0.0;
static double lastMouseY = 0.0;
static bool firstMouse = true;

// --- ERROR ---------------------------------------------------------------
void callbackError(int t_error, const char* t_description) {
    fputs(t_description, stderr);
}

// --- WINDOW EVENTS -------------------------------------------------------
void callbackWindowFocus(GLFWwindow* t_window, int t_focused) {
    printf("<callback> window focus\n");
}

void callbackWindowIconify(GLFWwindow* t_window, int t_iconified) {
    printf("<callback> window iconify\n");
}

void callbackWindowSize(GLFWwindow* window, int width, int height) {
    printf("resize %d, %d \n", width, height);
    glViewport(0, 0, width, height);
}

// --- KEYBOARD ------------------------------------------------------------
void callbackKey(GLFWwindow* t_window, int t_key, int t_scancode, int t_action, int t_mods) {

    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(t_window));
    Camera* cam = app->getCurrentCamera();

    // switching scenes:
    if (t_action == GLFW_PRESS) {
        if (t_key >= GLFW_KEY_1 && t_key <= GLFW_KEY_9) {
            int index = t_key - GLFW_KEY_1;
            app->switchScene(index);
        }
        if (t_key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(t_window, GL_TRUE);
    }

    // ---- CAMERA MOVEMENT (WSAD) ----
    if (t_action == GLFW_PRESS || t_action == GLFW_REPEAT) {
        if (t_key == GLFW_KEY_W) cam->moveForward();
        if (t_key == GLFW_KEY_S) cam->moveBackward();
        if (t_key == GLFW_KEY_A) cam->moveLeft();
        if (t_key == GLFW_KEY_D) cam->moveRight();

    }
}

// --- MOUSE BUTTON --------------------------------------------------------
void callbackButton(GLFWwindow* t_window, int t_button, int t_action, int t_mode) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(t_window));
    if (!app) return;

    if (t_button == GLFW_MOUSE_BUTTON_RIGHT && t_action == GLFW_PRESS) {
        firstMouse = true; // resetujeme první klik
    }
}

// --- MOUSE MOVEMENT ------------------------------------------------------
void callbackCursor(GLFWwindow* t_window, double t_x, double t_y) {

    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(t_window));
    if (!app || !app->camera) return;

    // pravé tlačítko musí být drženo → pohyb kamery
    if (glfwGetMouseButton(t_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {

        if (firstMouse) {
            lastMouseX = t_x;
            lastMouseY = t_y;
            firstMouse = false;
        }

        double dx = t_x - lastMouseX;
        double dy = t_y - lastMouseY;

        app->camera->addMouseDelta((float)dx, (float)dy);

        lastMouseX = t_x;
        lastMouseY = t_y;
    }
}
