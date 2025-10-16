#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Forward declarations
class ShaderProgram;
class Model;
class DrawableObject;

class Application {
private:
    GLFWwindow* window;
    int width, height;
    
    // OOP objekty místo raw OpenGL handles
    ShaderProgram* shaderProgram;
    Model* model;
    DrawableObject* drawableObject;
    
public:
    Application() : window(nullptr), shaderProgram(nullptr), 
                   model(nullptr), drawableObject(nullptr) {}
    
    void initialization();
    void createShaders();
    void createModels();
    void createDrawableObjects();
    void run();
    void printVersionInfo();
    void updateViewport();
};