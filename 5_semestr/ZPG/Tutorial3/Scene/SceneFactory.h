#pragma once
//Include GLEW
#include <GL/glew.h>
//Include GLFW
#include <GLFW/glfw3.h> 

#include <vector>
#include "Scene.h"
#include "../Shader/ShaderProgram.h"
#include "../Model.h"

// sem include modelových dat
#include "../Models/sphere.h"
#include "../Models/tree.h"
#include "../Models/bushes.h"
#include "../Models/gift.h"
#include "../Models/suzi_flat.h"
#include "../Models/suzi_smooth.h"
#include "../Models/plain.h"

class SceneFactory {
public:
    static std::vector<Scene*> createAllScenes(ShaderProgram* shader);

private:
    static Scene* createScene1(ShaderProgram* shader);
    static Scene* createScene2(ShaderProgram* shader);
    static Scene* createScene3(ShaderProgram* shader);
    static Scene* createScene4(ShaderProgram* shader);
    static Scene* createScene5(ShaderProgram* shader);
};
