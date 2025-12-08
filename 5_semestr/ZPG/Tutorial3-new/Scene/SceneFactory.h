#pragma once
//Include GLEW
#include <GL/glew.h>
//Include GLFW
#include <GLFW/glfw3.h> 

#include <vector>
#include <random>
#include <chrono>

#include "Scene.h"
#include "../Shader/ShaderProgram.h"
#include "../ModelObject/Model.h"
#include "../Transform/Translation.h"
#include "../Transform/Rotation.h"
#include "../Transform/Scale.h"
#include "../Shader/ShaderType.h"

class SceneFactory {
public:
    static std::vector<Scene*> createAllScenes();

private:
    //Tutorila 1
    static Scene* createScene1();
    static Scene* createScene2();
    static Scene* createScene3();
    static Scene* createScene4();
    static Scene* createScene5();

    //Tutorila 3
    static Scene* createSceneSphereLights();    
    static Scene* createSceneDifferentModes();
    static Scene* createForestScene();

    //Tutorial 4
    static Scene* createSceneTinyObjects();
    static Scene* createSceneFormula1();
    static Scene* createSceneSolarSystem();

	
};
