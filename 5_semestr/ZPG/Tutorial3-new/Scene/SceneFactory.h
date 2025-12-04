#pragma once
//Include GLEW
#include <GL/glew.h>
//Include GLFW
#include <GLFW/glfw3.h> 

#include <vector>
#include "Scene.h"
#include "../Shader/ShaderProgram.h"
#include "../ModelObject/Model.h"

// sem include modelov�ch dat
#include "../Models/sphere.h"
#include "../Models/tree.h"
#include "../Models/bushes.h"
#include "../Models/gift.h"
#include "../Models/suzi_flat.h"
#include "../Models/suzi_smooth.h"
#include "../Models/plain.h"
#include "../Transform/Translation.h"
#include "../Transform/Rotation.h"
#include "../Transform/Scale.h"
#include "../Shader/ShaderType.h"

class SceneFactory {
public:
    static std::vector<Scene*> createAllScenes();

private:
    static Scene* createScene1();
    static Scene* createScene2();
    static Scene* createScene3();
    static Scene* createScene4();
    static Scene* createScene5();
    static Scene* createSceneSphereLights();

	
};
