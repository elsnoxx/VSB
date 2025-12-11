#include "ShaderProgram.h"


ShaderProgram::ShaderProgram(const char* vertexSrc, const char* fragmentSrc) {
    // 1. Compile the vertex shader
    VertexShader vertex(vertexSrc);

    // 2. Zkompilujeme fragment shader
    // 2. Compile the fragment shader
    FragmentShader fragment(fragmentSrc);

    // 3. Create program
    id = glCreateProgram();

    // 4. Attach shaders
    glAttachShader(id, vertex.getId());
    glAttachShader(id, fragment.getId());

    // 5. Link program
    glLinkProgram(id);


    // 6. Error checking
    GLint success;
    glGetProgramiv(id, GL_LINK_STATUS, &success);

    if (!success) {
        char log[512];
        glGetProgramInfoLog(id, 512, nullptr, log);
        std::cerr << "ShaderProgram linking error:\n" << log << std::endl;
    }

    use();
    setUniform("textureUnitID", 0);
    glUseProgram(0);
}

void ShaderProgram::update(ObservableSubjects subject)
{
    use();

    if (subject == ObservableSubjects::SCamera) {
        if (camera) {
            setUniform("viewMatrix", camera->getViewMatrix());
            setUniform("projectionMatrix", camera->getProjectionMatrix());
            // shader code expects uniform named "viewPosition"
            setUniform("viewPosition", camera->getPosition());
        }
        else {
            std::cerr << "ShaderProgram::update() WARNING: camera is null\n";
        }
    }


    glUseProgram(id);
}

void ShaderProgram::attachCamera(Camera* cam)
{
    if (!cam) return;

    if (camera)
		camera->detach(this);
    camera = cam;
    camera->attach(this);
	update(ObservableSubjects::SCamera);
}


ShaderProgram::~ShaderProgram() {
    glDeleteProgram(id);
}

void ShaderProgram::use() const {
    glUseProgram(id);
}

void ShaderProgram::setUniform(const char* name, const glm::mat4& matrix) const {
    GLint location = glGetUniformLocation(id, name);
    if (location != -1) glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(matrix));
}

void ShaderProgram::setUniform(const char* name, const glm::vec3& vec) const {
    GLint location = glGetUniformLocation(id, name);
    if (location != -1) glUniform3fv(location, 1, glm::value_ptr(vec));
}

void ShaderProgram::setUniform(const char* name, float value) const {
    GLint location = glGetUniformLocation(id, name);
    if (location != -1) glUniform1f(location, value);
}

void ShaderProgram::setUniform(const char* name, int value) const {
    GLint location = glGetUniformLocation(id, name);
    if (location != -1) glUniform1i(location, value);
}
