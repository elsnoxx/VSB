#include "ShaderProgram.h"
#include "../Scene/Camera.h"
#include <algorithm>

ShaderProgram::ShaderProgram(const VertexShader& vertex, const FragmentShader& fragment) {
    id = glCreateProgram();
    glAttachShader(id, vertex.getId());
    glAttachShader(id, fragment.getId());
    glLinkProgram(id);

    GLint success;
    glGetProgramiv(id, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(id, 512, nullptr, log);
        std::cerr << "Shader program linking error: " << log << std::endl;
    }
}

ShaderProgram::~ShaderProgram() {
    removeAllCameras();
    glDeleteProgram(id);
}

void ShaderProgram::use() const {
    glUseProgram(id);
}

void ShaderProgram::setUniformMat4(const char* name, const glm::mat4& matrix) const {
    GLint location = glGetUniformLocation(id, name);
    if (location != -1) glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(matrix));
}

void ShaderProgram::setUniformVec3(const char* name, const glm::vec3& vec) const {
    GLint location = glGetUniformLocation(id, name);
    if (location != -1) glUniform3fv(location, 1, glm::value_ptr(vec));
}

void ShaderProgram::setUniformFloat(const char* name, float value) const {
    GLint location = glGetUniformLocation(id, name);
    if (location != -1) glUniform1f(location, value);
}

void ShaderProgram::setUniformInt(const char* name, int value) const {
    GLint location = glGetUniformLocation(id, name);
    if (location != -1) glUniform1i(location, value);
}

void ShaderProgram::addCamera(Camera* cam) {
    if (!cam) return;
    
    // Zkontroluj, zda kamera už není přidána
    if (std::find(m_cameras.begin(), m_cameras.end(), cam) != m_cameras.end()) {
        return;
    }
    
    m_cameras.push_back(cam);
    cam->addObserver(this);
    
    // Pokud je to první kamera, aktualizuj uniformy
    if (m_cameras.size() == 1) {
        updateCameraUniforms();
    }
}

void ShaderProgram::removeCamera(Camera* cam) {
    if (!cam) return;
    
    auto it = std::find(m_cameras.begin(), m_cameras.end(), cam);
    if (it != m_cameras.end()) {
        cam->removeObserver(this);
        m_cameras.erase(it);
    }
}

void ShaderProgram::removeAllCameras() {
    for (Camera* cam : m_cameras) {
        if (cam) cam->removeObserver(this);
    }
    m_cameras.clear();
}

void ShaderProgram::onCameraChanged(Camera* camera) {
    // Aktualizuj uniformy, pokud se změnila aktivní kamera (první v seznamu)
    if (!m_cameras.empty() && m_cameras[0] == camera) {
        updateCameraUniforms();
    }
}

void ShaderProgram::updateCameraUniforms() {
    if (m_cameras.empty()) return;
    
    Camera* activeCamera = m_cameras[0]; // použij první kameru jako aktivní
    
    use(); // aktivuj shader program
    
    glm::mat4 view = activeCamera->getViewMatrix();
    glm::mat4 proj = activeCamera->getProjectionMatrix();
    
    setUniformMat4("viewMatrix", view);
    setUniformMat4("projectionMatrix", proj);
}
