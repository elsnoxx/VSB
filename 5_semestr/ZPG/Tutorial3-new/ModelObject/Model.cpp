#include "Model.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "../lib/tiny_object/tiny_obj_loader.h"


Model::Model(const float* data, size_t size, int count)
    : vertexCount(count)
{
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glEnableVertexAttribArray(0); // pozice
    glEnableVertexAttribArray(1); // barva

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (GLvoid*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (GLvoid*)(3 * sizeof(float)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

Model::Model(const char* name) {
    std::string inputfile = std::string("ModelObject/assets/") + name;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str(), "ModelObject/assets/");

    if (!warn.empty()) std::cout << "tinyobj warn: " << warn << std::endl;
    if (!err.empty()) std::cerr << "tinyobj err: " << err << std::endl;
    if (!ret) throw std::runtime_error("Failed to load OBJ file!");

    // interleaved: pos(3), normal(3), uv(2) => stride = 8 floats
    std::vector<float> vertices;
    vertices.reserve(attrib.vertices.size() / 3 * 8);

    for (const auto& shape : shapes) {
        for (const auto& idx : shape.mesh.indices) {
            // position
            vertices.push_back(attrib.vertices[3 * idx.vertex_index + 0]);
            vertices.push_back(attrib.vertices[3 * idx.vertex_index + 1]);
            vertices.push_back(attrib.vertices[3 * idx.vertex_index + 2]);

            // normal
            if (idx.normal_index >= 0) {
                vertices.push_back(attrib.normals[3 * idx.normal_index + 0]);
                vertices.push_back(attrib.normals[3 * idx.normal_index + 1]);
                vertices.push_back(attrib.normals[3 * idx.normal_index + 2]);
            }
            else {
                vertices.push_back(0.0f);
                vertices.push_back(0.0f);
                vertices.push_back(0.0f);
            }

            // texcoord
            if (idx.texcoord_index >= 0) {
                vertices.push_back(attrib.texcoords[2 * idx.texcoord_index + 0]);
                vertices.push_back(attrib.texcoords[2 * idx.texcoord_index + 1]);
            }
            else {
                vertices.push_back(0.0f);
                vertices.push_back(0.0f);
            }
        }
    }

    // upload
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    vertexCount = static_cast<int>(vertices.size() / 8);

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // attribute locations: 0 = pos, 1 = normal, 2 = texcoord
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));

    // unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

Model::~Model() {
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
}

void Model::draw() const {
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, vertexCount);
}
