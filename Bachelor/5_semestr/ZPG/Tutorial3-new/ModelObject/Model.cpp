#include "Model.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "../lib/tiny_object/tiny_obj_loader.h"


// Construct a Model from raw interleaved vertex data.
// Parameters:
//  - data: pointer to float array containing interleaved vertex attributes
//  - size: size in bytes of the data buffer
//  - count: number of vertices
// The constructor creates a VBO, uploads the data, creates a VAO and
// sets attribute pointers. The layout is inferred from `count` and `size`.
Model::Model(const float* data, size_t size, int count)
    : vertexCount(count)
{
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // compute how many floats each vertex contains
    int floatsPerVertex = 0;
    if (count > 0) floatsPerVertex = static_cast<int>(size / sizeof(float) / count);
    GLsizei stride = floatsPerVertex * sizeof(float);

    // position (vec3) always at location 0
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (GLvoid*)0);

    // normal (vec3) at location 1 if present (at least 6 floats)
    if (floatsPerVertex >= 6) {
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (GLvoid*)(3 * sizeof(float)));
    }

    // texcoord (vec2) at location 2 if present (at least 8 floats)
    if (floatsPerVertex >= 8) {
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (GLvoid*)(6 * sizeof(float)));
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// Construct a Model by loading an OBJ file (tinyobjloader).
// The OBJ is parsed and converted to an interleaved float buffer with the
// layout: position(3), normal(3), texcoord(2). Missing normals/texcoords are
// replaced with zeros. The resulting buffer is uploaded to a VBO/VAO.
Model::Model(const char* name) {
    std::string inputfile = std::string("ModelObject/assets/") + name;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    std::string baseDir = inputfile.substr(0, inputfile.find_last_of("/\\") + 1);

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str(), baseDir.c_str());

    if (!warn.empty()) printf("[Model] tinyobj warning: %s\n", warn.c_str());
    if (!err.empty())  printf("[Model] tinyobj error: %s\n", err.c_str());
    if (!ret) {
        std::cerr << "[Model] Failed to load/parse .obj file: " << inputfile << "\n";
        return;
    }
    // Log basic info about loaded shapes/materials
    printf("[Model] load request filename='%s' base_dir='%s' -> shapes=%zu materials=%zu\n", inputfile.c_str(), baseDir.c_str(), shapes.size(), materials.size());
    for (size_t i = 0; i < materials.size(); ++i) {
        printf("[Model] material[%zu] name='%s' diffuseTex='%s'\n", i, materials[i].name.c_str(), materials[i].diffuse_texname.c_str());
    }

    // interleaved: pos(3), normal(3), uv(2) => stride = 8 floats
    std::vector<float> vertices;
    vertices.reserve(attrib.vertices.size() / 3 * 8);

    bool hasTex = false;

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
                hasTex = true;
            }
            else {
                vertices.push_back(0.0f);
                vertices.push_back(0.0f);
            }
        }
    }

    if (!hasTex) {
        std::cerr << "[Model] Warning: model '" << name << "' has NO texcoords. Textured shader will sample (0,0).\n";
    }
    else {
        std::cerr << "[Model] Info: model '" << name << "' contains texcoords.\n";
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
    // Delete GL objects (VBO and VAO)
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
}

void Model::draw() const {
    // Bind VAO and issue draw call. Model uses non-indexed triangles.
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, vertexCount);
}
