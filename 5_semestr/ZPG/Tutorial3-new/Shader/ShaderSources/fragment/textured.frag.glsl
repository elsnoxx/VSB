#version 330 core

in vec2 uv;
in vec3 color;

out vec4 FragColor;

uniform sampler2D textureUnitID;

void main() {
    vec4 tex = texture(textureUnitID, uv);
    FragColor = tex;
}