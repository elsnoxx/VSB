#version 330 core

uniform sampler2D textureUnitID;

in vec2 uv;

out vec4 fragColor;

uniform float ambient = 1.0f;

void main () {
    vec4 tex = texture(textureUnitID, uv);
    fragColor = tex * ambient;
}
