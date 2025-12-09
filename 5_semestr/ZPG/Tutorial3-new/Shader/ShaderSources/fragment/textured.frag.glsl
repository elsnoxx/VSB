#version 450 core
in vec2 uv;
out vec4 fragColor;

uniform sampler2D texture0;
uniform sampler2D texture1;

void main() {
    vec4 c0 = texture(texture0, uv);
    vec4 c1 = texture(texture1, uv);

    fragColor = mix(c0, c1, c1.a);
}