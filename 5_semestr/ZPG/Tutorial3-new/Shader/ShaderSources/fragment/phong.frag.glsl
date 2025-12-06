#version 330 core
in vec3 worldPosition;
in vec3 worldNormal;
in vec3 albedo;
out vec4 frag_colour;

uniform vec3 lightPosition;
uniform vec3 viewPosition;
uniform float shininess;
uniform float ambientStrength;
uniform vec3 materialDiffuse;
uniform vec3 materialSpecular;

const vec3 DEFAULT_DIFFUSE = vec3(0.35, 0.35, 0.35);
const vec3 DEFAULT_SPECULAR = vec3(1.0);

void main()
{
    vec3 baseColor = (length(albedo) > 0.001) ? albedo :
                     (length(materialDiffuse) > 0.001 ? materialDiffuse : DEFAULT_DIFFUSE);

    vec3 N = normalize(worldNormal);
    vec3 L = normalize(lightPosition - worldPosition);
    float diff = max(dot(N, L), 0.0);

    vec3 V = normalize(viewPosition - worldPosition);
    vec3 R = reflect(-L, N);
    float spec = (diff>0.0) ? pow(max(dot(V, R), 0.0), shininess) : 0.0;

    float amb = (ambientStrength > 0.0) ? ambientStrength : 0.15;
    vec3 color = amb * baseColor + diff * baseColor + spec * materialSpecular;
    frag_colour = vec4(color, 1.0);
}