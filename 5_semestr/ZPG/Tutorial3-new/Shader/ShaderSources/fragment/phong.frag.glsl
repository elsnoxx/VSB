#version 450 core
in vec3 worldPosition;
in vec3 worldNormal;
in vec3 albedo;
out vec4 frag_colour;

uniform int numLights;
uniform bool lightIsOn[16];
uniform int lightTypes[16];           // 0 = directional, 1 = point, 2 = spot
uniform vec3 lightPositions[16];
uniform vec3 lightDirections[16];     // normalized direction vector (where the light points)
uniform vec3 lightColors[16];
uniform float lightIntensities[16];
uniform float lightCutOffs[16];       // cos(innerAngle)
uniform float lightOuterCutOffs[16];  // cos(outerAngle)

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
    vec3 V = normalize(viewPosition - worldPosition);

    if (!gl_FrontFacing) {
        frag_colour = vec4(max(ambientStrength, 0.0) * baseColor, 1.0);
        return;
    }

    vec3 totalDiffuse = vec3(0.0);
    vec3 totalSpec = vec3(0.0);

    int n = clamp(numLights, 0, 16);
    for (int i = 0; i < n; ++i) {
        if (!lightIsOn[i]) continue; // skip if turned off

        vec3 Lvec = lightPositions[i] - worldPosition;
        float dist = length(Lvec);
        vec3 L = normalize(Lvec);

        float diff = 0.0;
        float spec = 0.0;
        float attenuation = 1.0;
        float spotEffect = 1.0;

        if (lightTypes[i] == 0) { // DIRECTIONAL
            vec3 dir = normalize(lightDirections[i]);
            L = normalize(-dir); // treat directional as coming from -direction
            diff = max(dot(N, L), 0.0);
            vec3 R = reflect(-L, N);
            spec = (diff > 0.0) ? pow(max(dot(V, R), 0.0), max(shininess, 1.0)) : 0.0;
            attenuation = 1.0;
        }
        else {
            // POINT or SPOT
            diff = max(dot(N, L), 0.0);
            vec3 R = reflect(-L, N);
            spec = (diff > 0.0) ? pow(max(dot(V, R), 0.0), max(shininess, 1.0)) : 0.0;
            attenuation = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist);

            if (lightTypes[i] == 2) { // SPOT
                vec3 spotDir = normalize(lightDirections[i]);
                float theta = dot(L, spotDir);
                float inner = lightCutOffs[i];
                float outer = lightOuterCutOffs[i];
                float epsilon = max(inner - outer, 0.0001);
                float intensity = clamp((theta - outer) / epsilon, 0.0, 1.0);
                spotEffect = intensity;
            }
        }

        vec3 lightCol = lightColors[i] * lightIntensities[i] * spotEffect;

        totalDiffuse += attenuation * lightCol * diff;
        totalSpec += attenuation * lightCol * spec;
    }

    float amb = max(ambientStrength, 0.0);
    vec3 color = amb * baseColor + baseColor * totalDiffuse + materialSpecular * totalSpec;
    frag_colour = vec4(color, 1.0);
}