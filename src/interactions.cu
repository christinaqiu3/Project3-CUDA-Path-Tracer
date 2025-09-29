#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__
float fresnelSchlick(float cosTheta, float etaI, float etaT) {
    float r0 = (etaI - etaT) / (etaI + etaT);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosTheta, 5.0f);
}



__host__ __device__
glm::vec3 sampleSpecularTrans(
    const glm::vec3& albedo,
    const glm::vec3& normal,
    const glm::vec3& wo, // outgoing direction, pointing away from surface
    float ior,           // index of refraction for the material
    glm::vec3& wiW,      // output: world-space refracted direction
    int& sampledType,    // output: type identifier
	thrust::default_random_engine& rng
) {


    // Assume air = 1.0, glass = ior
    float etaA = 1.0f;
    float etaB = ior;

    // Cosine of angle between wo and normal
    bool entering = glm::dot(-wo, normal) > 0.0f; // return w.z; // why not wo.z > 0 ?
    float etaI = entering ? etaA : etaB;
    float etaT = entering ? etaB : etaA;
    float eta = etaI / etaT;

    // Faceforward the normal
    glm::vec3 n = entering ? normal : -normal; // return (dot(n, v) < 0.f) ? -n : n; // input vec3(0.f,0.f,1.f),wo

    float cosTheta = glm::clamp(glm::dot(-wo, n), 0.0f, 1.0f);

    // Use Fresnel-Schlick to decide between reflection and refraction
    float reflectProb = fresnelSchlick(cosTheta, etaI, etaT);

    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    if (u01(rng) < reflectProb) {
        // Reflect
        wiW = glm::reflect(-wo, n);
        sampledType = 0;
    }
    else {
        // Refract
        wiW = glm::refract(wo, n, eta);

        if (glm::length(wiW) < 1e-6f) {
            wiW = glm::reflect(wo, n);
			sampledType = 0; // SPEC_REFL
        }
        else {
            sampledType = 1; // REFRACT
        }
    }
    return albedo; // / fabs(glm::dot(wiW, n)); // why not albedo / fabs(wiW.z);
}


__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.

	// specular reflection
    if (m.hasRefractive > 0.0f) {
        glm::vec3 wiW;
        int sampledType;
        glm::vec3 glassColor = sampleSpecularTrans(
            m.color,
			normal,
			pathSegment.ray.direction,
			m.indexOfRefraction,
			wiW,
			sampledType,
            rng
		);
        pathSegment.ray.origin = intersect + wiW * 0.001f;
        pathSegment.ray.direction = wiW;
        pathSegment.throughput *= glassColor;
        pathSegment.remainingBounces--;
        return;
    }

    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	pathSegment.ray.origin = intersect + .001f * normal;
	pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    glm::vec3 r = m.color / (m.color + glm::vec3(1.0));
    glm::vec3 gammaCorrect = pow(r, glm::vec3(1.0 / 2.2));
    pathSegment.throughput *= m.color; //gammaCorrect;
	pathSegment.remainingBounces--;
}
