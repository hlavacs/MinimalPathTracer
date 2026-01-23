#define _USE_MATH_DEFINES
#include <math.h>   //
#include <random>
#include <stdlib.h> // 
#include <stdio.h>  // 
#include <iostream>
#include <fstream>
#include <memory>
#include <functional>
#include <omp.h>
#include <glm/glm.hpp>

using Vec = glm::dvec3; 

inline double rand01(){
  static thread_local std::mt19937 rng(12345); // Fixed seed for reproducibility.
  static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng);
}

inline double halton(int index, int base) {
  double f = 1.0;
  double r = 0.0;
  int i = index;
  while (i > 0) {
    f /= base;
    r += f * (i % base);
    i /= base;
  }
  return r;
}

inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
inline Vec clamp(const Vec &v) { return Vec(clamp(v.x), clamp(v.y), clamp(v.z)); }

struct Ray { 
  Vec m_o, m_v;
  double m_tmin = 1e-4;
  double m_tmax = 1e10;
  Ray() = default;
  Ray(Vec o, Vec v, double tmin = 1e-4, double tmax = 1e10) : m_o(o), m_v(v), m_tmin(tmin), m_tmax(tmax) {} 
};

struct Camera {
  Vec m_o, m_v; //origin and direction
  int m_w, m_h; //width and height 
  Vec m_left_upper_corner;
  Vec m_horizontal;
  Vec m_vertical;

  Camera(Vec o, Vec v, Vec up, int w, int h, double fovDeg=30.0) : m_o(o), m_v(glm::normalize(v)), m_w(w), m_h(h) {
    double aspect = h > 0 ? static_cast<double>(w) / h : 1.0;
    double halfHeight = tan((fovDeg * M_PI / 180.0) * 0.5);
    double halfWidth = aspect * halfHeight;

    Vec forward = m_v;
    Vec right = glm::normalize(glm::cross(forward, up));
    Vec camUp = glm::normalize(glm::cross(right, forward));

    m_horizontal = right * (2.0 * halfWidth);
    m_vertical = camUp * (2.0 * halfHeight);
    m_left_upper_corner = m_o + forward - m_horizontal * 0.5 + m_vertical * 0.5;
  }

  Ray ray(double x, double y) { 
    Vec d = m_left_upper_corner + m_horizontal * (x / m_w) - m_vertical * (y / m_h) - m_o;
    return Ray(m_o, glm::normalize(d));
  }
};

enum class Material { DIFF, SPEC, REFR, LIGHT };  // material types, used in radiance()

struct Hit {
  double t;       // distance along ray to hit
  Vec p, n, e, c; // position, normal, emission, color
  double m_roughness; // specular roughness (0 = mirror)
  Material m_mat; // material type
};

struct Hitable {
  virtual Hit intersect(const Ray &ray) const = 0; // returns hit info, t=0 if no hit
  virtual Vec rnd_point() const = 0; // returns a random point on the surface
};

auto traverse( const auto &objects, const Ray &ray ) {
  Hit best{.t = 0.0};
  bool found = false;
  for (const std::unique_ptr<Hitable> &obj : objects) {
    Hit h = obj->intersect(ray);
    if (h.t > ray.m_tmin && h.t < ray.m_tmax && (!found || h.t < best.t)) { best = h; found = true; }
  }
  return std::make_pair(found, best);
}


std::vector<Hitable*> Lights;

struct Sphere : public Hitable {
  double m_r;          // radius
  Vec m_p, m_e, m_c;   // position, emission, color
  double m_roughness;  // specular roughness (0 = mirror)
  Material m_mat;      // reflection type (DIFFuse, SPECular, REFRactive)

  Sphere(double r, Vec p, Vec e, Vec c, Material mat, double roughness = 0.0):
    m_r(r), m_p(p), m_e(e), m_c(c), m_roughness(roughness), m_mat(mat) {}

  Hit intersect(const Ray &ray) const { // returns hit info, t=0 if no hit
    // Ray-sphere intersection: solve quadratic for t along ray direction.
    Vec op = m_p - ray.m_o; // vector from ray origin to sphere center
    double t, eps = 1e-4;
    double b = glm::dot(op, ray.m_v);
    double det = b * b - glm::dot(op, op) + m_r * m_r;
    // If discriminant is negative, the ray misses the sphere.
    if (det < 0) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), 0.0, m_mat};
    det = sqrt(det);
    // Return nearest positive hit; 0 means no hit.
    t = (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
    if (t == 0) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), 0.0, m_mat};
    Vec p = ray.m_o + ray.m_v * t;
    Vec n = glm::normalize(p - m_p);
    return Hit{t, p, n, m_e, m_c, m_roughness, m_mat};
  }

  Vec rnd_point() const {
    double z = 1.0 - 2.0 * rand01();
    double r = sqrt(glm::max(0.0, 1.0 - z * z));
    double phi = 2.0 * M_PI * rand01();
    double x = r * cos(phi);
    double y = r * sin(phi);
    return m_p + Vec(x, y, z) * m_r;
  }
};

struct Quad_XY : public Hitable {
  double m_x0, m_x1, m_y0, m_y1;  // extents along X and Y
  double d;  // distance from origin along Z axis
  Vec m_e, m_c;   // emission, color
  double m_roughness; // specular roughness (0 = mirror)
  Material m_mat; // reflection type (DIFFuse, SPECular, REFRactive
  bool m_flip;

  Quad_XY(double x0, double x1, double y0, double y1, double dist, Vec e, Vec c, Material mat, bool flip=false, double roughness = 0.0):
    m_x0(x0), m_x1(x1), m_y0(y0), m_y1(y1), d(dist), m_e(e), m_c(c), m_roughness(roughness), m_mat(mat), m_flip(flip) {}

  Hit intersect(const Ray &ray) const {
    double t = (d - ray.m_o.z) / ray.m_v.z; // solve for intersection with plane at z=d
    if (t <= 0 || t > 1e10) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), 0.0, m_mat};

    Vec p = ray.m_o + ray.m_v * t;
    if (p.x < m_x0 || p.x > m_x1 || p.y < m_y0 || p.y > m_y1) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), 0.0, m_mat};

    Vec n = m_flip ? Vec(0, 0, -1) : Vec(0, 0, 1);
    return Hit{t, p, n, m_e, m_c, m_roughness, m_mat};
  }

  Vec rnd_point() const {
    double x = m_x0 + (m_x1 - m_x0) * rand01();
    double y = m_y0 + (m_y1 - m_y0) * rand01();
    return Vec(x, y, d);
  }
};

struct Quad_XZ : public Hitable {
  double m_x0, m_x1, m_z0, m_z1;  // extents along X and Z
  double d;  // distance from origin along Y axis
  Vec m_e, m_c;   // emission, color
  double m_roughness; // specular roughness (0 = mirror)
  Material m_mat; // reflection type (DIFFuse, SPECular, REFRactive
  bool m_flip;

  Quad_XZ(double x0, double x1, double z0, double z1, double dist, Vec e, Vec c, Material mat, bool flip=false, double roughness = 0.0):
    m_x0(x0), m_x1(x1), m_z0(z0), m_z1(z1), d(dist), m_e(e), m_c(c), m_roughness(roughness), m_mat(mat), m_flip(flip) {}

  Hit intersect(const Ray &ray) const {
    double t = (d - ray.m_o.y) / ray.m_v.y; // solve for intersection with plane at y=d
    if (t <= 0 || t > 1e10) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), 0.0, m_mat};

    Vec p = ray.m_o + ray.m_v * t;
    if (p.x < m_x0 || p.x > m_x1 || p.z < m_z0 || p.z > m_z1) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), 0.0, m_mat};

    Vec n = m_flip ? Vec(0, -1, 0) : Vec(0, 1, 0);
    return Hit{t, p, n, m_e, m_c, m_roughness, m_mat};
  }

  Vec rnd_point() const {
    double x = m_x0 + (m_x1 - m_x0) * rand01();
    double z = m_z0 + (m_z1 - m_z0) * rand01();
    return Vec(x, d, z);
  }
};

struct Quad_YZ : public Hitable {
  double m_y0, m_y1, m_z0, m_z1;  // extents along Y and Z
  double d;  // distance from origin along X axis
  Vec m_e, m_c;   // emission, color
  double m_roughness; // specular roughness (0 = mirror)
  Material m_mat; // reflection type (DIFFuse, SPECular, REFRactive
  bool m_flip;

  Quad_YZ(double y0, double y1, double z0, double z1, double dist, Vec e, Vec c, Material mat, bool flip=false, double roughness = 0.0):
    m_y0(y0), m_y1(y1), m_z0(z0), m_z1(z1), d(dist), m_e(e), m_c(c), m_roughness(roughness), m_mat(mat), m_flip(flip) {}

  Hit intersect(const Ray &ray) const {
    double t = (d - ray.m_o.x) / ray.m_v.x; // solve for intersection with plane at x=d
    if (t <= 0 || t > 1e10) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), 0.0, m_mat};

    Vec p = ray.m_o + ray.m_v * t;
    if (p.y < m_y0 || p.y > m_y1 || p.z < m_z0 || p.z > m_z1) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), 0.0, m_mat};

    Vec n = m_flip ? Vec(-1, 0, 0) : Vec(1, 0, 0);
    return Hit{t, p, n, m_e, m_c, m_roughness, m_mat};
  }

  Vec rnd_point() const {
    double y = m_y0 + (m_y1 - m_y0) * rand01();
    double z = m_z0 + (m_z1 - m_z0) * rand01();
    return Vec(d, y, z);
  }
};

auto create_cornell(int count, auto &objects) {
  const double zShift = -6.5;
  const double xShift = -1.0;

  objects.emplace_back(std::make_unique<Sphere>(10000.0, Vec(0, -10000, 0), Vec(), Vec(0.3, 0.90, 0.25), Material::DIFF));

  // Cornell box walls
  objects.push_back(std::make_unique<Quad_YZ>(0, 2, zShift + 0, zShift + 2, xShift + 0, Vec(0), Vec(0.75, 0.25, 0.25), Material::DIFF, false)); // Left wall (faces +X)
  objects.push_back(std::make_unique<Quad_YZ>(0, 2, zShift + 0, zShift + 2, xShift + 2, Vec(0), Vec(0.25, 0.25, 0.75), Material::DIFF, true));  // Right wall (faces -X)
  objects.push_back(std::make_unique<Quad_XZ>(xShift + 0, xShift + 2, zShift + 0, zShift + 2, 0, Vec(0), Vec(0.75, 0.75, 0.75), Material::DIFF, false)); // Floor (faces +Y)
  objects.push_back(std::make_unique<Quad_XZ>(xShift + 0, xShift + 2, zShift + 0, zShift + 2, 2, Vec(0), Vec(0.75, 0.75, 0.75), Material::DIFF, true));  // Ceiling (faces -Y)
  objects.push_back(std::make_unique<Quad_XY>(xShift + 0, xShift + 2, 0, 2, zShift, Vec(0), Vec(0.75, 0.75, 0.75), Material::DIFF, true)); // Back wall (faces -Z)
  objects.push_back(std::make_unique<Quad_XZ>(xShift + 0.8, xShift + 1.2, zShift + 0.8, zShift + 1.2, 1.99, Vec(12, 12, 12), Vec(0), Material::LIGHT, true));  // Ceiling (faces -Y)
  Lights.push_back(objects.back().get());

  // Two spheres resting on the floor inside the Cornell box.
  objects.push_back(std::make_unique<Sphere>(0.35, Vec(xShift + 0.65, 0.35, zShift + 0.6), Vec(), Vec(0.95, 0.95, 0.95), Material::SPEC, 0.0));
  objects.push_back(std::make_unique<Sphere>(0.5, Vec(xShift + 1.35, 0.5, zShift + 1.4), Vec(), Vec(0.95, 0.95, 0.95), Material::SPEC, 0.5));

  return 3.0 / 3.3; // aspect ratio
}


void create_random_sphere(auto &objects) {
  static std::vector<Vec> centers; // track prior centers for spacing
  double radius = 0.1 + rand01() * 0.3; // small sphere radius
  Vec pos; // candidate position
  for (int tries = 0; tries < 50; ++tries) { // retry to enforce spacing
    double x = (rand01() * 2.0 - 1.0) * 4.0; // wider X spread
    double z = -4.0 - rand01() * 22.0; // more variance in camera distance
    pos = Vec(x, radius, z); // place on ground plane
    bool ok = true; // spacing check
    for (const Vec &c : centers) { // compare to previous
      double minDist = 2.5 * radius; // keep separation
      if (glm::length(pos - c) < minDist) { ok = false; break; } // too close
    }
    if (ok) { break; } // accept position
  }
  centers.push_back(pos); // remember position
  Vec color(0.2 + rand01() * 0.8, 0.2 + rand01() * 0.8, 0.2 + rand01() * 0.8); // random albedo
  bool makeSpec = rand01() < 0.6; // specular probability
  double roughness = makeSpec ? rand01()/3.0 : 0.0; // roughness for specular
  Material mat = makeSpec ? Material::SPEC : Material::DIFF; // choose material
  objects.emplace_back(std::make_unique<Sphere>(radius, pos, Vec(), color, mat, roughness)); // add sphere
  return; // no aspect ratio needed
}

auto create_spheres(int count, auto &objects) {
  // Ground and a simple overhead light.
  objects.emplace_back(std::make_unique<Sphere>(10000.0, Vec(0, -10000, 0), Vec(), Vec(0.3, 0.90, 0.25), Material::DIFF));
  objects.emplace_back(std::make_unique<Sphere>(1.0, Vec(10, 10, 0), Vec(5, 5, 5), Vec(), Material::LIGHT));
  Lights.push_back(objects.back().get());

  for (int i = 0; i < count; ++i) { create_random_sphere(objects); }
  return 16.0 / 9.0;
}

double russian_roulette(const Vec &throughput, int depth) {
  if (depth < 5) return 1.0;
  double p = glm::max(throughput.x, glm::max(throughput.y, throughput.z));
  p = glm::clamp(p, 0.05, 0.95);
  return rand01() < p ? (1.0 / p) : 0.0;
}

Vec sky_radiance(const Ray &r) {
  double t = 0.5 * (r.m_v.y + 1.0);
  t = t * t;
  Vec sky = Vec(1.0, 1.0, 1.0) * (1.0 - t) + Vec(0.2, 0.4, 1.0) * t;
  return sky;
}

Ray sample_diffuse_ray(const Vec &p, const Vec &nl, const Vec &albedo, Vec &f, double &pdf) { // diffuse sample entry
  // Cosine-weighted hemisphere sampling around the normal. // sampling method
  double r1 = 2.0 * M_PI * rand01(); // azimuthal sample
  double r2 = rand01(); // radial sample
  double r2s = sqrt(r2); // radius term
  Vec u = glm::normalize(glm::cross(fabs(nl.x) > 0.1 ? Vec(0, 1, 0) : Vec(1, 0, 0), nl)); // tangent
  Vec v = glm::cross(nl, u); // bitangent
  Vec d = glm::normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + nl * sqrt(1 - r2)); // sample dir
  double cosTheta = glm::max(0.0, glm::dot(nl, d)); // cosine term
  f = albedo * (1.0 / M_PI); // Lambertian BRDF
  pdf = cosTheta * (1.0 / M_PI); // cosine-weighted pdf
  return Ray(p + 0e-4 * nl, d); // spawn ray
} // end diffuse sample

Ray sample_specular_ray(const Vec &p, const Vec &nl, const Vec &inDir, double roughness, const Vec &albedo, Vec &f, double &pdf) { // specular sample entry
  double r = glm::clamp(roughness, 0.0, 1.0); // clamp roughness
  Vec refl = glm::normalize(inDir - 2.0 * glm::dot(inDir, nl) * nl); // mirror direction
  if (r <= 0.0) { // perfect mirror
    f = albedo; // specular color
    pdf = glm::max(1e-12, glm::dot(nl, refl)); // pdf placeholder
    return Ray(p + 1e-4 * nl, refl); // spawn mirror ray
  } // end mirror case

  double r1 = 2.0 * M_PI * rand01(); // azimuthal sample
  double r2 = rand01(); // radial sample
  double r2s = sqrt(r2); // radius term
  Vec u = glm::normalize(glm::cross(fabs(refl.x) > 0.1 ? Vec(0, 1, 0) : Vec(1, 0, 0), refl)); // tangent
  Vec v = glm::cross(refl, u); // bitangent
  Vec hemi = glm::normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + refl * sqrt(1 - r2)); // hemisphere dir
  Vec d = glm::normalize((1.0 - r) * refl + r * hemi); // mix reflection
  f = albedo; // specular color
  pdf = glm::max(1e-12, glm::dot(nl, d)); // pdf placeholder
  return Ray(p + 1e-4 * nl, d); // spawn glossy ray
} // end specular sample


Vec radiance(const auto &objects, const Ray &r) {
  Ray ray = r;
  int curDepth = 0; // recursion depth
  Vec tp = Vec(1.0); // throughput
  Vec radianceSum = Vec(0.0);

  for (;;) {
    auto [found, best] = traverse(objects, ray);
    if (!found) { return radianceSum + tp * sky_radiance(ray); }  
    if (curDepth >= 32 || best.m_mat == Material::LIGHT) 
      return radianceSum + tp * best.e; // max depth or hit light

    radianceSum += tp * best.e; // accumulate emitted radiance

    Vec nl = glm::dot(best.n, ray.m_v) < 0 ? best.n : -best.n; // oriented normal

    Vec f;
    double pdf;
    if (best.m_mat == Material::SPEC) {
      ray = sample_specular_ray(best.p, nl, ray.m_v, best.m_roughness, best.c, f, pdf);
    } else {
      ray = sample_diffuse_ray(best.p, nl, best.c, f, pdf);
    }

    tp *= f * (glm::max(0.0, glm::dot(nl, ray.m_v)) / pdf); // update throughput

    double rrWeight = russian_roulette(tp, curDepth); // Russian roulette
    if (rrWeight == 0.0) { return radianceSum; }
    tp *= rrWeight; // unbiased RR compensation

    ++curDepth; // increase recursion depth
  }
}


int main(int argc, char *argv[]){
  int scene = argc>=2 ? atoi(argv[1]) : 0; // # scene
  int samples = argc==3 ? atoi(argv[2]) : 128; // # samples
  double div{1./samples};

  std::vector<std::unique_ptr<Hitable>> objects;
  double aspect;
  if( scene == 0 ) { aspect = create_spheres(25, objects); } 
  else { aspect = create_cornell(0, objects); }

  int w{1920}, h{(int)(w / aspect)}; // image resolution
  Camera cam(Vec(0,1,0), Vec(0,0,-1), Vec(0,1,0), w, h);
  std::vector<Vec> colors;
  colors.resize(w * h);
  Ray ray;

#pragma omp parallel for schedule(dynamic, 8) private(ray)       // OpenMP

  for (int y=0; y<h; y++) {                       // Loop over image rows
    if (omp_get_thread_num() == 0 || omp_get_num_threads() == 1) {
      int percent = static_cast<int>(100.0 * y / (h - 1));
      std::cout << "\r\033 Rendering (" << samples << " spp) " << percent << "%" << std::flush;
    }
    for (unsigned short x=0; x<w; x++) { // Loop cols
      Vec sum(0);
      for(int s=0; s<samples; s++) {
        int hIndex = s + 1;
        ray = cam.ray(x + halton(hIndex, 2), y + halton(hIndex, 3)); // Generate camera ray
        sum = sum + radiance(objects, ray); // Trace ray and accumulate radiance
      }
      colors[x + y * w] = clamp(sum * div) * 255.0;
    }
  }

  std::ofstream f("image.ppm");         // Write image to PPM file.
  f << "P3\n" << w << " " << h << "\n" << 255 << "\n";
  for (int i=0; i<w*h; i++)
    f << int(colors[i].x) << " " << int(colors[i].y) << " " << int(colors[i].z) << " ";
}
