#define _USE_MATH_DEFINES
#include <math.h>   //
#include <random>
#include <stdlib.h> // 
#include <stdio.h>  // 
#include <iostream>
#include <fstream>
#include <memory>
#include <glm/glm.hpp>

using Vec = glm::dvec3; 

inline double rand01(){
  static thread_local std::mt19937 rng(12345); // Fixed seed for reproducibility.
  static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng);
}

inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
inline Vec clamp(const Vec &v) { return Vec(clamp(v.x), clamp(v.y), clamp(v.z)); }

struct Ray { 
  Vec m_o, m_v;
  Ray() = default;
  Ray(Vec o, Vec v) : m_o(o), m_v(v) {} 
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

enum Material { DIFF, SPEC, REFR, LIGHT };  // material types, used in radiance()

struct Hit {
  double t;
  Vec p, n, e, c;
  Material m_mat;
};

struct Hitable {
  virtual Hit intersect(const Ray &ray) const = 0; // returns hit info, t=0 if no hit
};

struct Sphere : public Hitable {
  double m_r;          // radius
  Vec m_p, m_e, m_c;   // position, emission, color
  Material m_mat;      // reflection type (DIFFuse, SPECular, REFRactive)

  Sphere(double r, Vec p, Vec e, Vec c, Material mat):
    m_r(r), m_p(p), m_e(e), m_c(c), m_mat(mat) {}

  Hit intersect(const Ray &ray) const { // returns hit info, t=0 if no hit
    // Ray-sphere intersection: solve quadratic for t along ray direction.
    Vec op = m_p - ray.m_o; // vector from ray origin to sphere center
    double t, eps = 1e-4;
    double b = glm::dot(op, ray.m_v);
    double det = b * b - glm::dot(op, op) + m_r * m_r;
    // If discriminant is negative, the ray misses the sphere.
    if (det < 0) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), m_mat};
    det = sqrt(det);
    // Return nearest positive hit; 0 means no hit.
    t = (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
    if (t == 0) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), m_mat};
    Vec p = ray.m_o + ray.m_v * t;
    Vec n = glm::normalize(p - m_p);
    return Hit{t, p, n, m_e, m_c, m_mat};
  }
};

struct Quad : public Hitable {
  Vec m_v0, m_v1, m_v2, m_v3; // corners in CCW order
  Vec m_e, m_c;           // emission, color
  Material m_mat;     // reflection type (DIFFuse, SPECular, REFRactive)

  Quad(Vec V0, Vec V1, Vec V2, Vec V3, Vec e, Vec c, Material mat) :
    m_v0(V0), m_v1(V1), m_v2(V2), m_v3(V3), m_e(e), m_c(c), m_mat(mat) {}

  Hit intersect(const Ray &ray) const {
    Vec n = glm::normalize(glm::cross(m_v1 - m_v0, m_v2 - m_v0));
    double denom = glm::dot(n, ray.m_v);
    if (fabs(denom) < 1e-6) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), m_mat};
    double t = glm::dot(m_v0 - ray.m_o, n) / denom;
    if (t <= 1e-4) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), m_mat};
    Vec p = ray.m_o + ray.m_v * t;

    // Inside-out test for CCW quad.
    Vec c0 = glm::cross(m_v1 - m_v0, p - m_v0);
    Vec c1 = glm::cross(m_v2 - m_v1, p - m_v1);
    Vec c2 = glm::cross(m_v3 - m_v2, p - m_v2);
    Vec c3 = glm::cross(m_v0 - m_v3, p - m_v3);
    if (glm::dot(n, c0) < 0 || glm::dot(n, c1) < 0 || glm::dot(n, c2) < 0 || glm::dot(n, c3) < 0)
      return Hit{0.0, Vec(), Vec(), Vec(), Vec(), m_mat};

    return Hit{t, p, n, m_e, m_c, m_mat};
  }
};

std::vector<std::unique_ptr<Hitable>> create_spheres(int count) {
  std::vector<std::unique_ptr<Hitable>> result;

  // Ground and a simple overhead light.
  result.emplace_back(std::make_unique<Sphere>(10000.0, Vec(0, -10000, 0), Vec(), Vec(0.3, 0.90, 0.25), DIFF));
  result.emplace_back(std::make_unique<Sphere>(1.0, Vec(10, 10, 0), Vec(5, 5, 5), Vec(), LIGHT));
  result.emplace_back(std::make_unique<Sphere>(0.4, Vec(0, 0.4, -5), Vec(), Vec(0.8, 0.8, 0.8), DIFF));

  for (int i = 0; i < count; ++i) {
    double x = (rand01() * 2.0 - 1.0) * 5.0;
    double z = -5.0 - rand01() * 12.0;
    double t = (z - (-5.0)) / (-21.0 - (-5.0)); // 0 near, 1 far
    double sizeJitter = 0.3 + rand01() * 0.9;
    double radius = (0.15 + t * 1.35) * sizeJitter;
    Vec pos(x, radius, z);
    Vec color(0.2 + rand01() * 0.8, 0.2 + rand01() * 0.8, 0.2 + rand01() * 0.8);
    result.emplace_back(std::make_unique<Sphere>(radius, pos, Vec(), color, DIFF));
  }

  return result;
}

std::vector<std::unique_ptr<Hitable>> objects;

Vec radiance(const Ray &r, int depth){
  Hit best;
  best.t = 0;
  bool found = false;
  for (const std::unique_ptr<Hitable> &obj : objects) {
    Hit h = obj->intersect(r);
    if (h.t > 0 && (!found || h.t < best.t)) { best = h; found = true; }
  }
  if (!found) {
    // Simple sky gradient based on ray direction.
    double tSky = 0.5 * (r.m_v.y + 1.0);
    tSky = tSky * tSky;
    return Vec(1.0, 1.0, 1.0) * (1.0 - tSky) + Vec(0.2, 0.4, 1.0) * tSky;
  }

  Vec x = best.p;
  Vec n = best.n;
  Vec nl = glm::dot(n, r.m_v) < 0 ? n : -n;

  if (depth >= 20 || best.m_mat == LIGHT) return best.e;

  // Cosine-weighted hemisphere sampling around the normal.
  double r1 = 2.0 * M_PI * rand01();
  double r2 = rand01();
  double r2s = sqrt(r2);
  Vec w = nl;
  Vec u = glm::normalize(glm::cross(fabs(w.x) > 0.1 ? Vec(0, 1, 0) : Vec(1, 0, 0), w));
  Vec v = glm::cross(w, u);
  Vec d = glm::normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2));

  return best.e + best.c * radiance(Ray(x, d), depth + 1);
}


int main(int argc, char *argv[]){
  int w = 800, h = (int)(w * 9. / 16.), samples = argc==2 ? atoi(argv[1]) : 256; // # samples
  objects = create_spheres(10);
  std::vector<Vec> colors;
  colors.resize(w*h);
  Camera cam(Vec(0,1,0), Vec(0,0,-1), Vec(0,1,0), w, h);
  Ray ray;

#pragma omp parallel for schedule(dynamic, 1) private(ray)       // OpenMP

  for (int y=0; y<h; y++) {                       // Loop over image rows
    int percent = static_cast<int>(100.0 * y / (h - 1));
    std::cout << "\rRendering (" << samples << " spp) " << percent << "%     " << std::flush;

    for (unsigned short x=0; x<w; x++) { // Loop cols
      Vec sum(0);
      for(int s=0; s<samples; s++) {
        ray = cam.ray(x + rand01(), y + rand01());
        sum = sum + radiance(ray, 1);
      }
      colors[x + y * w] = clamp(sum * (1./samples)) * 255.0;
    }
  }

  std::ofstream f("image.ppm");         // Write image to PPM file.
  f << "P3\n" << w << " " << h << "\n" << 255 << "\n";
  for (int i=0; i<w*h; i++)
    f << int(colors[i].x) << " " << int(colors[i].y) << " " << int(colors[i].z) << " ";
}
