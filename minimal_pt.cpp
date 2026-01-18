#define _USE_MATH_DEFINES
#include <math.h>   //
#include <random>
#include <stdlib.h> // 
#include <stdio.h>  // 
#include <iostream>
#include <fstream>
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

struct Hitable {
  virtual double intersect(const Ray &ray) const = 0; // returns distance, 0 if no hit
};

struct Sphere : public Hitable {
  double m_r;          // radius
  Vec m_p, m_e, m_c;   // position, emission, color
  Material m_mat;      // reflection type (DIFFuse, SPECular, REFRactive)

  Sphere(double r, Vec p, Vec e, Vec c, Material mat):
    m_r(r), m_p(p), m_e(e), m_c(c), m_mat(mat) {}

  double intersect(const Ray &ray) const { // returns distance, 0 if no hit
    // Ray-sphere intersection: solve quadratic for t along ray direction.
    Vec op = m_p - ray.m_o; // vector from ray origin to sphere center
    double t, eps = 1e-4;
    double b = glm::dot(op, ray.m_v);
    double det = b * b - glm::dot(op, op) + m_r * m_r;
    // If discriminant is negative, the ray misses the sphere.
    if (det < 0) return 0;
    det = sqrt(det);
    // Return nearest positive hit; 0 means no hit.
    return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
  }
};

std::vector<Sphere> create_spheres(int count) {
  std::vector<Sphere> result;

  // Ground and a simple overhead light.
  result.emplace_back(10000.0, Vec(0, -10000, 0), Vec(),        Vec(0.3, 0.80, 0.25), DIFF);
  result.emplace_back(    2.0, Vec( 10,    10, 0), Vec(12, 12, 12), Vec(),            LIGHT);

  for (int i = 0; i < count; ++i) {
    double x = (rand01() * 2.0 - 1.0) * 5.0;
    double z = -5.0 - rand01() * 12.0;
    double t = (z - (-5.0)) / (-21.0 - (-5.0)); // 0 near, 1 far
    double sizeJitter = 0.3 + rand01() * 0.9;
    double radius = (0.15 + t * 1.35) * sizeJitter;
    Vec pos(x, radius, z);
    Vec color(0.2 + rand01() * 0.8, 0.2 + rand01() * 0.8, 0.2 + rand01() * 0.8);
    result.emplace_back(radius, pos, Vec(), color, DIFF);
  }

  return result;
}

std::vector<Sphere> spheres;

Vec radiance(const Ray &r, int depth){
  double t = 0;
  const Sphere *hit = nullptr;
  for (const Sphere &s : spheres) {
    double d = s.intersect(r);
    if (d && (!hit || d < t)) { t = d; hit = &s; }
  }
  if (!hit) {
    // Simple sky gradient based on ray direction.
    double tSky = 0.5 * (r.m_v.y + 1.0);
    tSky = tSky * tSky;
    return 0.5*Vec(1.0, 1.0, 1.0) * (1.0 - tSky) + Vec(0.2, 0.4, 1.0) * tSky;
  }

  const Sphere &obj = *hit;
  Vec x = r.m_o + r.m_v * t;
  Vec n = glm::normalize(x - obj.m_p);
  Vec nl = glm::dot(n, r.m_v) < 0 ? n : -n;

  if (depth >= 20 || obj.m_mat == LIGHT) return obj.m_e;

  // Cosine-weighted hemisphere sampling around the normal.
  double r1 = 2.0 * M_PI * rand01();
  double r2 = rand01();
  double r2s = sqrt(r2);
  Vec w = nl;
  Vec u = glm::normalize(glm::cross(fabs(w.x) > 0.1 ? Vec(0, 1, 0) : Vec(1, 0, 0), w));
  Vec v = glm::cross(w, u);
  Vec d = glm::normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2));

  return obj.m_e + obj.m_c * radiance(Ray(x, d), depth + 1);
}


int main(int argc, char *argv[]){
  int w = 800, h = w * 9. / 16., samples = argc==2 ? atoi(argv[1]) : 256; // # samples
  spheres = create_spheres(10);
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
