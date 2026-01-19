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

enum Material { DIFF, SPEC, REFR, LIGHT };  // material types, used in radiance()

struct Hit {
  double t;       // distance along ray to hit
  Vec p, n, e, c; // position, normal, emission, color
  Material m_mat; // material type
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

struct Quad_XY : public Hitable {
  double m_x0, m_x1, m_y0, m_y1;  // extents along X and Y
  double d;  // distance from origin along Z axis
  Vec m_e, m_c;   // emission, color
  Material m_mat; // reflection type (DIFFuse, SPECular, REFRactive
  bool m_flip;

  Quad_XY(double x0, double x1, double y0, double y1, double dist, Vec e, Vec c, Material mat, bool flip=false):
    m_x0(x0), m_x1(x1), m_y0(y0), m_y1(y1), d(dist), m_e(e), m_c(c), m_mat(mat), m_flip(flip) {}

  Hit intersect(const Ray &ray) const {
    double t = (d - ray.m_o.z) / ray.m_v.z; // solve for intersection with plane at z=d
    if (t <= 0 || t > 1e10) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), m_mat};

    Vec p = ray.m_o + ray.m_v * t;
    if (p.x < m_x0 || p.x > m_x1 || p.y < m_y0 || p.y > m_y1) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), m_mat};

    Vec n = m_flip ? Vec(0, 0, -1) : Vec(0, 0, 1);
    return Hit{t, p, n, m_e, m_c, m_mat};
  }
};

struct Quad_XZ : public Hitable {
  double m_x0, m_x1, m_z0, m_z1;  // extents along X and Z
  double d;  // distance from origin along Y axis
  Vec m_e, m_c;   // emission, color
  Material m_mat; // reflection type (DIFFuse, SPECular, REFRactive
  bool m_flip;

  Quad_XZ(double x0, double x1, double z0, double z1, double dist, Vec e, Vec c, Material mat, bool flip=false):
    m_x0(x0), m_x1(x1), m_z0(z0), m_z1(z1), d(dist), m_e(e), m_c(c), m_mat(mat), m_flip(flip) {}

  Hit intersect(const Ray &ray) const {
    double t = (d - ray.m_o.y) / ray.m_v.y; // solve for intersection with plane at y=d
    if (t <= 0 || t > 1e10) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), m_mat};

    Vec p = ray.m_o + ray.m_v * t;
    if (p.x < m_x0 || p.x > m_x1 || p.z < m_z0 || p.z > m_z1) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), m_mat};

    Vec n = m_flip ? Vec(0, -1, 0) : Vec(0, 1, 0);
    return Hit{t, p, n, m_e, m_c, m_mat};
  }
};

struct Quad_YZ : public Hitable {
  double m_y0, m_y1, m_z0, m_z1;  // extents along Y and Z
  double d;  // distance from origin along X axis
  Vec m_e, m_c;   // emission, color
  Material m_mat; // reflection type (DIFFuse, SPECular, REFRactive
  bool m_flip;

  Quad_YZ(double y0, double y1, double z0, double z1, double dist, Vec e, Vec c, Material mat, bool flip=false):
    m_y0(y0), m_y1(y1), m_z0(z0), m_z1(z1), d(dist), m_e(e), m_c(c), m_mat(mat), m_flip(flip) {}

  Hit intersect(const Ray &ray) const {
    double t = (d - ray.m_o.x) / ray.m_v.x; // solve for intersection with plane at x=d
    if (t <= 0 || t > 1e10) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), m_mat};

    Vec p = ray.m_o + ray.m_v * t;
    if (p.y < m_y0 || p.y > m_y1 || p.z < m_z0 || p.z > m_z1) return Hit{0.0, Vec(), Vec(), Vec(), Vec(), m_mat};

    Vec n = m_flip ? Vec(-1, 0, 0) : Vec(1, 0, 0);
    return Hit{t, p, n, m_e, m_c, m_mat};
  }
};

auto create_cornell(int count, auto &objects) {
  const double zShift = -6.5;
  const double xShift = -1.0;

  objects.emplace_back(std::make_unique<Sphere>(10000.0, Vec(0, -10000, 0), Vec(), Vec(0.3, 0.90, 0.25), DIFF));

  // Cornell box walls
  objects.push_back(std::make_unique<Quad_YZ>(0, 2, zShift + 0, zShift + 2, xShift + 0, Vec(0), Vec(0.75, 0.25, 0.25), DIFF, false)); // Left wall (faces +X)
  objects.push_back(std::make_unique<Quad_YZ>(0, 2, zShift + 0, zShift + 2, xShift + 2, Vec(0), Vec(0.25, 0.25, 0.75), DIFF, true));  // Right wall (faces -X)
  objects.push_back(std::make_unique<Quad_XZ>(xShift + 0, xShift + 2, zShift + 0, zShift + 2, 0, Vec(0), Vec(0.75, 0.75, 0.75), DIFF, false)); // Floor (faces +Y)
  objects.push_back(std::make_unique<Quad_XZ>(xShift + 0, xShift + 2, zShift + 0, zShift + 2, 2, Vec(0), Vec(0.75, 0.75, 0.75), DIFF, true));  // Ceiling (faces -Y)
  objects.push_back(std::make_unique<Quad_XY>(xShift + 0, xShift + 2, 0, 2, zShift, Vec(0), Vec(0.75, 0.75, 0.75), DIFF, true)); // Back wall (faces -Z)
  objects.push_back(std::make_unique<Quad_XZ>(xShift + 0.8, xShift + 1.2, zShift + 0.8, zShift + 1.2, 1.99, Vec(5, 5, 5), Vec(0), LIGHT, true));  // Ceiling (faces -Y)

  // Two spheres resting on the floor inside the Cornell box.
  objects.push_back(std::make_unique<Sphere>(0.35, Vec(xShift + 0.65, 0.35, zShift + 0.6), Vec(), Vec(0.95, 0.95, 0.95), DIFF));
  objects.push_back(std::make_unique<Sphere>(0.5, Vec(xShift + 1.35, 0.5, zShift + 1.4), Vec(), Vec(0.95, 0.95, 0.95), DIFF));

  return 3.0 / 3.3; // aspect ratio
}

auto create_spheres(int count, auto &objects) {
  // Ground and a simple overhead light.
  objects.emplace_back(std::make_unique<Sphere>(10000.0, Vec(0, -10000, 0), Vec(), Vec(0.3, 0.90, 0.25), DIFF));
  objects.emplace_back(std::make_unique<Sphere>(1.0, Vec(10, 10, 0), Vec(5, 5, 5), Vec(), LIGHT));
  objects.emplace_back(std::make_unique<Sphere>(0.4, Vec(0, 0.4, -5), Vec(), Vec(0.8, 0.8, 0.8), DIFF));

  for (int i = 0; i < count; ++i) {
    double x = (rand01() * 2.0 - 1.0) * 5.0;
    double z = -5.0 - rand01() * 12.0;
    double t = (z - (-5.0)) / (-21.0 - (-5.0)); // 0 near, 1 far
    double sizeJitter = 0.3 + rand01() * 0.9;
    double radius = (0.15 + t * 1.35) * sizeJitter;
    Vec pos(x, radius, z);
    Vec color(0.2 + rand01() * 0.8, 0.2 + rand01() * 0.8, 0.2 + rand01() * 0.8);
    objects.emplace_back(std::make_unique<Sphere>(radius, pos, Vec(), color, DIFF));
  }

  double aspect = 16.0 / 9.0;
  return aspect;
}


bool russian_roulette(const Vec &throughput, int depth) {
  if (depth < 5) return true;
  double p = glm::max(throughput.x, glm::max(throughput.y, throughput.z));
  return rand01() < p;
}


Vec radiance(const auto &objects, const Ray &r) {
  Ray ray = r;
  int curDepth = 0; // recursion depth
  Vec tp = Vec(1.0); // throughput
  Vec radianceSum = Vec(0.0);

  for (;;) {
    if(!russian_roulette(tp, curDepth)) { 
       return radianceSum;
    }

    Hit best{.t = 0.0};
    bool found = false;
    for (const std::unique_ptr<Hitable> &obj : objects) {
      Hit h = obj->intersect(ray);
      if (h.t > ray.m_tmin && h.t < ray.m_tmax && (!found || h.t < best.t)) { best = h; found = true; }
    }
    if (!found) {
      // Simple sky gradient based on ray direction.
      double tSky = 0.5 * (ray.m_v.y + 1.0);
      tSky = tSky * tSky;
      Vec sky = Vec(1.0, 1.0, 1.0) * (1.0 - tSky) + Vec(0.2, 0.4, 1.0) * tSky;
      return radianceSum + tp * sky;
    }

    Vec w = glm::dot(best.n, ray.m_v) < 0 ? best.n : -best.n;
    if (curDepth >= 20 || best.m_mat == LIGHT) return radianceSum + tp * best.e;

    radianceSum += tp * best.e;

    // Cosine-weighted hemisphere sampling around the normal.
    double r1 = 2.0 * M_PI * rand01();
    double r2 = rand01();
    double r2s = sqrt(r2);
    Vec u = glm::normalize(glm::cross(fabs(w.x) > 0.1 ? Vec(0, 1, 0) : Vec(1, 0, 0), w));
    Vec v = glm::cross(w, u);
    Vec d = glm::normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2));

    ray = Ray(best.p + 1e-4 * w, d);
    tp = tp * best.c;
    ++curDepth;
  }
}


int main(int argc, char *argv[]){
  int samples = argc==2 ? atoi(argv[1]) : 256; // # samples
  double div{1./samples};

  std::vector<std::unique_ptr<Hitable>> objects;
  //double aspect = create_spheres(10, objects);
  double aspect = create_cornell(0, objects);;

  int w{1024}, h{(int)(w / aspect)}; // image resolution
  Camera cam(Vec(0,1,0), Vec(0,0,-1), Vec(0,1,0), w, h);
  std::vector<Vec> colors;
  colors.resize(w * h);
  Ray ray;

#pragma omp parallel for schedule(dynamic, 8) private(ray)       // OpenMP

  for (int y=0; y<h; y++) {                       // Loop over image rows
    int percent = static_cast<int>(100.0 * y / (h - 1));
    std::cout << "\r\033Rendering (" << samples << " spp) " << percent << "%            " << std::flush;

    for (unsigned short x=0; x<w; x++) { // Loop cols
      Vec sum(0);
      for(int s=0; s<samples; s++) {
        int hIndex = s + 1;
        ray = cam.ray(x + halton(hIndex, 2), y + halton(hIndex, 3));
        sum = sum + radiance(objects, ray);
      }
      colors[x + y * w] = clamp(sum * div) * 255.0;
    }
  }

  std::ofstream f("image.ppm");         // Write image to PPM file.
  f << "P3\n" << w << " " << h << "\n" << 255 << "\n";
  for (int i=0; i<w*h; i++)
    f << int(colors[i].x) << " " << int(colors[i].y) << " " << int(colors[i].z) << " ";
}
