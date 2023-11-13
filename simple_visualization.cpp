#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <array>
#include <fstream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../grade/third_party/third_party/stb_image.h"
#include "../grade/third_party/third_party/stb_image_write.h"

//assume square grid NxNxN representing area [0,L]x[0,L]x[0,L]
#define PI 3.1415926535897932384

static unsigned N = 0;
static unsigned Np = 0;
static double Lx = 0;
static double Ly = 0;
static double Lz = 0;
static double h = 0;
static double tau = 0;

class Grid
{
public:
  Grid()
  {
    assert(((long)Np)*((long)Np)*((long)Np) < 1<<30);
    try
    {
      data = new double[Np*Np*Np];
    }
    catch (const std::exception& e)
    {
      fprintf(stderr, "Failed to allocate grid with size %d: %s\n",N, e.what());
    }
    
  }
  void clear_and_fill_borders(double val)
  {
    std::fill_n(data, Np*Np*Np, val);
  }
  inline double &at(unsigned i, unsigned j, unsigned k)
  {
    return data[i*Np*Np + j*Np + k];
  }
  const double *get_data() const
  {
    return data;
  }
  ~Grid()
  {
    delete data;
  }

  double *data;
};

double max_diff(Grid &g1, Grid &g2)
{
  unsigned sz = Np*Np*Np;
  double max_diff = 0;
  std::vector<unsigned> max_pos;
  for (unsigned i = 0;i<sz;i++)
    max_diff = std::max(max_diff, std::abs(g1.get_data()[i] - g2.get_data()[i]));
  return max_diff;
}

inline double u_analytical(double x, double y, double z, double t)
{
  return sin(PI*x/Lx)*sin(PI*y/Ly)*sin(PI*z/Lz)*cos(PI*t);
}

inline double a_squared()
{
  return (Lx*Lx*Ly*Ly*Lz*Lz)/(Lx*Lx + Ly*Ly + Lz*Lz);
}

void fill_reference_grid(Grid &g, double t)
{
  for (unsigned i=0;i<=N;i++)
    for (unsigned j=0;j<=N;j++)
      for (unsigned k=0;k<=N;k++)
        g.at(i,j,k) = u_analytical(i*h,j*h,k*h, t);
}

float sample_trilinear(Grid &g, float fx, float fy, float fz)
{
  int x = fx;
  int y = fy;
  int z = fz;
  float dx = fx - x;
  float dy = fy - y;
  float dz = fz - z;
  #define C(i, j, k) g.at(std::min(x + i, (int)Np - 1), std::min(y + j, (int)Np - 1), std::min(z + k, (int)Np - 1))
  float c00 = C(0, 0, 0) * (1 - dx) + C(1, 0, 0) * dx;
  float c01 = C(0, 0, 1) * (1 - dx) + C(1, 0, 1) * dx;
  float c10 = C(0, 1, 0) * (1 - dx) + C(1, 1, 0) * dx;
  float c11 = C(0, 1, 1) * (1 - dx) + C(1, 1, 1) * dx;

  float c0 = c00 * (1 - dy) + c10 * dy;
  float c1 = c01 * (1 - dy) + c11 * dy;

  float c = c0 * (1 - dz) + c1 * dz;

  return c;
}

void set_color(float f, float min, float max, char *color)
{
  f = (std::min(std::max(f, min), max) - min)/(max - min);//to [0,1]
  f = 2*f - 1;//to [-1,1]
  color[0] = f > 0 ? 255*f : 0;
  color[1] = 0;
  color[2] = f < 0 ? -255*f : 0;
}

void visualize_grid(const std::array<float,12> &plane, Grid &g, int w, int h, const char *filename, float min, float max)
{
  char *data = new char[3*w*h];

  for (int i=0;i<h;i++)
  {
    for (int j=0;j<w;j++)
    {
      float dy = (float)i/h;
      float dx = (float)j/w;
      float x = (plane[0]*dx + plane[3]*(1-dx))*dy + (plane[6]*dx + plane[9]*(1-dx))*(1-dy);
      float y = (plane[1]*dx + plane[4]*(1-dx))*dy + (plane[7]*dx + plane[10]*(1-dx))*(1-dy);
      float z = (plane[2]*dx + plane[5]*(1-dx))*dy + (plane[8]*dx + plane[11]*(1-dx))*(1-dy);
      set_color(sample_trilinear(g, x,y,z), min, max, data + 3*(i*w + j));
    }
  }

  stbi_write_png(filename, w, h, 3, data, 0);
  delete[] data;
}

//256 grid, 20 steps
//simple grid, no openmp - 540-560 ms
//simple grid, openmp - 440-450 ms

void save(Grid &g, const std::string& filename) 
{
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open file for writing.");
        }

        file << "Nx: " << Np << " Ny: " << Np << " Nz: " << Np << "\n";
        for (int i = 0; i < Np; ++i) {
            for (int j = 0; j < Np; ++j) {
                for (int k = 0; k < Np; ++k) {
                    file << g.at(i, j, k) << ",";
                }
            }
        }

        file.close();
    }

int main(int argc, char **argv)
{
  float init_ms = 0;
  float solve_ms = 0;
  float compare_ms = 0;
  bool compare = true;
  bool visualize = true;

  assert(argc == 5);
  N = atoi(argv[1]) - 1;
  Lx = atoi(argv[2]);
  Ly = atoi(argv[3]);
  Lz = atoi(argv[4]);

  assert(Lx == Ly);
  assert(Lx == Lz);

  h = Lx/N;
  Np = N+1;

  //???
  //0.2*h*h*h to make it stable actually..
  tau = h/15;

  const unsigned steps = 21;

  Grid reference_grid;
  Grid diff_grid;
  Grid ring[3];

  fill_reference_grid(ring[0], 0*tau);
  //fill_reference_grid(ring[1], 1*tau);
  {
    Grid &P1 = ring[0];
    double c = (tau*tau*a_squared()/(h*h));
    for (unsigned i=1;i<N;i++)
      for (unsigned j=1;j<N;j++)
        for (unsigned k=1;k<N;k++)
          {
            double p1 = P1.at(i,j,k);
            double p3 = P1.at(i-1,j,k);
            double p4 = P1.at(i+1,j,k);
            double p5 = P1.at(i,j-1,k);
            double p6 = P1.at(i,j+1,k);
            double p7 = P1.at(i,j,k-1);
            double p8 = P1.at(i,j,k+1);
            ring[1].at(i,j,k) = p1 +0.5*tau*tau*(-6*p1 +p3+p4+p5+p6+p7+p8);
          }
  }
  ring[2].clear_and_fill_borders(0);

  unsigned cur_grid = 2;
  for (unsigned step = 2; step<steps; step++)
  {
    Grid &P1 = ring[(3 + cur_grid - 1) % 3];
    Grid &P2 = ring[(3 + cur_grid - 2) % 3];
    double c = (tau*tau*a_squared()/(h*h));
    for (unsigned i=1;i<N;i++)
      for (unsigned j=1;j<N;j++)
        for (unsigned k=1;k<N;k++)
          {
            double p1 = P1.at(i,j,k);
            double p2 = P2.at(i,j,k);
            double p3 = P1.at(i-1,j,k);
            double p4 = P1.at(i+1,j,k);
            double p5 = P1.at(i,j-1,k);
            double p6 = P1.at(i,j+1,k);
            double p7 = P1.at(i,j,k-1);
            double p8 = P1.at(i,j,k+1);
            ring[cur_grid].at(i,j,k) = 2*p1 - p2 + c*(-6*p1 +p3+p4+p5+p6+p7+p8);
          }
    if (compare)
    {
      fill_reference_grid(reference_grid, step*tau);
      double diff = max_diff(ring[cur_grid], reference_grid);
      printf("%u: max diff %lg\n", step, diff);
    }
    if (visualize)
    {
      for (unsigned i = 0;i<Np*Np*Np;i++)
        diff_grid.data[i] = reference_grid.data[i] - ring[cur_grid].data[i]; 
      std::array<float,12> plane = {0,0,0, (float)N,(float)N,0, 0,0,(float)N, (float)N,(float)N,(float)N};
      char buf[1024];
      save(ring[cur_grid], "res_matrix_"+std::to_string(step)+".txt");
      save(reference_grid, "ref_matrix_"+std::to_string(step)+".txt");
      save(diff_grid, "diff_matrix_"+std::to_string(step)+".txt");
      //sprintf(buf, "diff_grid_%d.png", step);
      //visualize_grid(plane, diff_grid, 1024*sqrt(2), 1024, buf, -1e-5, 1e-5);
      //sprintf(buf, "reference_grid_%d.png", step);
      //visualize_grid(plane, reference_grid, 1024*sqrt(2), 1024, buf, -1, 1);
      //sprintf(buf, "result_grid_%d.png", step);
      //visualize_grid(plane, ring[cur_grid], 1024*sqrt(2), 1024, buf, -1, 1);
    }

    cur_grid = (cur_grid + 1)%3;
  }

  //printf("took %.3f ms (%.3f + %.3f + %.3f)\n", init_ms + compare_ms + solve_ms, init_ms, solve_ms, compare_ms);

  return 0;
}