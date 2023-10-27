#include <chrono>
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>
#include <cassert>

//assume square grid NxNxN representing area [0,L]x[0,L]x[0,L]
#define PI 3.1415926535897932384

static int N = 0;
static int Np = 0;
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
    assert(((long)Np)*((long)Np)*((long)Np) < INT32_MAX);
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
private:
  double *data = nullptr;
};

double max_diff(Grid &g1, Grid &g2)
{
  unsigned sz = Np*Np*Np;
  double max_diff = 0;
  std::vector<unsigned> max_pos;
  for (unsigned i = 0;i<sz;i++)
    max_diff = std::max(max_diff, std::abs(g1.get_data()[i] - g2.get_data()[i]));
  /*
  max_diff = 0;
  for (unsigned i=0;i<=N;i++)
    for (unsigned j=0;j<=N;j++)
      for (unsigned k=0;k<=N;k++)
      {
        double d = std::abs(g1.at(i,j,k) - g2.at(i,j,k));
        if (d > max_diff)
        {
          max_diff = d;
          max_pos = std::vector<unsigned>{i,j,k};
        }
      }
  printf("%lg %u %u %u -- ", max_diff, max_pos[0], max_pos[1], max_pos[2]);
  printf("%lg %lg\n",g1.at(max_pos[0], max_pos[1], max_pos[2]), g2.at(max_pos[0], max_pos[1], max_pos[2]));
  */
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

inline double delta_h()
{

}

int main(int argc, char **argv)
{
  assert(argc == 5);
  N = std::stoi(std::string(argv[1]));
  Lx = std::stod(std::string(argv[2]));
  Ly = std::stod(std::string(argv[3]));
  Lz = std::stod(std::string(argv[4]));

  assert(Lx == Ly);
  assert(Lx == Lz);

  h = Lx/N;
  Np = N+1;

  //???
  //0.2*h*h*h to make it stable actually..
  tau = 0.001;

  constexpr unsigned steps = 20;

  Grid reference_grid;
  Grid ring[3];
  fill_reference_grid(ring[0], 0*tau);
  fill_reference_grid(ring[1], 1*tau);
  unsigned cur_grid = 2;
  for (unsigned step = 2; step<steps; step++)
  {
    fill_reference_grid(reference_grid, step*tau);

    Grid &P1 = ring[(3 + cur_grid - 1) % 3];
    Grid &P2 = ring[(3 + cur_grid - 2) % 3];

    ring[cur_grid].clear_and_fill_borders(0);
    for (unsigned i=1;i<N;i++)
      for (unsigned j=1;j<N;j++)
        for (unsigned k=1;k<N;k++)
          ring[cur_grid].at(i,j,k) = 2*P1.at(i,j,k) - P2.at(i,j,k) + (tau*tau*a_squared()/(h*h)) * \
          (-6*P1.at(i,j,k) + P1.at(i-1,j,k) + P1.at(i+1,j,k) + P1.at(i,j-1,k) + P1.at(i,j+1,k) + P1.at(i,j,k-1) + P1.at(i,j,k+1));

    double diff = max_diff(ring[cur_grid], reference_grid);
    printf("%u: max diff %lg\n", step, diff);
    cur_grid = (cur_grid + 1)%3;
  }

  return 0;
}