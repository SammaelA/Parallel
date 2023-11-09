#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <omp.h>
#include <array>

#define THREADS_NB 8

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

  }
  void init( int _x0, int _y0, int _z0, int _xn, int _yn, int _zn)
  {
    x0 = _x0;
    y0 = _y0;
    z0 = _z0;
    xn = _xn;
    yn = _yn;
    zn = _zn;
    printf("[%d %d %d] - %d %d %d\n",x0,y0,z0,xn,yn,zn);

    try
    {
      data = new double[xn*yn*zn];
    }
    catch (const std::exception& e)
    {
      fprintf(stderr, "Failed to allocate grid with size %dx%dx%d: %s\n",xn,yn,zn, e.what());
    }
    
  }
  void clear_and_fill_borders(double val)
  {
    std::fill_n(data, xn*yn*zn, val);
  }
  inline double &at(int i, int j, int k)
  {
    return data[i*yn*zn + j*zn + k];
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
  int x0,y0,z0;
  int xn,yn,zn;
};

double max_diff(Grid &g1, Grid &g2)
{
  int sz = g1.xn*g1.yn*g1.zn;
  assert(sz == g2.xn*g2.yn*g2.zn);
  double max_diff = 0;
  std::vector<int> max_pos;
  for (int i = 0;i<sz;i++)
    max_diff = std::max(max_diff, std::abs(g1.get_data()[i] - g2.get_data()[i]));

  max_diff = 0;
  for (int i=0;i<g1.xn;i++)
    for (int j=0;j<g1.yn;j++)
      for (int k=0;k<g1.zn;k++)
      {
        double d = std::abs(g1.at(i,j,k) - g2.at(i,j,k));
        if (d > max_diff)
        {
          max_diff = d;
          max_pos = std::vector<int>{i,j,k};
        }
      }
  printf("%lg %lg %lg %lg -- ", max_diff, max_pos[0]*h, max_pos[1]*h, max_pos[2]*h);
  printf("%lg %lg\n",g1.at(max_pos[0], max_pos[1], max_pos[2]), g2.at(max_pos[0], max_pos[1], max_pos[2]));
  
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
  #pragma omp parallel for num_threads(THREADS_NB)
  for (int i=0;i<g.xn;i++)
    for (int j=0;j<g.yn;j++)
      for (int k=0;k<g.zn;k++)
        g.at(i,j,k) = u_analytical((i+g.x0)*h,(j+g.y0)*h,(k+g.z0)*h, t);
}

template <int max_depth>
std::vector<std::vector<uint16_t>> get_all_factors(uint16_t n, uint16_t max_fact, int depth)
{
  if (depth >= max_depth-1)
  {
    if (n > max_fact)
      return {};
    else
      return {{n}};
  }
  if (n==1)
  {
    return {std::vector<uint16_t>(max_depth - depth, 1)};
  }
  else
  {
    std::vector<std::vector<uint16_t>> factors;
    for (uint16_t i=2;i<=max_fact;i++) 
    {
      if (n % i == 0)
      {
        auto ch_factors = get_all_factors<max_depth>(n/i, i, depth+1);
        for (auto &v : ch_factors)
        {
          v.push_back(i);
          factors.push_back(v);
        }
      }
    }
    return factors;
  }
}

void get_best_blocks_layout(int n_blocks, int *blocks_x, int *blocks_y, int *blocks_z)
{
  auto res = get_all_factors<3>(n_blocks, n_blocks, 0);
  *blocks_x = res[0][0];
  *blocks_y = res[0][1];
  *blocks_z = res[0][2];
}

int main(int argc, char **argv)
{
  float init_ms = 0;
  float solve_ms = 0;
  float compare_ms = 0;
  bool compare = true;

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

  const int steps = 21;

  std::vector<std::array<Grid, 3>> ring_buffers;
  int blocks = THREADS_NB;
  int blocks_x=4, blocks_y=4, blocks_z=4;
  get_best_blocks_layout(blocks, &blocks_x, &blocks_y, &blocks_z);
  ring_buffers.resize(blocks);
  for (int i=0;i<blocks_x;i++)
  {
    for (int j=0;j<blocks_y;j++)
    {
      for (int k=0;k<blocks_z;k++)
      {
        for (int bn = 0; bn < 3; bn++)
        {
          int bst_x = std::min(std::max(0, i*(Np/blocks_x) - 1), Np);
          int bst_y = std::min(std::max(0, j*(Np/blocks_y) - 1), Np);
          int bst_z = std::min(std::max(0, k*(Np/blocks_z) - 1), Np);
          int ben_x = std::min(std::max(0, (i+1)*(Np/blocks_x) + 1), Np);
          int ben_y = std::min(std::max(0, (j+1)*(Np/blocks_y) + 1), Np);
          int ben_z = std::min(std::max(0, (k+1)*(Np/blocks_z) + 1), Np);
          printf("%d %d %d %d %d %d\n",bst_x, bst_y, bst_z, ben_x, ben_y, ben_z );
          ring_buffers[i*blocks_y*blocks_z + j*blocks_z + k][bn].init(
            bst_x, bst_y, bst_z, ben_x - bst_x, ben_y - bst_y, ben_z - bst_z);
        }
      }      
    }
  }
  clock_t t1 = clock();
  #pragma omp parallel for num_threads(THREADS_NB)
  for (int i=0;i<blocks;i++)
  {
    fill_reference_grid(ring_buffers[i][0], 0*tau);
    fill_reference_grid(ring_buffers[i][1], 1*tau);
    ring_buffers[i][2].clear_and_fill_borders(0);
  }
  /*
  {
    Grid &P1 = ring[0];
    double c = (tau*tau*a_squared()/(h*h));
    #pragma omp parallel for num_threads(THREADS_NB)
    for (int i=1;i<N;i++)
      for (int j=1;j<N;j++)
        for (int k=1;k<N;k++)
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
  */

  clock_t t2 = clock();
  init_ms = 1000*(double)(t2 - t1) / (CLOCKS_PER_SEC * THREADS_NB);

  int cur_grid = 2;
  for (int step = 2; step<3; step++)
  {
    t1 = clock();
    #pragma omp parallel for num_threads(THREADS_NB)
    for (int block_n=0;block_n<blocks;block_n++)
    {
      Grid &P1 = ring_buffers[block_n][(3 + cur_grid - 1) % 3];
      Grid &P2 = ring_buffers[block_n][(3 + cur_grid - 2) % 3];
      Grid &P_res = ring_buffers[block_n][cur_grid];
      double c = (tau * tau * a_squared() / (h * h));

      for (int i = 1; i < P_res.xn - 1; i++)
        for (int j = 1; j < P_res.yn - 1; j++)
          for (int k = 1; k < P_res.zn - 1; k++)
          {
            double p1 = P1.at(i, j, k);
            double p2 = P2.at(i, j, k);
            double p3 = P1.at(i - 1, j, k);
            double p4 = P1.at(i + 1, j, k);
            double p5 = P1.at(i, j - 1, k);
            double p6 = P1.at(i, j + 1, k);
            double p7 = P1.at(i, j, k - 1);
            double p8 = P1.at(i, j, k + 1);
            P_res.at(i, j, k) = 2 * p1 - p2 + c * (-6 * p1 + p3 + p4 + p5 + p6 + p7 + p8);
            printf("[%d]%d %d %d %lg %lg %lg %lg\n",block_n,i,j,k, P_res.at(i, j, k), p1, p2, p3);
          }
    }
    t2 = clock();
    solve_ms += 1000*(double)(t2 - t1) / (CLOCKS_PER_SEC * THREADS_NB);
    
    //transfer
    for (int i=0;i<blocks_x;i++)
    {
      for (int j=0;j<blocks_y;j++)
      {
        for (int k=0;k<blocks_z;k++)
        {
          Grid &S = ring_buffers[i*blocks_y*blocks_z + j*blocks_z + k][cur_grid];
          if (i != 0)
          {
            Grid &R = ring_buffers[(i-1)*blocks_y*blocks_z + j*blocks_z + k][cur_grid];
            printf("(%d %d %d) %d %d   %d %d   %d %d\n", i,j,k, S.xn, R.xn, S.yn, R.yn, S.zn, R.zn);
            
            assert(S.yn == R.yn && S.zn == R.zn);
            for (int y = 0; y < S.yn; y++)
              for (int z = 0; z < S.zn; z++)
                R.data[(R.xn-1)*S.yn*S.zn + y*S.zn + z] = S.data[(1)*S.yn*S.zn + y*S.zn + z];
          }
          if (i != blocks_x - 1)
          {
            Grid &R = ring_buffers[(i+1)*blocks_y*blocks_z + j*blocks_z + k][cur_grid];
            printf("(%d %d %d) %d %d   %d %d   %d %d\n", i,j,k, S.xn, R.xn, S.yn, R.yn, S.zn, R.zn);
            
            assert(S.yn == R.yn && S.zn == R.zn);
            for (int y = 0; y < S.yn; y++)
              for (int z = 0; z < S.zn; z++)
                R.data[(0)*S.yn*S.zn + y*S.zn + z] = S.data[(S.xn - 2)*S.yn*S.zn + y*S.zn + z];          
          }

          if (j != 0)
          {
            Grid &R = ring_buffers[i*blocks_y*blocks_z + (j-1)*blocks_z + k][cur_grid];
            printf("(%d %d %d) %d %d   %d %d   %d %d\n", i,j,k, S.xn, R.xn, S.yn, R.yn, S.zn, R.zn);
            
            assert(S.xn == R.xn && S.zn == R.zn);
            for (int x = 0; x < S.xn; x++)
              for (int z = 0; z < S.zn; z++)
                R.data[x*S.yn*S.zn + (R.yn-1)*S.zn + z] = S.data[x*S.yn*S.zn + (1)*S.zn + z];
          }
          if (j != blocks_y - 1)
          {
            Grid &R = ring_buffers[i*blocks_y*blocks_z + (j+1)*blocks_z + k][cur_grid];
            printf("(%d %d %d) %d %d   %d %d   %d %d\n", i,j,k, S.xn, R.xn, S.yn, R.yn, S.zn, R.zn);
            
            assert(S.xn == R.xn && S.zn == R.zn);
            for (int x = 0; x < S.xn; x++)
              for (int z = 0; z < S.zn; z++)
                R.data[x*S.yn*S.zn + (0)*S.zn + z] = S.data[x*S.yn*S.zn + (S.yn-2)*S.zn + z];          
          }

          if (k != 0)
          {
            Grid &R = ring_buffers[i*blocks_y*blocks_z + j*blocks_z + k-1][cur_grid];
            printf("(%d %d %d) %d %d   %d %d   %d %d\n", i,j,k, S.xn, R.xn, S.yn, R.yn, S.zn, R.zn);
            assert(S.xn == R.xn && S.yn == R.yn);
            for (int x = 0; x < S.xn; x++)
              for (int y = 0; y < S.yn; y++)
                R.data[x*S.yn*S.zn + y*S.zn + (R.zn-1)] = S.data[x*S.yn*S.zn + y*S.zn + (1)];
          }
          if (k != blocks_z - 1)
          {
            Grid &R = ring_buffers[i*blocks_y*blocks_z + j*blocks_z + k+1][cur_grid];
            printf("(%d %d %d) %d %d   %d %d   %d %d\n", i,j,k, S.xn, R.xn, S.yn, R.yn, S.zn, R.zn);
            assert(S.xn == R.xn && S.yn == R.yn);
            for (int x = 0; x < S.xn; x++)
              for (int y = 0; y < S.yn; y++)
                R.data[x*S.yn*S.zn + y*S.zn + (0)] = S.data[x*S.yn*S.zn + y*S.zn + (S.zn-2)];          
          }
        }      
      }
    }

    if (compare)
    {
      double diff = 0;
      t1 = clock();
      ///#pragma omp parallel for num_threads(THREADS_NB)
      for (int block_n=0;block_n<blocks;block_n++)
      {
      fill_reference_grid(ring_buffers[block_n][(cur_grid + 1)%3], step*tau);
      diff = std::max(diff, max_diff(ring_buffers[block_n][cur_grid], ring_buffers[block_n][(cur_grid + 1)%3]));
      }
      printf("%d: max diff %lg\n", step, diff);
      t2 = clock();
      compare_ms += 1000*(double)(t2 - t1) / (CLOCKS_PER_SEC * THREADS_NB);
    }

    cur_grid = (cur_grid + 1)%3;
  }

  printf("took %.3f ms (%.3f + %.3f + %.3f)\n", init_ms + compare_ms + solve_ms, init_ms, solve_ms, compare_ms);

  return 0;
}