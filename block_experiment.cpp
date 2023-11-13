#include <chrono>
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>
#include <cassert>

//assume square grid NxNxN representing area [0,L]x[0,L]x[0,L]
#define PI 3.1415926535897932384
#define BS 32

static int N = 0;
static int Np = 0;
static int Nb = 0;
static double Lx = 0;
static double Ly = 0;
static double Lz = 0;
static double h = 0;
static double tau = 0;

class SimpleGrid
{
public:
  SimpleGrid()
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
  inline double &at(int i, int j, int k)
  {
    return data[i*Np*Np + j*Np + k];
  }
  const double *get_data() const
  {
    return data;
  }
  ~SimpleGrid()
  {
    delete data;
  }
private:
  double *data = nullptr;
};

class BlockGrid
{
public:
  //static constexpr int BS = 4;
  BlockGrid()
  {
    assert((Np/BS)*BS == Np);
    assert(((long)Np)*((long)Np)*((long)Np) < INT32_MAX);
    try
    {
      data_raw = new double[Np*Np*Np + 2*BS*Np*Np];
      data = data_raw;
    }
    catch (const std::exception& e)
    {
      fprintf(stderr, "Failed to allocate grid with size %d: %s\n",N, e.what());
    }
    
  }
  void clear_and_fill_borders(double val)
  {
    std::fill_n(data_raw, Np*Np*Np + 2*BS*Np*Np, val);
  }
  inline double &at(int i, int j, int k)
  {
    return data[((i/BS)*Nb*Nb + (j/BS)*Nb + k/BS)*(BS*BS*BS) + (i%BS)*BS*BS + (j%BS)*BS + k%BS];
  }
  const double *get_data() const
  {
    return data;
  }
  ~BlockGrid()
  {
    delete[] data_raw;
  }
//private:
  double *data_raw = nullptr;
  double *data = nullptr;
};

using Grid = BlockGrid;

int get_p(int i, int j, int k)
{
  return ((i/BS)*Nb*Nb + (j/BS)*Nb + k/BS)*(BS*BS*BS) + (i%BS)*BS*BS + (j%BS)*BS + k%BS;
}

double max_diff(Grid &g1, Grid &g2)
{
  int sz = Np*Np*Np;
  double max_diff = 0;
  std::vector<int> max_pos;
  #pragma omp parallel for
  for (int i = 0;i<sz;i++)
    max_diff = std::max(max_diff, std::abs(g1.get_data()[i] - g2.get_data()[i]));
  
  max_diff = 0;
  for (int i=0;i<N;i++)
    for (int j=0;j<N;j++)
      for (int k=0;k<N;k++)
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
  #pragma omp parallel for
  for (int i=0;i<=N;i++)
    for (int j=0;j<=N;j++)
      for (int k=0;k<=N;k++)
        g.at(i,j,k) = u_analytical(i*h,j*h,k*h, t);
}

//256 grid, 20 steps
//simple grid, no openmp - 540-560 ms
//simple grid, openmp - 440-450 ms

int main(int argc, char **argv)
{
  float init_ms = 0;
  float solve_ms = 0;
  float compare_ms = 0;
  bool compare = true;

  assert(argc == 5);
  N = std::stoi(std::string(argv[1])) - 1;
  Lx = std::stod(std::string(argv[2]));
  Ly = std::stod(std::string(argv[3]));
  Lz = std::stod(std::string(argv[4]));

  assert(Lx == Ly);
  assert(Lx == Lz);

  h = Lx/N;
  Np = N+1;
  Nb = Np/BS;

  //???
  //0.2*h*h*h to make it stable actually..
  tau = 1.0/(128*15);

  constexpr int steps = 21;

  Grid reference_grid;
  Grid ring[3];
  alignas(64) int index_offset[7*BS*BS*BS];
  for (int i=0;i<BS;i++)
    for (int j=0;j<BS;j++)
      for (int k=0;k<BS;k++)
      {
        index_offset[7*(i*BS*BS + j*BS + k)+0] = get_p(BS+i,BS+j,BS+k) - get_p(BS,BS,BS);
        index_offset[7*(i*BS*BS + j*BS + k)+1] = get_p(BS+i-1,BS+j,BS+k) - get_p(BS,BS,BS);
        index_offset[7*(i*BS*BS + j*BS + k)+2] = get_p(BS+i+1,BS+j,BS+k) - get_p(BS,BS,BS);
        index_offset[7*(i*BS*BS + j*BS + k)+3] = get_p(BS+i,BS+j-1,BS+k) - get_p(BS,BS,BS);
        index_offset[7*(i*BS*BS + j*BS + k)+4] = get_p(BS+i,BS+j+1,BS+k) - get_p(BS,BS,BS);
        index_offset[7*(i*BS*BS + j*BS + k)+5] = get_p(BS+i,BS+j,BS+k-1) - get_p(BS,BS,BS);
        index_offset[7*(i*BS*BS + j*BS + k)+6] = get_p(BS+i,BS+j,BS+k+1) - get_p(BS,BS,BS);
      }

  auto t1 = std::chrono::steady_clock::now();
  fill_reference_grid(ring[0], 0*tau);
  ring[1].clear_and_fill_borders(0);
  {
    Grid &P1 = ring[0];
    #pragma omp parallel for
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
    //for (int index =0; index<7*BS*BS*BS; index+=1)
    //  printf("index[%u] = %d\n", index, index_offset[index]);
  }
  //fill_reference_grid(ring[1], 1*tau);
  ring[2].clear_and_fill_borders(0);
  auto t2 = std::chrono::steady_clock::now();
  init_ms = 1e-3*std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

  int cur_grid = 2;
  for (int step = 2; step<steps; step++)
  {
    Grid &P1 = ring[(3 + cur_grid - 1) % 3];
    Grid &P2 = ring[(3 + cur_grid - 2) % 3];
    double c = (tau*tau*a_squared()/(h*h));
    t1 = std::chrono::steady_clock::now();
    //ring[cur_grid].clear_and_fill_borders(0);
    #pragma omp parallel for
    for (int bi = 0; bi < Nb; bi++)
    {
      for (int bj = 0; bj < Nb; bj++)
      {
        for (int bk = 0; bk < Nb; bk++)
        {
          //printf("bi = %d\n",bi);
          int base_index = (bi * Nb * Nb + bj * Nb + bk) * (BS * BS * BS);
          if ((bi == 0) || (bj == 0) || (bk == 0))
          {
            int i0 = (bi == 0);
            int j0 = (bj == 0);
            int k0 = (bk == 0);
            for (int i = i0; i < BS; i++)
            {
              for (int j = j0; j < BS; j++)
              {
                for (int k = k0; k < BS; k++)
                {
                  int index = 7 * (i * BS * BS + j * BS + k);
                  //int a = base_index + index_offset[index];
                  //int b = get_p(BS * bi + i, BS * bj + j, BS * bk + k);
                  //if (BS * bi + i + 1 == 128 && P1.data[get_p(BS * bi + i+1, BS * bj + j, BS * bk + k)] > 1e-12)
                  //  printf("(%d %d %d) %f \n",BS * bi + i+1,BS * bj + j,BS * bk + k, P1.data[get_p(BS * bi + i+1, BS * bj + j, BS * bk + k)]);
                  //if (BS * bj + j + 1 == 128 && P1.data[get_p(BS * bi + i, BS * bj + j+1, BS * bk + k)] > 1e-12)
                  //  printf("(%d %d %d) %f \n",BS * bi + i,BS * bj + j+1,BS * bk + k, P1.data[get_p(BS * bi + i, BS * bj + j+1, BS * bk + k)]);
                  //printf("bi = %d\n",bi);
                  // if (a != b)
                  //printf("%d %d %d %d (%d %d %d) %d %d\n", bi,Nb,i,i0,  BS * bi + i,BS * bj + j,BS * bk + k, a, b);
                  //for (int g = 0;g<7;g++)
                  //{
                  //  if (base_index + index_offset[index + g] >= Np*Np*Np)
                  //    printf("%d %d %d %d (%d %d %d) %d %d -- %lg\n", bi,Nb,i,i0,  BS * bi + i,BS * bj + j,BS * bk + k, a, b, P1.data[base_index + index_offset[index+g]]);
                  //}
                  double p1 = P1.data[base_index + index_offset[index]];
                  double p2 = P2.data[base_index + index_offset[index]];
                  double p3 = P1.data[base_index + index_offset[index + 1]];
                  double p4 = P1.data[base_index + index_offset[index + 2]];
                  double p5 = P1.data[base_index + index_offset[index + 3]];
                  double p6 = P1.data[base_index + index_offset[index + 4]];
                  double p7 = P1.data[base_index + index_offset[index + 5]];
                  double p8 = P1.data[base_index + index_offset[index + 6]];
                  ring[cur_grid].data[base_index + index_offset[index]] = 2 * p1 - p2 + c * (-6 * p1 + p3 + p4 + p5 + p6 + p7 + p8);
                }
              }
            }
          }
          else
          {
            for (int index = 0; index < 7 * BS * BS * BS; index += 7)
            {
              double p1 = P1.data[base_index + index_offset[index]];
              double p2 = P2.data[base_index + index_offset[index]];
              double p3 = P1.data[base_index + index_offset[index + 1]];
              double p4 = P1.data[base_index + index_offset[index + 2]];
              double p5 = P1.data[base_index + index_offset[index + 3]];
              double p6 = P1.data[base_index + index_offset[index + 4]];
              double p7 = P1.data[base_index + index_offset[index + 5]];
              double p8 = P1.data[base_index + index_offset[index + 6]];
              ring[cur_grid].data[base_index + index_offset[index]] = 2 * p1 - p2 + c * (-6 * p1 + p3 + p4 + p5 + p6 + p7 + p8);
            }
          }
        }
      }
    }
    t2 = std::chrono::steady_clock::now();
    solve_ms += 1e-3*std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    if (compare)
    {
      t1 = std::chrono::steady_clock::now();
      fill_reference_grid(reference_grid, step*tau);
      double diff = max_diff(ring[cur_grid], reference_grid);
      printf("%u: max diff %lg\n", step, diff);
      t2 = std::chrono::steady_clock::now();
      compare_ms += 1e-3*std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    }

    cur_grid = (cur_grid + 1)%3;
  }

  printf("took %.3f ms (%.3f + %.3f + %.3f)\n", init_ms + compare_ms + solve_ms, init_ms, solve_ms, compare_ms);

  return 0;
}