#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <omp.h>
#include <array>
#include <mpi.h>

#define THREADS_NB omp_get_max_threads()

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

  double *data;
  int x0,y0,z0;
  int xn,yn,zn;
};

double max_diff(Grid &g1, Grid &g2)
{
  std::vector<int> max_pos;
  double max_diff = 0;
  for (int i=1;i<g1.xn-1;i++)
    for (int j=1;j<g1.yn-1;j++)
      for (int k=1;k<g1.zn-1;k++)
      {
        double d = std::abs(g1.at(i,j,k) - g2.at(i,j,k));
        if (d > max_diff)
        {
          max_diff = d;
          max_pos = std::vector<int>{i,j,k};
        }
      }
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
  #pragma omp parallel for num_threads(omp_get_max_threads())
  for (int i=1;i<g.xn-1;i++)
    for (int j=1;j<g.yn-1;j++)
      for (int k=1;k<g.zn-1;k++)
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

#define X_UP_BUF 0
#define X_DOWN_BUF 1
#define Y_UP_BUF 2
#define Y_DOWN_BUF 3
#define Z_UP_BUF 4
#define Z_DOWN_BUF 5
#define RO 6

void send_slice(int recieve_block_n, const double *buf, int size)
{
  MPI_Send(buf, size, MPI_DOUBLE, recieve_block_n, 0, MPI_COMM_WORLD);
}

void recieve_from_buffer(Grid &R, const double *buf, int i)
{
  if (i == X_DOWN_BUF)
  {
    int t = 0;
    for (int y = 0; y < R.yn; y++)
      for (int z = 0; z < R.zn; z++)
        R.data[(R.xn - 1) * R.yn * R.zn + y * R.zn + z] = buf[t++];
  }
  else if (i == X_UP_BUF)
  {
    int t = 0;
    for (int y = 0; y < R.yn; y++)
      for (int z = 0; z < R.zn; z++)
        R.data[(0) * R.yn * R.zn + y * R.zn + z] = buf[t++];
  }

  else if (i == Y_DOWN_BUF)
  {
    int t = 0;
    for (int x = 0; x < R.xn; x++)
      for (int z = 0; z < R.zn; z++)
        R.data[x * R.yn * R.zn + (R.yn - 1) * R.zn + z] = buf[t++];
  }
  else if (i == Y_UP_BUF)
  {
    int t = 0;
    for (int x = 0; x < R.xn; x++)
      for (int z = 0; z < R.zn; z++)
        R.data[x * R.yn * R.zn + (0) * R.zn + z] = buf[t++];
  }

  else if (i == Z_DOWN_BUF)
  {
    int t = 0;
    for (int x = 0; x < R.xn; x++)
      for (int y = 0; y < R.yn; y++)
        R.data[x * R.yn * R.zn + y * R.zn + (R.zn - 1)] = buf[t++];
  }
  else if (i == Z_UP_BUF)
  {
    int t = 0;
    for (int x = 0; x < R.xn; x++)
      for (int y = 0; y < R.yn; y++)
        R.data[x * R.yn * R.zn + y * R.zn + (0)] = buf[t++];
  }
}

void transfer(int i, int j, int k, int blocks_x, int blocks_y, int blocks_z, int cur_grid, 
              std::array<Grid, 3> &ring_buffer, std::array<double *, 2*RO> &tmp_buffers)
{
  #define GET_BN(a,b,c) ((a) * blocks_y * blocks_z + (b) * blocks_z + (c))
  Grid &S = ring_buffer[cur_grid];
  MPI_Request send_requests[RO];
  MPI_Request recieve_requests[RO];
  bool send_finished[RO];
  bool recieve_finished[RO];
  for (int rn=0;rn<RO;rn++)
  {
    send_finished[rn] = true;
    recieve_finished[rn] = true;
  }
  
  //request data from all neigbours
  if (i != 0) 
    MPI_Irecv(tmp_buffers[RO + X_UP_BUF], S.yn*S.zn, MPI_DOUBLE, GET_BN(i-1,j,k), 0, MPI_COMM_WORLD, &recieve_requests[X_UP_BUF]);
  if (i != blocks_x-1) 
    MPI_Irecv(tmp_buffers[RO + X_DOWN_BUF], S.yn*S.zn, MPI_DOUBLE, GET_BN(i+1,j,k), 0, MPI_COMM_WORLD, &recieve_requests[X_DOWN_BUF]);
  if (j != 0) 
    MPI_Irecv(tmp_buffers[RO + Y_UP_BUF], S.xn*S.zn, MPI_DOUBLE, GET_BN(i,j-1,k), 0, MPI_COMM_WORLD, &recieve_requests[Y_UP_BUF]);
  if (j != blocks_y-1) 
    MPI_Irecv(tmp_buffers[RO + Y_DOWN_BUF], S.xn*S.zn, MPI_DOUBLE, GET_BN(i,j+1,k), 0, MPI_COMM_WORLD, &recieve_requests[Y_DOWN_BUF]);
  if (k != 0)
    MPI_Irecv(tmp_buffers[RO + Z_UP_BUF], S.xn*S.yn, MPI_DOUBLE, GET_BN(i,j,k-1), 0, MPI_COMM_WORLD, &recieve_requests[Z_UP_BUF]);
  if (k != blocks_z-1)
    MPI_Irecv(tmp_buffers[RO + Z_DOWN_BUF], S.xn*S.yn, MPI_DOUBLE, GET_BN(i,j,k+1), 0, MPI_COMM_WORLD, &recieve_requests[Z_DOWN_BUF]);

  //put all transfer data to buffers and send them to neighbours
  if (i != 0)
  {
    int t = 0;
    for (int y = 0; y < S.yn; y++)
      for (int z = 0; z < S.zn; z++)
        tmp_buffers[X_UP_BUF][t++] = S.data[(1) * S.yn * S.zn + y * S.zn + z];
    MPI_Isend(tmp_buffers[X_UP_BUF], S.yn*S.zn, MPI_DOUBLE, GET_BN(i-1,j,k), 0, MPI_COMM_WORLD, &send_requests[X_UP_BUF]);
    send_finished[X_UP_BUF] = false;
    recieve_finished[X_UP_BUF] = false;
  }
  if (i != blocks_x - 1)
  {
    int t = 0;
    for (int y = 0; y < S.yn; y++)
      for (int z = 0; z < S.zn; z++)
        tmp_buffers[X_DOWN_BUF][t++] = S.data[(S.xn - 2) * S.yn * S.zn + y * S.zn + z];
    MPI_Isend(tmp_buffers[X_DOWN_BUF], S.yn*S.zn, MPI_DOUBLE, GET_BN(i+1,j,k), 0, MPI_COMM_WORLD, &send_requests[X_DOWN_BUF]);
    send_finished[X_DOWN_BUF] = false;
    recieve_finished[X_DOWN_BUF] = false;
  }

  if (j != 0)
  {
    int t = 0;
    for (int x = 0; x < S.xn; x++)
      for (int z = 0; z < S.zn; z++)
        tmp_buffers[Y_UP_BUF][t++] = S.data[x * S.yn * S.zn + (1) * S.zn + z];
    MPI_Isend(tmp_buffers[Y_UP_BUF], S.xn*S.zn, MPI_DOUBLE, GET_BN(i,j-1,k), 0, MPI_COMM_WORLD, &send_requests[Y_UP_BUF]);
    send_finished[Y_UP_BUF] = false;
    recieve_finished[Y_UP_BUF] = false;
  }
  if (j != blocks_y - 1)
  {
    int t = 0;
    for (int x = 0; x < S.xn; x++)
      for (int z = 0; z < S.zn; z++)
        tmp_buffers[Y_DOWN_BUF][t++] = S.data[x * S.yn * S.zn + (S.yn - 2) * S.zn + z];
    MPI_Isend(tmp_buffers[Y_DOWN_BUF], S.xn*S.zn, MPI_DOUBLE, GET_BN(i,j+1,k), 0, MPI_COMM_WORLD, &send_requests[Y_DOWN_BUF]);  
    send_finished[Y_DOWN_BUF] = false;
    recieve_finished[Y_DOWN_BUF] = false;
  }

  if (k != 0)
  {
    int t = 0;
    for (int x = 0; x < S.xn; x++)
      for (int y = 0; y < S.yn; y++)
        tmp_buffers[Z_UP_BUF][t++] = S.data[x * S.yn * S.zn + y * S.zn + (1)];
    MPI_Isend(tmp_buffers[Z_UP_BUF], S.xn*S.yn, MPI_DOUBLE, GET_BN(i,j,k-1), 0, MPI_COMM_WORLD, &send_requests[Z_UP_BUF]);  
    send_finished[Z_UP_BUF] = false;
    recieve_finished[Z_UP_BUF] = false;
  }
  if (k != blocks_z - 1)
  {
    int t = 0;
    for (int x = 0; x < S.xn; x++)
      for (int y = 0; y < S.yn; y++)
        tmp_buffers[Z_DOWN_BUF][t++] = S.data[x * S.yn * S.zn + y * S.zn + (S.zn - 2)];
    MPI_Isend(tmp_buffers[Z_DOWN_BUF], S.xn*S.yn, MPI_DOUBLE, GET_BN(i,j,k+1), 0, MPI_COMM_WORLD, &send_requests[Z_DOWN_BUF]);  
    send_finished[Z_DOWN_BUF] = false;
    recieve_finished[Z_DOWN_BUF] = false;
  }

  //wait to actually recieve all required data
  bool all_finished = false;
  int flag;
  MPI_Status status;
  while (!all_finished)
  {
    all_finished = true;
    for (int rn=0;rn<RO;rn++)
    {
      if (!send_finished[rn])
      {
        MPI_Test(&send_requests[rn], &flag, &status);
        send_finished[rn] = flag;
        all_finished = all_finished && flag;
      }
    }

    for (int rn=0;rn<RO;rn++)
    {
      if (!recieve_finished[rn])
      {
        MPI_Test(&recieve_requests[rn], &flag, &status);
        recieve_finished[rn] = flag;
        all_finished = all_finished && flag;
        if (flag)
          recieve_from_buffer(S, tmp_buffers[RO+rn], rn);
      }
    }
  }
}

int main(int argc, char **argv)
{
  int block_n, blocks;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &blocks);
	MPI_Comm_rank(MPI_COMM_WORLD, &block_n);
  double start = MPI_Wtime();

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
  tau = 1.0/(128*15);

  const int steps = 21;

  std::array<Grid, 3> ring_buffer;
  int blocks_x = 2, blocks_y=1, blocks_z=1;
  get_best_blocks_layout(blocks, &blocks_x, &blocks_y, &blocks_z);
  int bk = block_n % blocks_z;
  int bj = (block_n / blocks_z) % blocks_y;
  int bi = block_n / (blocks_y*blocks_z);

  //initialize ring buffer
  int max_block_len = 0;
  {
    for (int bn = 0; bn < 3; bn++)
    {
      int bst_x = std::min(std::max(0, bi * (Np / blocks_x) - 1), Np);
      int bst_y = std::min(std::max(0, bj * (Np / blocks_y) - 1), Np);
      int bst_z = std::min(std::max(0, bk * (Np / blocks_z) - 1), Np);
      int ben_x = std::min(std::max(0, (bi + 1) * (Np / blocks_x) + 1), Np);
      int ben_y = std::min(std::max(0, (bj + 1) * (Np / blocks_y) + 1), Np);
      int ben_z = std::min(std::max(0, (bk + 1) * (Np / blocks_z) + 1), Np);
      max_block_len = std::max(std::max(max_block_len, ben_x - bst_x), std::max(ben_y - bst_y, ben_z - bst_z));
      ring_buffer[bn].init(bst_x, bst_y, bst_z, ben_x - bst_x, ben_y - bst_y, ben_z - bst_z);
    }
  }

  std::array<double *, 2*RO> tmp_bufs;
  for (int i=0;i<2*RO;i++)
    tmp_bufs[i] = new double[max_block_len*max_block_len];

  clock_t t1 = clock();

  //fill 0 and 1 levels of the buffer
  {
    fill_reference_grid(ring_buffer[0], 0 * tau);
    Grid &P1 = ring_buffer[0];
    Grid &P_res = ring_buffer[1];
    double c = (tau * tau * a_squared() / (h * h));

#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int i = 1; i < P_res.xn - 1; i++)
      for (int j = 1; j < P_res.yn - 1; j++)
        for (int k = 1; k < P_res.zn - 1; k++)
        {
          double p1 = P1.at(i, j, k);
          double p3 = P1.at(i - 1, j, k);
          double p4 = P1.at(i + 1, j, k);
          double p5 = P1.at(i, j - 1, k);
          double p6 = P1.at(i, j + 1, k);
          double p7 = P1.at(i, j, k - 1);
          double p8 = P1.at(i, j, k + 1);
          P_res.at(i, j, k) = p1 + 0.5 * tau * tau * (-6 * p1 + p3 + p4 + p5 + p6 + p7 + p8);
        }
    ring_buffer[2].clear_and_fill_borders(0);
  }

  //send and recieve data on block sides
  transfer(bi, bj, bk, blocks_x, blocks_y, blocks_z, 1, ring_buffer, tmp_bufs);

  clock_t t2 = clock();
  init_ms = 1000*(double)(t2 - t1) / (CLOCKS_PER_SEC * THREADS_NB);
  double block_max_diff = 0;
  int cur_grid = 2;
  for (int step = 2; step<steps; step++)
  {
    t1 = clock();

    //calculate values on new layer
    {
      Grid &P1 = ring_buffer[(3 + cur_grid - 1) % 3];
      Grid &P2 = ring_buffer[(3 + cur_grid - 2) % 3];
      Grid &P_res = ring_buffer[cur_grid];
      double c = (tau * tau * a_squared() / (h * h));
    #pragma omp parallel for num_threads(omp_get_max_threads())
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
          }
    }
    t2 = clock();
    solve_ms += 1000*(double)(t2 - t1) / (CLOCKS_PER_SEC * THREADS_NB);
    
    //transfer
    t1 = clock();
    transfer(bi, bj, bk, blocks_x, blocks_y, blocks_z, cur_grid, ring_buffer, tmp_bufs);
    t2 = clock();
    solve_ms += 1000*(double)(t2 - t1) / (CLOCKS_PER_SEC * THREADS_NB);

    if (compare)
    {
      t1 = clock();
      fill_reference_grid(ring_buffer[(cur_grid + 1)%3], step*tau);
      block_max_diff = std::max(block_max_diff, max_diff(ring_buffer[cur_grid], ring_buffer[(cur_grid + 1)%3]));
      t2 = clock();
      compare_ms += 1000*(double)(t2 - t1) / (CLOCKS_PER_SEC * THREADS_NB);
    }

    cur_grid = (cur_grid + 1)%3;
  }

  double block_time = 1000*(MPI_Wtime() - start);
  double total_time = 0;
  double total_max_diff = 0;
  MPI_Reduce(&block_max_diff, &total_max_diff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&block_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (block_n == 0)
  {
    printf("max diff %lg\n", total_max_diff);
    printf("took %.3f ms\n", total_time);
  }

  for (int bn = 0; bn < 3; bn++)
    delete[] ring_buffer[bn].data;
  for (int i=0;i<2*RO;i++)
    delete[] tmp_bufs[i]; 

  MPI_Finalize();
  return 0;
}