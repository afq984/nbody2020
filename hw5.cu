#include <atomic>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace param {
const int n_steps = 200000;
const double dt = 60;
const double eps = 1e-3;
const double G = 6.674e-11;
__device__ double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
const double planet_radius = 1e7;
const double missile_speed = 1e6;
double get_missile_cost(double t) { return 1e5 + 1e3 * t; }

const double eps2 = eps * eps;
const double planet_radius2 = planet_radius * planet_radius;
}  // namespace param

#define BlockSize 32

struct Vec3 {
    double x, y, z;
};

struct Prop {
    double m;
    bool is_device;
};

void read_input(const char* filename, int& n, int& planet, int& asteroid,
    std::vector<Vec3>& q, std::vector<Vec3>& v, std::vector<Prop>& prop) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    q.resize(n);
    v.resize(n);
    prop.resize(n);
    for (int i = 0; i < n; i++) {
        std::string type;
        fin >> q[i].x >> q[i].y >> q[i].z >> v[i].x >> v[i].y >> v[i].z >> prop[i].m >>
            type;
        prop[i].is_device = type == "device";
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

template <bool with_devices>
__device__ void run_step(int step, int n, Vec3* q, Vec3* nq, Vec3* v, Prop* p) {
    int i = blockIdx.x;
    double ax = 0;
    double ay = 0;
    double az = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (i == j) continue;
        double mj = p[j].m;
        if (p[j].is_device) {
            mj = with_devices ? param::gravity_device_mass(mj, step * param::dt) : 0;
        }
        double dx = q[j].x - q[i].x;
        double dy = q[j].y - q[i].y;
        double dz = q[j].z - q[i].z;
        double dist3 = rsqrt(dx * dx + dy * dy + dz * dz + param::eps2);
        dist3 = dist3 * dist3 * dist3;
        ax += mj * dx * dist3;
        ay += mj * dy * dist3;
        az += mj * dz * dist3;
    }
    atomicAdd(&v[i].x, param::G * param::dt * ax);
    atomicAdd(&v[i].y, param::G * param::dt * ay);
    atomicAdd(&v[i].z, param::G * param::dt * az);
    __syncthreads();
    if (threadIdx.x == 0) {
        nq[i].x = q[i].x + v[i].x * param::dt;
        nq[i].y = q[i].y + v[i].y * param::dt;
        nq[i].z = q[i].z + v[i].z * param::dt;
    }
}

__global__ void prob1(double* answer, int planet, int asteroid, int step, int n, Vec3* q,
    Vec3* nq, Vec3* v, Prop* p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double dx = q[planet].x - q[asteroid].x;
        double dy = q[planet].y - q[asteroid].y;
        double dz = q[planet].z - q[asteroid].z;
        double distance = sqrt(dx * dx + dy * dy + dz * dz);
        *answer = fmin(*answer, distance);
    }
    run_step<false>(step + 1, n, q, nq, v, p);
}

__global__ void prob2(int* answer, int planet, int asteroid, int step, int n, Vec3* q,
    Vec3* nq, Vec3* v, Prop* p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double dx = q[planet].x - q[asteroid].x;
        double dy = q[planet].y - q[asteroid].y;
        double dz = q[planet].z - q[asteroid].z;
        double distance2 = dx * dx + dy * dy + dz * dz;
        if (distance2 < param::planet_radius2) {
            *answer = min(*answer, step);
        }
    }
    run_step<true>(step + 1, n, q, nq, v, p);
}

__global__ void prob3(int* reach_at_step, int device, int planet, int asteroid, int step,
    int n, Vec3* q, Vec3* nq, Vec3* v, Prop* p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double dx = q[planet].x - q[device].x;
        double dy = q[planet].y - q[device].y;
        double dz = q[planet].z - q[device].z;
        double distance2 = dx * dx + dy * dy + dz * dz;
        double traveled = step * param::dt * param::missile_speed;
        if (distance2 < traveled * traveled) {
            *reach_at_step = min(*reach_at_step, step);
            p[device].m = 0;
        }
        dx = q[planet].x - q[asteroid].x;
        dy = q[planet].y - q[asteroid].y;
        dz = q[planet].z - q[asteroid].z;
        if (dx * dx + dy * dy + dz * dz < param::planet_radius2) {
            *reach_at_step = -2;
        }
    }
    run_step<true>(step + 1, n, q, nq, v, p);
}

template <class T>
T* xCudaMallocVector(const std::vector<T>& host) {
    T* devPtr;
    cudaMalloc(&devPtr, host.size() * sizeof(T));
    return devPtr;
}

template <class T>
void xCudaCopyVector(T* devPtr, const std::vector<T>& host) {
    cudaMemcpy(devPtr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template <class T>
void xCudaReturnVector(T* devPtr, std::vector<T>& host) {
    cudaMemcpy(host.data(), devPtr, host.size() * sizeof(T), cudaMemcpyDeviceToHost);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    int n, planet, asteroid;
    std::vector<Vec3> q, v;
    std::vector<Prop> p;
    read_input(argv[1], n, planet, asteroid, q, v, p);

    std::vector<std::function<std::function<void()>(Vec3*, Vec3*, Vec3*, Prop*)>> tasks;

    // Problem 1
    double min_dist = std::numeric_limits<double>::infinity();
    tasks.push_back([&](Vec3* dq, Vec3* dnq, Vec3* dv,
                        Prop* dp) -> std::function<void()> {
        xCudaCopyVector(dq, q);
        xCudaCopyVector(dv, v);
        xCudaCopyVector(dp, p);
        double* dev_min_dist;
        cudaMalloc(&dev_min_dist, sizeof min_dist);
        cudaMemcpy(dev_min_dist, &min_dist, sizeof min_dist, cudaMemcpyHostToDevice);
        for (int step = 0; step <= param::n_steps; step++) {
            prob1<<<n, BlockSize>>>(dev_min_dist, planet, asteroid, step, n, dq, dnq, dv, dp);
            std::swap(dq, dnq);
        }
        cudaMemcpy(&min_dist, dev_min_dist, sizeof min_dist, cudaMemcpyDeviceToHost);
        printf("prob1, cgle=%d\n", cudaGetLastError());
        return []() {};
    });

    // Problem 2
    int hit_time_step = param::n_steps + 1;
    tasks.push_back(
        [&](Vec3* dq, Vec3* dnq, Vec3* dv, Prop* dp) -> std::function<void()> {
            int* dev_hit_time_step;
            xCudaCopyVector(dq, q);
            xCudaCopyVector(dv, v);
            xCudaCopyVector(dp, p);
            cudaMalloc(&dev_hit_time_step, sizeof hit_time_step);
            cudaMemcpy(dev_hit_time_step, &hit_time_step, sizeof hit_time_step,
                cudaMemcpyHostToDevice);
            for (int step = 0; step <= param::n_steps; step++) {
                prob2<<<n, BlockSize>>>(
                    dev_hit_time_step, planet, asteroid, step, n, dq, dnq, dv, dp);
                std::swap(dq, dnq);
            }
            cudaMemcpy(&hit_time_step, dev_hit_time_step, sizeof hit_time_step,
                cudaMemcpyDeviceToHost);
            if (hit_time_step > param::n_steps) {
                hit_time_step = -2;
            }
            printf("prob2, cgle=%d\n", cudaGetLastError());
            return []() {};
        });

    // Problem 3
    // TODO
    int gravity_device_id = -1;
    double missile_cost = param::n_steps + 1;
    for (int i = 0; i < n; i++) {
        if (p[i].is_device) {
            tasks.push_back(
                [&, i](Vec3* dq, Vec3* dnq, Vec3* dv, Prop* dp) -> std::function<void()> {
                    int host_reached_at = param::n_steps + 1;
                    int* dev_reached_at;
                    cudaMalloc(&dev_reached_at, sizeof host_reached_at);
                    cudaMemcpy(dev_reached_at, &host_reached_at, sizeof host_reached_at,
                        cudaMemcpyHostToDevice);
                    xCudaCopyVector(dq, q);
                    xCudaCopyVector(dv, v);
                    xCudaCopyVector(dp, p);
                    for (int step = 0; step <= param::n_steps; step++) {
                        prob3<<<n, BlockSize>>>(dev_reached_at, i, planet, asteroid, step, n, dq,
                            dnq, dv, dp);
                        std::swap(dq, dnq);
                    }
                    cudaMemcpy(&host_reached_at, dev_reached_at, sizeof host_reached_at,
                        cudaMemcpyDeviceToHost);
                    printf("prob3, cgle=%d, %d, %d\n", cudaGetLastError(), i,
                        host_reached_at);
                    if (host_reached_at != -2 and host_reached_at != param::n_steps + 1) {
                        return [&]() {
                            if (host_reached_at < missile_cost) {
                                missile_cost = host_reached_at;
                                gravity_device_id = i;
                            }
                        };
                    } else {
                        return []() {};
                    }
                });
        }
    }

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::atomic_int task(0);
    std::vector<std::thread> threads;
    std::mutex mux;
    for (int dev = 0; dev < deviceCount; dev++) {
        threads.emplace_back([&, dev]() {
            cudaSetDevice(dev);
            Vec3* dq = xCudaMallocVector(q);
            Vec3* dnq = xCudaMallocVector(q);
            Vec3* dv = xCudaMallocVector(v);
            Prop* dp = xCudaMallocVector(p);
            for (int i = task.fetch_add(1); i < tasks.size(); i = task.fetch_add(1)) {
                auto done = tasks[i](dq, dnq, dv, dp);
                {
                    std::lock_guard<std::mutex> l(mux);
                    done();
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    if (gravity_device_id == -1) {
        missile_cost = 0;
    } else {
        missile_cost = param::get_missile_cost(missile_cost * param::dt);
    }

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
}
