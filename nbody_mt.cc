#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace param {
const int n_steps = 200000;
const double dt = 60;
const double eps = 1e-3;
const double G = 6.674e-11;
double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
const double planet_radius = 1e7;
const double missile_speed = 1e6;
double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

class Timer {
    const char* msg;
    int id;
    std::chrono::steady_clock::time_point t0;

public:
    Timer(const char* msg, int id)
        : msg(msg), id(id), t0(std::chrono::steady_clock::now()) {}
    ~Timer() {
        printf("%s %d %f\n", msg, id,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
};

void read_input(const char* filename, int& n, int& planet, int& asteroid,
    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
    std::vector<double>& m, std::vector<bool>& is_device) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    m.resize(n);
    is_device.resize(n);
    for (int i = 0; i < n; i++) {
        std::string type;
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type;
        is_device[i] = type == "device";
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

void run_step(int step, int n, std::vector<double>& qx, std::vector<double>& qy,
    std::vector<double>& qz, std::vector<double>& vx, std::vector<double>& vy,
    std::vector<double>& vz, const std::vector<double>& m,
    const std::vector<bool>& is_device) {
    // compute accelerations
    for (int i = 0; i < n; i++) {
        double ax = 0;
        double ay = 0;
        double az = 0;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double mj = m[j];
            if (is_device[j]) {
                mj = param::gravity_device_mass(mj, step * param::dt);
            }
            double dx = qx[j] - qx[i];
            double dy = qy[j] - qy[i];
            double dz = qz[j] - qz[i];
            double dist3 =
                pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);
            ax += param::G * mj * dx / dist3;
            ay += param::G * mj * dy / dist3;
            az += param::G * mj * dz / dist3;
        }

        // update velocities
        vx[i] += ax * param::dt;
        vy[i] += ay * param::dt;
        vz[i] += az * param::dt;
    }

    // update positions
    for (int i = 0; i < n; i++) {
        qx[i] += vx[i] * param::dt;
        qy[i] += vy[i] * param::dt;
        qz[i] += vz[i] * param::dt;
    }
}

double prob1(char** argv) {
    Timer _("prob1", 0);
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<bool> is_device;
    double min_dist = std::numeric_limits<double>::infinity();
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, is_device);
    for (int i = 0; i < n; i++) {
        if (is_device[i]) {
            m[i] = 0;
        }
    }
    for (int step = 1; step <= param::n_steps; step++) {
        run_step(step, n, qx, qy, qz, vx, vy, vz, m, is_device);
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        min_dist = std::min(min_dist, sqrt(dx * dx + dy * dy + dz * dz));
    }
    return min_dist;
}

int prob2(char** argv) {
    Timer _("prob2", 0);
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<bool> is_device;
    int hit_time_step = -2;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, is_device);
    for (int step = 1; step <= param::n_steps; step++) {
        run_step(step, n, qx, qy, qz, vx, vy, vz, m, is_device);
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
            hit_time_step = step;
            printf(
                "hit at %d with %e\n", hit_time_step, sqrt(dx * dx + dy * dy + dz * dz));
            break;
        }
    }
    return hit_time_step;
}

std::pair<int, double> prob3(char** argv, int i) {
    Timer _("prob3", i);
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<bool> is_device;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, is_device);
    int reach_at_step = param::n_steps + 1;
    std::pair<double, int> worst{std::numeric_limits<double>::infinity(), -2};
    for (int step = 0; step <= param::n_steps; step++) {
        double dx = qx[planet] - qx[i];
        double dy = qy[planet] - qy[i];
        double dz = qz[planet] - qz[i];
        if (dx * dx + dy * dy + dz * dz <
            pow(step * param::missile_speed * param::dt, 2)) {
            reach_at_step = std::min(reach_at_step, step);
            m[i] = 0;
        }
        if (step > 0) {
            run_step(step, n, qx, qy, qz, vx, vy, vz, m, is_device);
        }
        dx = qx[planet] - qx[asteroid];
        dy = qy[planet] - qy[asteroid];
        dz = qz[planet] - qz[asteroid];
        if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
            printf("targeting %d hit at %d %e\n", i, step,
                sqrt(dx * dx + dy * dy + dz * dz));
            return {-1, std::numeric_limits<double>::infinity()};
        }
        worst = std::min(worst, std::make_pair(sqrt(dx * dx + dy * dy + dz * dz), step));
    }
    double cost = param::get_missile_cost(param::dt * reach_at_step);
    printf("targeting %d reach at %d worst at %d %e\n", i, reach_at_step, worst.second,
        worst.first);
    return {i, cost};
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }

    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<bool> is_device;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, is_device);
    auto min_dist = std::async(prob1, argv);
    auto hit_time_step = std::async(prob2, argv);
    int gravity_device_id = -1;
    double missile_cost = std::numeric_limits<double>::infinity();
    std::vector<decltype(std::async(prob3, argv, 1))> p3s;
    for (int i = 0; i < n; i++) {
        if (is_device[i]) {
            p3s.emplace_back(std::async(prob3, argv, i));
        }
    }
    for (auto& r : p3s) {
        int id;
        double cost;
        std::tie(id, cost) = r.get();
        if (cost < missile_cost) {
            gravity_device_id = id;
            missile_cost = cost;
        }
    }
    if (gravity_device_id == -1) {
        missile_cost = 0;
    }

    write_output(
        argv[2], min_dist.get(), hit_time_step.get(), gravity_device_id, missile_cost);
}
