#ifndef IMAGING_TOPO_CLUSTER_H
#define IMAGING_TOPO_CLUSTER_H

#include <vector>

struct Hit {
    int layer;
    double x, y, z;
    double energy;
    int detector; // 0=Si, 1=Hcal
};

struct Cluster {
    std::vector<Hit> hits;  // hit 정보 직접 보관
    float energy = 0.0;
};

bool is_neighbor(
    const Hit& h1,
    const Hit& h2,
    double same_dx,
    double same_dy,
    double diff_dx,
    double diff_dy,
    int    max_layer_diff
);

std::vector<Cluster> topo_cluster(
    const std::vector<Hit>& hits,
    double seed_thr,
    double neighbor_thr,
    double min_hit_thr,
    double same_dx,
    double same_dy,
    double diff_dx,
    double diff_dy,
    int    max_layer_diff,
    int    min_cluster_hits,
    double min_cluster_energy
);

#endif
