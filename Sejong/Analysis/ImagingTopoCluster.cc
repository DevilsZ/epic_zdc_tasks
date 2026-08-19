bool is_neighbor(
    const Hit& h1,
    const Hit& h2,
    double same_dx,
    double same_dy,
    double diff_dx,
    double diff_dy,
    int    max_layer_diff
) {
    // Separate detectors in the clustering process
    if (h1.detector != h2.detector) return false;
    
    int ldiff = std::abs(h1.layer - h2.layer);

    if (ldiff == 0) {
        return (std::abs(h1.x - h2.x) <= same_dx &&
                std::abs(h1.y - h2.y) <= same_dy);
    }

    if (ldiff <= max_layer_diff) {
        return (std::abs(h1.x - h2.x) <= diff_dx &&
                std::abs(h1.y - h2.y) <= diff_dy);
    }

    return false;
}

std::vector<Cluster> topo_cluster(
    std::vector<Hit>& hits,
    int    max_layer_diff,
    int    min_cluster_hits,
    double min_cluster_energy
) {
    double seed_thr, neighbor_thr, min_hit_thr;
    double same_dx, same_dy, diff_dx, diff_dy;


    int N = hits.size();
    std::vector<bool> used(N, false);
    std::vector<Cluster> clusters;

    std::vector<int> order(N);
    for (int i = 0; i < N; ++i) order[i] = i;

    std::sort(order.begin(), order.end(),
              [&](int a, int b) {
                  return hits[a].energy > hits[b].energy;
              });

    for (int seed_idx : order) {
        if(hits[seed_idx].detector == 0){
            seed_thr = 0.1; //seed hit E
            neighbor_thr = 0.09; //neighbor hit E
            min_hit_thr = 0.007; //minimum hit E 
            same_dx = 10.0 * 1; //same layer->dx
            same_dy = 10.0 * 1; //same layer->dy
            diff_dx = 10.0 * 1; //different layer->dx
            diff_dy = 10.0 * 1; //different layer->dy
        }
        else{
            seed_thr = 0.1;
            neighbor_thr = 0.025;
            min_hit_thr = 0.025;
            same_dx = 25.0 * 10;
            same_dy = 27.2 * 10;
            diff_dx = 25.0 * 10;
            diff_dy = 27.2 * 10;
            max_layer_diff = 30;
        }

        if (hits[seed_idx].energy < seed_thr) break;
        if (used[seed_idx]) continue;

        Cluster cl;
        cl.energy = 0.0;

        std::vector<int> queue;
        queue.push_back(seed_idx);
        used[seed_idx] = true;

        for (size_t qi = 0; qi < queue.size(); ++qi) {
            int i = queue[qi];
            
            // hit 자체를 cluster에 저장
            cl.hits.push_back(hits[i]);
            
            cl.energy += hits[i].energy;

            for (int j = 0; j < N; ++j) {
                if (used[j]) continue;
                if (hits[j].energy < min_hit_thr) continue;

                if (is_neighbor(hits[i], hits[j],
                                same_dx, same_dy,
                                diff_dx, diff_dy,
                                max_layer_diff)) {

                    if (hits[i].energy >= neighbor_thr) {
                        used[j] = true;
                        queue.push_back(j);
                    }
                }
            }
        }

        // }

        if ((int)cl.hits.size() >= min_cluster_hits &&
            cl.energy >= min_cluster_energy) {
            clusters.push_back(cl);
        }
    }

    return clusters;
}
