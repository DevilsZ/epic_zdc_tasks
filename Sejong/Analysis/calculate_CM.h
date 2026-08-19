#include <vector>
#include <Math/Vector3D.h>

void calculate_CM(const std::vector<float>& energies, const std::vector<float>& x_positions, const std::vector<float>& y_positions, float& CM_x, float& CM_y, float& CM_energy)
{

    float weighted_sum_x = 0.0;
    float weighted_sum_y = 0.0;
    float total_energy = 0.0;

    // total values
    size_t n = energies.size();
    for (size_t i = 0; i < n; i++) {
        total_energy += energies[i];
        weighted_sum_x += energies[i] * x_positions[i];
        weighted_sum_y += energies[i] * y_positions[i];
    }

    // calculate CM
    CM_x = weighted_sum_x / total_energy;
    CM_y = weighted_sum_y / total_energy;
    
    CM_energy = total_energy;
}