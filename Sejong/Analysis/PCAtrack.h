#ifndef PCA_TRACK_H
#define PCA_TRACK_H

#include <vector>
#include <Eigen/Dense>

struct Track3D
{
    Eigen::Vector3d point;
    Eigen::Vector3d direction;
};

Track3D FitTrackPCAWeighted(
    const std::vector<Eigen::Vector3d>& hits,
    const std::vector<double>& energies
);

#endif

Track3D FitTrackPCAWeighted(const std::vector<Eigen::Vector3d>& hits,
                            const std::vector<double>& energies)
{
    Track3D result;

    int N = hits.size();
    if (N < 2 || energies.size() != hits.size()) {
        std::cerr << "Invalid input\n";
        return result;
    }

    // 1️⃣ Weighted centroid
    Eigen::Vector3d centroid(0,0,0);
    double sumW = 0.0;

    for (int i=0; i<N; i++) {
        centroid += energies[i] * hits[i];
        sumW += energies[i];
    }
    centroid /= sumW;

    // 2️⃣ Weighted covariance matrix
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();

    for (int i=0; i<N; i++) {
        Eigen::Vector3d d = hits[i] - centroid;
        double w = energies[i];
        cov += w * (d * d.transpose());
    }

    cov /= sumW;

    // 3️⃣ Eigen decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Eigen decomposition failed\n";
        return result;
    }

    Eigen::Vector3d eigenValues = solver.eigenvalues();
    Eigen::Matrix3d eigenVectors = solver.eigenvectors();

    // 4️⃣ Largest eigenvalue index
    int maxIndex = 0;
    if (eigenValues[1] > eigenValues[maxIndex]) maxIndex = 1;
    if (eigenValues[2] > eigenValues[maxIndex]) maxIndex = 2;

    Eigen::Vector3d direction = eigenVectors.col(maxIndex).normalized();

    result.point = centroid;
    if(direction.z() > 0)   result.direction = direction;
    else result.direction = -direction;

    return result;
}
