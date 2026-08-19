# Lambda Reconstruction Method

**ZDC_Configuration**
  - Current ZDC configuration (consists of 20 W-Si layers)
  - The 4th and 10th layers are pixel layers, while the remaining 18 layers are pad layers.
  - Pixel layers: 0.5 mm × 0.5 mm pixel size
  - Pad layers: 1 cm × 1 cm cell size


**Analysis**

Lambda reconstruct process

             ┌── Si_determineLayer.h
             │
             ├── Hcal_determineLayer.h
             │
             ├── calculate_CM.h
             │
             ├── ImagingTopoCluster.cc
             │
             └── PCAtrack.cc
                     │
                     ↓
             Lambda_reconstruction.C

Explain in more detail

1. Store information of the true particles
2. Rotate hit positions along the Z-axis and store the rotated hit information
3. Perform clustering
4. Identify the neutron cluster in the HCal and the two photon clusters in the W-Si
5. Calculate the center of mass for each layer and store the reverse-rotated CM positions
6. Calculate the Z-vertex point using the rotated cluster centers
7. Fit tracks using PCA with the reverse-rotated Z-vertex point
8. Reconstruct the Lambda

* Rotate → Rotate the Y-axis by +0.025 rad 
* Reverse-rotate → Rotate the Y-axis by -0.025 rad





