#include <TMath.h>

const int Hcal_NUM_LAYERS = 64;
const double Hcal_first_layer = 35954; //mm
const double Hcal_layer_length = 28.3 ; //mm

int Hcal_determineLayer(float zPosition) {

  float Hcal_layerThickness = Hcal_layer_length; // mm
  float Hcal_firstLayerZ = Hcal_first_layer; // mm
  int Hcal_layer = ((zPosition - Hcal_firstLayerZ) / Hcal_layerThickness) + 20;
  // cout<<zPosition<<'\t'<<Hcal_firstLayerZ<<'\t'<<Hcal_layer<<endl;
  return (Hcal_layer >= 0 && Hcal_layer < NUM_LAYERS) ? Hcal_layer : -1;

}