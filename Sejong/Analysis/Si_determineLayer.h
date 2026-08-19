#include <TMath.h>

const int NUM_LAYERS = 84;
const double first_layer = 35793; //mm
const double layer_length = 6.64 ; //mm

int determineLayer(float zPosition) {

  float layerThickness = layer_length; // mm
  float firstLayerZ = first_layer; // mm
  int layer = ((zPosition - firstLayerZ) / layerThickness);
  // cout<<zPosition<<'\t'<<firstLayerZ<<'\t'<<layer<<endl;
  return (layer >= 0 && layer < NUM_LAYERS) ? layer : -1;

}