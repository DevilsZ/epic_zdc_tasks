import ROOT
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from collections import Counter

# ***************************
# Open the ROOT files
chain = ROOT.TChain("events")
chain.Add("/work/code/nb53_10k_80_260GeV.edm4hep.root")
print(f"Added {chain.GetEntries()}")

# ***************************
# Function to detremine point of intersection with ZDC face
z_ZDC = 35740  # first layer of WSi in mm
def intersection(vtx_vec, end_vec, z_ZDC=z_ZDC): 
  # *****************************************
  # Find point of intersection (x,y) using 3D paramteric equatons of a line 
  # r_vec = v0_vec + t * v_vec 
  #   x = x0 + t * vx
  #   y = y0 + t * vy
  #   z = z0 + t * vz -> t = (z-z0)/ (z1-z0)
  # *****************************************
  v = end_vec - vtx_vec
  if v[2] == 0:
    return None, None  # avoid division by zero

  t = (z_ZDC - vtx_vec[2]) / v[2]
  x = vtx_vec[0] + t * v[0]
  y = vtx_vec[1] + t * v[1]

  """   print("vtx:", vtx_vec)
  print("end:", end_vec)
  print("v:", v)
  print("t:", t)
  print("x_hit:", x) """

  return x, y

# ***************************
# Function to totate the hit data to the particle's reference frame 
def rotate_hit(hit, angle=0.025):
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    x_rot = hit[0] * cos_a + hit[2] * sin_a
    z_rot = -hit[0] * sin_a + hit[2] * cos_a

    if len(hit) == 4:
        return (x_rot, hit[1], z_rot, hit[3])
    else:
        return (x_rot, hit[1], z_rot)

# Define the front face of ZDC in particle reference frame 
ZDC_corners = [
  [-1240, 300, 35860],  # top right
  [-1240, -300, 35860], # bottom right
  [-594, -300, 35860],  # bottom left
  [-594, 300, 35860]    # top left
]

ZDC_corners_rot = [rotate_hit(corner) for corner in ZDC_corners]

# ***************************
# 2D histogram: XY Distribution
nbins_WSi = 40
nbins_HCal = 40
min = -400
max = 400
h_XY_WSi = ROOT.TH2F("h_XY_wsi", "XY Hit Distribution (WSi); X [mm]; Y [mm]", nbins_WSi, min, max, nbins_WSi, min, max)
h_XY_HCal = ROOT.TH2F("h_XY_hcal", "XY Hit Distribution (SiPM-on-tile); X [mm]; Y [mm]", nbins_HCal, min, max, nbins_HCal, min, max)

# Loop over events
for event in chain:
  MCParticles_vec = event.MCParticles
  hcal_hits = event.HcalFarForwardZDCHits
  wsi_hits  = event.ZDC_WSi_Hits

  pid_list = []
  for particle in MCParticles_vec:
    pid_list.append(particle.PDG)

  if len(pid_list) != 1: # Ignore events where the neutron decays before reaching the ZDC 
    continue

  neu = MCParticles_vec[0]
  vtx = np.array([neu.vertex.x, neu.vertex.y, neu.vertex.z])
  end = np.array([neu.endpoint.x, neu.endpoint.y, neu.endpoint.z])
  x_hit, y_hit = intersection(vtx_vec=vtx, end_vec=end)

  for hit in hcal_hits: 
    hit = [hit.position.x, hit.position.y, hit.position.z]
    hit = rotate_hit(hit=hit) 
    h_XY_HCal.Fill(hit[0], hit[1])

  for hit in wsi_hits: 
    hit = [hit.position.x, hit.position.y, hit.position.z]
    hit = rotate_hit(hit=hit) 
    h_XY_WSi.Fill(hit[0], hit[1])

# ***************************
# Draw histogram
def draw_histogram(hist, outname):
    c = ROOT.TCanvas(hist.GetName(), hist.GetTitle(), 800, 700)
    # Adjust margins
    c.SetLeftMargin(0.15)
    c.SetRightMargin(0.15)
    c.SetTopMargin(0.15)

    hist.Draw("COLZ")

    # Adjust stats box transparency
    c.Update()
    st = hist.GetListOfFunctions().FindObject("stats")
    if st:
        st.SetX1NDC(0.70)
        st.SetX2NDC(0.90)
        st.SetFillColorAlpha(st.GetFillColor(), 0.30)
        st.Draw()  # redraw on top

    c.Modified()
    c.Update()
    c.SaveAs(outname)

# Draw and save HCal and WSi histograms
draw_histogram(h_XY_HCal, "XY_HCal_coverage.png")
draw_histogram(h_XY_WSi,  "XY_WSi_coverage.png")



