import ROOT
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from collections import Counter

# Open the ROOT files
data_dir = "Lambda_noAngle"
chain = ROOT.TChain("events")

for filename in os.listdir(data_dir):
  if filename.endswith(".edm4hep.root"):
    filepath = os.path.join(data_dir, filename)
    chain.Add(filepath)

print(f"Added {chain.GetEntries()} entries from {data_dir}")

# Identify Lambda decay events of interest 
# PID: photon 22, pi0 111, neutron 2112, lambda 3122
def interesting_event(lst):
  required_pids = Counter({22: 2, 111: 1, 2112: 1, 3122: 1})
  counts = Counter(lst)
  return counts == required_pids

# 1D histograms: endpoint z (bin in momentum)
mom_range = [[110,140], [140,170], [170, 200], [200,230], [230,270]]
hists = []
nbins = 25         # number of bins for Z
z_min = 0          # minimum endpoint Z [mm]
z_max = 37000      # maximum endpoint Z [mm]

for i, (mom_min, mom_max) in enumerate(mom_range):
  h_endZ = ROOT.TH1F(f"h_endZ_{i}", f"Lambda Endpoint Z ({mom_min}-{mom_max} GeV);Endpoint Z [mm];Counts", nbins, z_min, z_max)
  hists.append(h_endZ)

# 1D histogram: Lambda endpoint z (all momenta)
h_endZ_1D = ROOT.TH1F("h_endZ_1D", "Lambda Endpoint Z; Endpoint Z [mm]; Counts", 50, 0, 37000)

# Loop over events
for event in chain:
  pid_list = []
  MCParticles_vec = event.MCParticles
  for particle in MCParticles_vec:
    pid_list.append(particle.PDG)

  if len(pid_list) != 5:
    continue

  if interesting_event(pid_list):
    lda = MCParticles_vec[0]
    #neu = MCParticles_vec[2]

    lda_p = np.sqrt(lda.momentum.x**2 + lda.momentum.y**2 + lda.momentum.z**2)

    for i, (mom_min, mom_max) in enumerate(mom_range):
      if mom_min <= lda_p < mom_max:
        hists[i].Fill(lda.endpoint.z)
        break

    h_endZ_1D.Fill(lda.endpoint.z)

# Bin in z (use this later for the neutron/photon starting points)
z_max = 35700.0  # mm (safety margin of 40 mm since WSi starts at 35740)
z_min = 0.0
n_z_bins = 6
z_positions = np.linspace(z_min, z_max, n_z_bins)
print("z positions for particle gun:", z_positions)

# Save the histograms as a pdf 
c = ROOT.TCanvas("c", "Lambda Endpoint Z", 800, 600)
pdf_name = "lambda_endpoints.pdf"

# Start the PDF
c.Print(f"{pdf_name}[")

# Plot the five momentum-bin histograms
for i, h in enumerate(hists):
    h.Draw()
    c.Print(pdf_name)

""" # Fit the combined histogram
zMin = h_endZ_1D.GetXaxis().GetXmin()
zMax = h_endZ_1D.GetXaxis().GetXmax()
fExp = ROOT.TF1("fExp", "[0]*exp(-x/[1])", zMin, zMax)
fExp.SetParameter(0, h_endZ_1D.GetMaximum())
fExp.SetParameter(1, 15000.0)
fExp.SetRange(3000.0, 35000.0)
h_endZ_1D.Fit(fExp, "R L")
lambda_mm = fExp.GetParameter(1)
lambda_err = fExp.GetParError(1)
print(f"Decay length λ = {lambda_mm:.1f} ± {lambda_err:.1f} mm")
 """
# Plot the combined histogram
h_endZ_1D.SetLineColor(ROOT.kBlack)
h_endZ_1D.SetLineWidth(2)
h_endZ_1D.Draw()
lines = []
#fExp.Draw("same")
for z in z_positions:
  line = ROOT.TLine(z, 0, z, h_endZ_1D.GetMaximum())
  line.SetLineColor(ROOT.kBlue)
  line.SetLineWidth(2)
  line.SetLineStyle(2)  # dashed
  line.Draw()
  lines.append(line)  # <-- important!
c.Print(pdf_name)

# Close the PDF
c.Print(f"{pdf_name}]")
