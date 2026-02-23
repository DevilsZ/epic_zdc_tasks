import ROOT
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# ******************************************************
#         Default values and Data processing 
# ******************************************************
# Define HCAL layer positions
n_layers = 64
z_start = 35860 # mm (LYSO ECal: 35800 mm)
z_end   = 37490 # mm (LYSO Ecal: 37400 mm) 
lay_thick = (z_end - z_start) / n_layers
hcalpts = [z_start + lay_thick * i for i in range(n_layers + 1)]

# ***************************
# Define the WSi layer positions 
n_layers_WSi = 20
z_start_WSi = 35740 # mm 
z_end_WSi   = 35860 # mm
lay_thick_WSi = (z_end_WSi - z_start_WSi) / n_layers_WSi
WSipts = [z_start_WSi + lay_thick_WSi * i for i in range(n_layers_WSi + 1)]

# ***************************
# Process the data to reconstruct track angle, etc. (functions used defined below)
energy_threshold_hcal = 1e-5  # GeV -> Arbitrarily chosen to remove noise 
sampling_fraction_hcal = 0.0203 # Determined in previous energy resolution study
theta_ZDC = 0.025  # radians -> for Z-axis correction (ZDC is not located directly along Z-axis)

def extract_data(chain):
  master_list = []
  for event in chain:
    # Get event and hit information
    MCParticles_vec = event.MCParticles
    hcal_hits = event.HcalFarForwardZDCHits
    wsi_hits  = event.ZDC_WSi_Hits

    # Prepare per-layer storage
    event_layers_HCal = [[] for _ in range(n_layers)]
    event_layers_WSi = [[] for _ in range(n_layers_WSi)]
    hit_list = []

    # Ensure that we only select events where the neutron made it to the ZDC and didnt decay 
    if len(MCParticles_vec) != 1:
      continue 

    particle = MCParticles_vec[0]
    if particle.PDG != 2112:
      continue

    # Collect hits
    for hit in hcal_hits:
      if hit.energy > energy_threshold_hcal:
        hit_list.append((hit.position.x, hit.position.y, hit.position.z, hit.energy, "h"))

    for hit in wsi_hits:
      if hit.energy > 0:
        hit_list.append((hit.position.x, hit.position.y, hit.position.z, hit.energy, "w"))

    if len(hit_list) == 0:
      continue 

    # Assign hits to layers
    for (x, y, z_raw, energy, det) in hit_list:
      if det == "h":
        z = -x * np.sin(theta_ZDC) + z_raw * np.cos(theta_ZDC)
        energy /= sampling_fraction_hcal

        for j in range(n_layers):
          if hcalpts[j] < z <= hcalpts[j+1]:
            event_layers_HCal[j].append((x, y, z, energy))

      if det == "w":
        z = -x * np.sin(theta_ZDC) + z_raw * np.cos(theta_ZDC)

        for j in range(n_layers_WSi):
          if WSipts[j] < z <= WSipts[j+1]:
            event_layers_WSi[j].append((x, y, z, energy))

    # Compute average hit positions for all layers in this event
    event_layers = event_layers_WSi + event_layers_HCal
    average_layers_HCal = average_hit(event_layers_HCal, weighting="log", hit_thres=10)
    error_layers_HCal = average_error(event_layers_HCal, average_layers_HCal, weighting="log")
    average_layers_WSi = average_hit(event_layers_WSi, weighting="linear", hit_thres=8)
    error_layers_WSi = average_error(event_layers_WSi, average_layers_WSi, weighting="linear")

    # Compute true MC track vector (as a tuple/vector)
    vx, vy, vz = particle.vertex.x, particle.vertex.y, particle.vertex.z
    ex, ey, ez = particle.endpoint.x, particle.endpoint.y, particle.endpoint.z
    dx, dy, dz = (ex - vx), (ey - vy), (ez - vz)

    px, py, pz = particle.momentum.x, particle.momentum.y, particle.momentum.z
    m = particle.mass
    E_tot = np.sqrt(px**2 + py**2 + pz**2 + m**2)

    R_true = np.sqrt(dx**2 + dy**2)
    theta_true = np.arctan2(R_true, dz)  # radians
    phi_true = np.arctan2(dy, dx) 

    mc_track = (dx, dy, dz, R_true, theta_true, phi_true, E_tot)  # vector format

    # Save to master list
    master_list.append({
      "event_id": event.EventHeader[0].eventNumber,
      "average_layers_HCal": average_layers_HCal,
      "error_layers_HCal": error_layers_HCal,
      "average_layers_WSi": average_layers_WSi,
      "error_layers_WSi": error_layers_WSi,
      "event_layers": event_layers,
      "mc_track": mc_track  # access elements as mc_track[0], mc_track[1], etc.
    })
  
  return master_list

# ******************************************************
#         Average Hit and Anuglar Resolition
# ******************************************************
# Average_hit and average_error are used in extract_data 
# Average hit position by layer 
def average_hit(event_layers, weighting=None, hit_thres=5, C=4.0):
  energy_weighted_averages = []

  for layer_hits in event_layers: 
    n_hits = len(layer_hits)
    if n_hits < hit_thres:
      energy_weighted_averages.append((None, None, None))
      continue

    if weighting is None:
      avg_x = sum(hit[0] for hit in layer_hits) / n_hits
      avg_y = sum(hit[1] for hit in layer_hits) / n_hits
      avg_z = sum(hit[2] for hit in layer_hits) / n_hits
      energy_weighted_averages.append((avg_x, avg_y, avg_z))

    elif weighting == "linear":
      total_energy = sum(hit[3] for hit in layer_hits)
      if total_energy > 0:
        avg_x = sum(hit[0] * hit[3] for hit in layer_hits) / total_energy
        avg_y = sum(hit[1] * hit[3] for hit in layer_hits) / total_energy
        avg_z = sum(hit[2] * hit[3] for hit in layer_hits) / total_energy
        energy_weighted_averages.append((avg_x, avg_y, avg_z))
      else:
        energy_weighted_averages.append((None, None, None))

    elif weighting == "log":
      energies = np.array([hit[3] for hit in layer_hits])
      total_energy = energies.sum()
      if total_energy > 0:
        weights = np.log(energies / total_energy) + C
        weights = np.clip(weights, 0.0, None)
        wsum = weights.sum()
        if wsum > 0:
          xs = np.array([hit[0] for hit in layer_hits])
          ys = np.array([hit[1] for hit in layer_hits])
          zs = np.array([hit[2] for hit in layer_hits])
          avg_x = np.sum(xs * weights) / wsum
          avg_y = np.sum(ys * weights) / wsum
          avg_z = np.sum(zs * weights) / wsum
          energy_weighted_averages.append((avg_x, avg_y, avg_z))
        else:
          energy_weighted_averages.append((None, None, None))
      else:
        energy_weighted_averages.append((None, None, None))

    elif weighting == "sqrt":
      energies = np.array([hit[3] for hit in layer_hits])
      weights = np.sqrt(energies)
      wsum = weights.sum()
      if wsum > 0:
        xs = np.array([hit[0] for hit in layer_hits])
        ys = np.array([hit[1] for hit in layer_hits])
        zs = np.array([hit[2] for hit in layer_hits])
        avg_x = np.sum(xs * weights) / wsum
        avg_y = np.sum(ys * weights) / wsum
        avg_z = np.sum(zs * weights) / wsum
        energy_weighted_averages.append((avg_x, avg_y, avg_z))
      else:
        energy_weighted_averages.append((None, None, None))

    else:
      raise ValueError(f"Unknown weighting scheme: {weighting}")

  return energy_weighted_averages

# ***************************
# Average hit position error by layer 
def average_error(event_layers, average_layers, weighting=None, C=4.0):
  error_layers = []

  for layer_hits, (xav, yav, zav) in zip(event_layers, average_layers):
    n_hits = len(layer_hits)
    if n_hits == 0 or xav is None or yav is None or zav is None:
      error_layers.append((None, None, None))
      continue

    # Determine weights
    if weighting is None:
      weights = np.ones(n_hits)
    elif weighting == "linear":
      energies = np.array([hit[3] for hit in layer_hits])
      total_energy = energies.sum()
      if total_energy > 0:
        weights = energies / total_energy
      else:
        error_layers.append((None, None, None))
        continue
    elif weighting == "log":
      energies = np.array([hit[3] for hit in layer_hits])
      total_energy = energies.sum()
      if total_energy > 0:
        weights = np.log(energies / total_energy) + C
        weights = np.clip(weights, 0.0, None)
        if np.all(weights == 0):
          error_layers.append((None, None, None))
          continue
      else:
        error_layers.append((None, None, None))
        continue
    elif weighting == "sqrt":
      energies = np.array([hit[3] for hit in layer_hits])
      weights = np.sqrt(energies)
      if np.all(weights == 0):
        error_layers.append((None, None, None))
        continue
    else:
      raise ValueError(f"Unknown weighting scheme: {weighting}")

    sum_wi = np.sum(weights)
    sum_wi2 = np.sum(weights**2)

    xs = np.array([hit[0] for hit in layer_hits])
    ys = np.array([hit[1] for hit in layer_hits])
    zs = np.array([hit[2] for hit in layer_hits])

    x_var = np.sum(weights * xs**2) / sum_wi - xav**2
    y_var = np.sum(weights * ys**2) / sum_wi - yav**2
    z_var = np.sum(weights * zs**2) / sum_wi - zav**2

    if sum_wi**2 != sum_wi2:
      errx = np.sqrt(max(0, x_var * sum_wi2 / (sum_wi**2 - sum_wi2)))
      erry = np.sqrt(max(0, y_var * sum_wi2 / (sum_wi**2 - sum_wi2)))
      errz = np.sqrt(max(0, z_var * sum_wi2 / (sum_wi**2 - sum_wi2)))
    else:
      errx = erry = errz = None

    error_layers.append((errx, erry, errz))

  return error_layers

# ***************************
# Calculate the angular resolution and (optionally) plot the data in the RZ Plane 
def ang_res_theta(event_data, filename=None, detector="HCal", plot=True):
  if detector == "HCal":
    average_layers = event_data["average_layers_HCal"]
    error_layers   = event_data["error_layers_HCal"]
  elif detector == "WSi":
    average_layers = event_data["average_layers_WSi"]
    error_layers   = event_data["error_layers_WSi"]
  else: 
    print("Uknown detector. Please enter HCal or WSi.")

  # Filter out layers with missing averages (None)
  valid = []
  for (ax, ay, az), (ex, ey, ez) in zip(average_layers, error_layers):
    if ax is not None and ay is not None and az is not None:
      valid.append((ax, ay, az, ex, ey, ez))

  if len(valid) == 0:
    return None, None, None

  # Convert to arrays
  x    = np.array([v[0] for v in valid], dtype=float)
  y    = np.array([v[1] for v in valid], dtype=float)
  z    = np.array([v[2] for v in valid], dtype=float)
  errx = np.array([v[3] for v in valid], dtype=float)
  erry = np.array([v[4] for v in valid], dtype=float)
  errz = np.array([v[5] for v in valid], dtype=float)

  # Calculate R and errR
  R = np.sqrt(x**2 + y**2)
  errR = np.sqrt((x * errx)**2 + (y * erry)**2) / R

  # Determine theta by linear fit
  graph = ROOT.TGraphErrors(len(valid), z, R, errz, errR)
  fit_func = ROOT.TF1("fit_func", "[0]*x + [1]", min(z), max(z))
  graph.Fit(fit_func, "QN")  # quiet fit

  slope = fit_func.GetParameter(0)
  slope_err = fit_func.GetParError(0)

  theta = np.arctan(slope)
  theta_err = np.arctan(slope + slope_err) - theta

  # Retrieve true track angle
  theta_true = event_data["mc_track"][4]  # radians

  if plot:
    if filename is None:
      filename = f"average_hits_{detector}.png"

    graph.SetTitle(f"Average {detector} Hit Position; Z [mm]; R [mm]")
    graph.SetMarkerStyle(20)
    if detector == "HCal": 
      graph.SetMarkerColor(ROOT.kBlack)
      graph.SetLineColor(ROOT.kBlack)
    else: 
      graph.SetMarkerColor(ROOT.kBlue)
      graph.SetLineColor(ROOT.kBlue)

    pad = 50
    graph.GetXaxis().SetLimits(min(z) - pad, max(z) + pad)
    graph.GetYaxis().SetRangeUser(min(R) - pad, max(R) + pad)

    # Draw canvas
    c1 = ROOT.TCanvas("c1", f"Average HCal Hit Positions", 1000, 800)
    c1.SetLeftMargin(0.15)    # make space for Y-axis label
    c1.SetBottomMargin(0.15)  # make space for X-axis label
    graph.Draw("AP")
    c1.Update()

    # Plot the true track as solid line
    mc = event_data["mc_track"]
    dx, dy, dz, R_true, theta_true, phi_true, _ = mc
    slope = R_true / dz  # slope in R vs Z

    # ZDC range
    z_start = min(z) - 50
    z_end   = max(z) + 50 
    R_start = slope * (z_start - 0) 
    R_end   = slope * (z_end - 0)

    # Draw line
    true_line = ROOT.TLine(z_start, R_start, z_end, R_end)
    if detector == "HCal":
      true_line.SetLineColor(ROOT.kBlack)
    else: 
      true_line.SetLineColor(ROOT.kBlue)
    true_line.SetLineStyle(2)  # dotted
    true_line.SetLineWidth(2)
    true_line.Draw("SAME")

    # Fit function 
    #fit_func.SetLineColor(ROOT.kBlack) 
    #fit_func.Draw("SAME")

    # Add legend
    legend = ROOT.TLegend(0.65, 0.75, 0.9, 0.9)
    legend.AddEntry(graph, "Average hits", "p")
    legend.AddEntry(true_line, "True track", "l")
    legend.SetFillColorAlpha(ROOT.kWhite, 0.4)
    legend.SetTextSize(0.03)
    legend.Draw()

    c1.Update()
    c1.SaveAs(filename)
    c1.Close()

  return theta, theta_err, theta_true

def ang_res_phi(event_data, filename=None, detector="HCal", plot=True):
  if detector == "HCal":
    average_layers = event_data["average_layers_HCal"]
    error_layers   = event_data["error_layers_HCal"]
  elif detector == "WSi":
    average_layers = event_data["average_layers_WSi"]
    error_layers   = event_data["error_layers_WSi"]
  else: 
    print("Unknown detector. Please enter HCal or WSi.")
    return None, None, None

  # Filter out layers with missing averages
  valid = []
  for (ax, ay, az), (ex, ey, ez) in zip(average_layers, error_layers):
    if ax is not None and ay is not None:
      valid.append((ax, ay, ex, ey))

  if len(valid) == 0:
    return None, None, None

  # Convert to arrays
  x    = np.array([v[0] for v in valid], dtype=float)
  y    = np.array([v[1] for v in valid], dtype=float)
  errx = np.array([v[2] for v in valid], dtype=float)
  erry = np.array([v[3] for v in valid], dtype=float)

  # Perform linear fit: y = m * x + b
  graph = ROOT.TGraphErrors(len(valid), x, y, errx, erry)
  fit_func = ROOT.TF1("fit_func_phi", "[0]*x + [1]", min(x), max(x))
  graph.Fit(fit_func, "QN")  # quiet fit

  slope = fit_func.GetParameter(0)
  slope_err = fit_func.GetParError(0)

  phi = np.arctan(slope)
  phi_err = np.arctan(slope + slope_err) - phi

  # Retrieve true track phi from MC
  phi_true = event_data["mc_track"][5]  # radians

  if plot:
    if filename is None:
      filename = f"average_hits_phi_{detector}.png"

    graph.SetTitle(f"Average {detector} Hit Positions; X [mm]; Y [mm]")
    graph.SetMarkerStyle(20)
    if detector == "HCal": 
      graph.SetMarkerColor(ROOT.kBlack)
      graph.SetLineColor(ROOT.kBlack)
    else: 
      graph.SetMarkerColor(ROOT.kBlue)
      graph.SetLineColor(ROOT.kBlue)

    pad = 50
    graph.GetXaxis().SetLimits(min(x) - pad, max(x) + pad)
    graph.GetYaxis().SetRangeUser(min(y) - pad, max(y) + pad)

    # Draw canvas
    c1 = ROOT.TCanvas("c1", f"Average {detector} Hit Positions", 1000, 800)
    c1.SetLeftMargin(0.15)
    c1.SetBottomMargin(0.15)
    graph.Draw("AP")
    c1.Update()

    # Draw MC track line
    mc = event_data["mc_track"]
    x0, y0, dx, dy, theta_mc, phi_mc, _ = mc 
    slope_mc = dy / dx

    x_start = min(x) - 50
    x_end   = max(x) + 50
    y_start = slope_mc * (x_start - x0) + y0
    y_end   = slope_mc * (x_end - x0) + y0

    true_line = ROOT.TLine(x_start, y_start, x_end, y_end)
    if detector == "HCal":
      true_line.SetLineColor(ROOT.kBlack)
    else:
      true_line.SetLineColor(ROOT.kBlue)
    true_line.SetLineStyle(2)  # dotted
    true_line.SetLineWidth(2)
    #true_line.Draw("SAME")

    # Add legend
    legend = ROOT.TLegend(0.65, 0.75, 0.9, 0.9)
    legend.AddEntry(graph, "Average hits", "p")
    #legend.AddEntry(true_line, "MC track", "l")
    legend.SetFillColorAlpha(ROOT.kWhite, 0.4)
    legend.SetTextSize(0.03)
    legend.Draw()

    c1.Update()
    c1.SaveAs(filename)
    c1.Close()

  return phi, phi_err, phi_true

# ******************************************************
#                 Plotting functions 
# ******************************************************
# Plot a 3D hitmap of the raw hits and average hit position per layer (per event)
def plot_hits_3d(event_data, n_WSi_layers=20, elev=20, azim=-25, filename="event_hits_slices.png"):
  event_layers = event_data["event_layers"]
  average_layers_HCal = event_data["average_layers_HCal"]
  error_layers_HCal = event_data["error_layers_HCal"]
  average_layers_WSi = event_data["average_layers_WSi"]
  error_layers_WSi = event_data["error_layers_WSi"]

  fig = plt.figure(figsize=(12, 8))
  ax = fig.add_subplot(111, projection='3d')

  # --- Plot all hits ---
  for layer_idx, layer_hits in enumerate(event_layers):
    if len(layer_hits) > 0:
      x_hits = [hit[0] for hit in layer_hits]
      y_hits = [layer_idx] * len(layer_hits)
      z_hits = [hit[1] for hit in layer_hits]

      color = '#9999FF' if layer_idx < n_WSi_layers else 'gray'
      ax.scatter(x_hits, y_hits, z_hits, c=color, marker='o', s=10, alpha=0.5)

  # --- Plot WSi averages + error bars ---
  if average_layers_WSi is not None and error_layers_WSi is not None:
    for layer_idx, (avg, err) in enumerate(zip(average_layers_WSi, error_layers_WSi)):
      avg_x, avg_y, avg_z = avg

      if avg_x is None or avg_y is None:
        continue

      err_x, err_y, err_z = err

      xerr = err_x if err_x is not None else 0
      zerr = err_y if err_y is not None else 0

      plot_layer_idx = layer_idx

      ax.errorbar(avg_x, plot_layer_idx, avg_y,
                  xerr=xerr, yerr=0, zerr=zerr,
                  fmt='o', color='blue', capsize=3)

  # --- Plot HCal averages + error bars ---
  for layer_idx, (avg, err) in enumerate(zip(average_layers_HCal, error_layers_HCal)):
    avg_x, avg_y, avg_z = avg

    if avg_x is None or avg_y is None:
      continue

    err_x, err_y, err_z = err

    xerr = err_x if err_x is not None else 0
    zerr = err_y if err_y is not None else 0

    plot_layer_idx = layer_idx + n_WSi_layers

    ax.errorbar(avg_x, plot_layer_idx, avg_y,
                xerr=xerr, yerr=0, zerr=zerr,
                fmt='o', color='black', capsize=3)

  ax.set_xlabel('X [mm]')
  ax.set_ylabel('ZDC Layer')
  ax.set_zlabel('Y [mm]')

  max_layer = max(i for i, layer in enumerate(event_layers) if len(layer) > 0)
  ax.set_xlim(-1050, -700)
  ax.set_ylim(0, max_layer + 1)
  ax.set_zlim(-200, 200)

  ax.view_init(elev=elev, azim=azim)
  ax.set_box_aspect([1, 2.5, 1])

  # Updated legend
  legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='WSi hits', markerfacecolor='#9999FF', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='WSi average', markerfacecolor='blue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='HCal hits', markerfacecolor='gray', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='HCal average', markerfacecolor='black', markersize=8)
  ]
  ax.legend(handles=legend_elements, loc='upper right')

  plt.tight_layout()
  plt.savefig(filename)
  plt.close()

# ***************************
# Histogram of difference between true and reconstructed theta values (all events)
def plot_theta_err(event_data, detector="HCal"):
  delta_theta_list = []

  for event in event_data:
    theta_reco, theta_err, theta_true = ang_res_theta(event, detector=detector, plot=False)
    if theta_reco is not None:
      delta_theta_list.append(theta_reco - theta_true)

  delta_theta_arr = np.array(delta_theta_list)
  n_events = len(delta_theta_arr)
  print(f"Total events in histogram: {n_events}")


  # Automatically generate filename if not provided
  filename = f"theta_err_{detector}.png"

  plt.figure(figsize=(8,6))
  if detector == "HCal":
    color = 'grey'
  elif detector == "WSi":
    color = 'skyblue'
  else:
    print("Unknown detector. Using default color.")
    color = 'grey'

  plt.hist(delta_theta_arr*1e3, bins=20, alpha=0.7, color=color, edgecolor='black')  # mrad
  plt.xlabel(r'$\theta_\mathrm{reco} - \theta_\mathrm{true}$ [mrad]')
  plt.ylabel('Number of Events')
  plt.title(f'{detector} Angular Resolution')
  plt.grid(True, linestyle='--', alpha=0.5)
  plt.tight_layout()
  plt.savefig(filename)

# ******************************************************
#                      Main 
# ******************************************************
def main():
  # Open the ROOT file
  chain = ROOT.TChain("events")
  chain.Add("nb11_10k_80_260GeV.edm4hep.root")
  chain.Add("nb12_10k_80_260GeV.edm4hep.root")
  chain.Add("nb13_10k_80_260GeV.edm4hep.root")
  chain.Add("nb21_10k_80_260GeV.edm4hep.root")
  chain.Add("nb22_10k_80_260GeV.edm4hep.root")
  chain.Add("nb23_10k_80_260GeV.edm4hep.root")

  # Extract the data, detremine rconstruted and true theta values 
  master_list = extract_data(chain) 
  event_num = 3
  #print(master_list[event_num]["error_layers_WSi"])

  # Make plots and get theta values for a single event 
  """  
  theta, theta_err, theta_true = ang_res_theta(master_list[event_num], filename=f"e{event_num}_avg_hits_HCal_theta.png", plot=True) #default is HCal
  theta_WSi, theta_err_WSi, theta_true_WSi = ang_res_theta(master_list[event_num], detector="WSi", filename=f"e{event_num}_avg_hits_WSi_theta.png", plot=True) 
  phi, phi_err, phi_true = ang_res_phi(master_list[event_num], filename=f"e{event_num}_avg_hits_HCal_phi.png", plot=True) #default is HCal
  plot_hits_3d(master_list[event_num], filename=f"e{event_num}_iso_hits.png") # isometric view 
  plot_hits_3d(master_list[event_num], elev=0, azim=0, filename=f"e{event_num}_flat_hits.png") # view in ZY plane 
  if theta is None:
    print("HCal Reconstructed:  Not enough hits in HCal")
  else:
    print(f"True Theta:          {theta_true:.6f}")
    print(f"HCal Reconstructed:  {theta:.6f} +/- {theta_err:.6f}")
  if theta_WSi is None:
    print("WSi Reconstructed:   Not enough hits in WSi")
  else:
    print(f"WSi Reconstructed:   {theta_WSi:.6f} +/- {theta_err_WSi:.6f}")"""
  

  # Make plots for the angular reconstruction of all events  
  plot_theta_err(master_list, detector="HCal")
  plot_theta_err(master_list, detector="WSi")

if __name__ == "__main__":
  main()