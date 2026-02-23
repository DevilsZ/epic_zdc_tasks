from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import*
import math
SIM = DD4hepSimulation()

# Simulation setup
SIM.enableGun = True

# Particle gun properties 
SIM.gun.particle = "neutron"
SIM.gun.momentumMin = 80*GeV 
SIM.gun.momentumMax = 260*GeV 
SIM.gun.multiplicity = 1

SIM.gun.position = (0, 0, 33320)

#shoot particles at an angle 
SIM.gun.distribution = "cos(theta)"
SIM.gun.thetaMin = 0.276*rad
SIM.gun.thetaMax = 0.290*rad

# Number of events 
SIM.numberOfEvents = 10000