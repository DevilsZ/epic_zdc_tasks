import math
from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import cm, mm, GeV, MeV, degree, radian
SIM = DD4hepSimulation()

particle = "neutron"

ionCrossingAngle = -0.025 * radian
ZDC_r_pos = 3579 * cm
ZDC_x_pos = ZDC_r_pos * math.sin(-0.025)
ZDC_y_pos = 0 * cm
ZDC_z_pos = ZDC_r_pos * math.cos(-0.025)


#SIM.numberOfEvents = 1000
    
SIM.enableGun = True
SIM.gun.position = (0.0, 0.0, 0.0)
SIM.gun.particle = particle
SIM.gun.direction = (math.sin(-0.025), 0, math.cos(-0.025))
SIM.gun.phiMin = -0
SIM.gun.phiMax = -0
SIM.gun.thetaMin = -0.025
SIM.gun.thetaMax = -0.025
SIM.gun.distribution = "uniform"
SIM.gun.multiplicity = 1


SIM.gun.isotrop = True

SIM.gun.momentumMin = 50*GeV
SIM.gun.momentumMax = 50*GeV

SIM.physics.list = "FTEP_BERT"
SIM.physics.rangecut = 0.7

