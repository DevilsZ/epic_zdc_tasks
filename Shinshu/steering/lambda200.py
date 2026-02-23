import math
from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import cm, mm, GeV, MeV, degree, radian
SIM = DD4hepSimulation()

SIM.ui.commandsConfigure = [
    "/particle/select lambda",
    # dump before
    "/particle/property/decay/dump",
    # proton pi-
    "/particle/property/decay/select 0",
    "/particle/property/decay/br 0",
    # neutron pi0
    "/particle/property/decay/select 1",
    "/particle/property/decay/br 1",
    # dump after
    "/particle/property/decay/dump",
]
SIM.ui.commandsInitialize = []
SIM.ui.commandsPostRun = []
SIM.ui.commandsPreRun = []
SIM.ui.commandsTerminate = []
particle = "lambda"

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

SIM.gun.momentumMin = 200*GeV
SIM.gun.momentumMax = 200*GeV

SIM.physics.list = "FTEP_BERT"
SIM.physics.rangecut = 0.7

