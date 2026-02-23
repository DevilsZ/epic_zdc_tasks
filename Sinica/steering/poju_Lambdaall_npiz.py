
#############################################
import math

from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import cm, mm, GeV, MeV, degree, radian
SIM = DD4hepSimulation()



SIM.ui.commandsConfigure = [
    "/particle/select lambda",
    #dump before
    "/particle/property/decay/dump",
    #proton pi-
    "/particle/property/decay/select 0",
    "/particle/property/decay/br 0",
    #neutron pi0
    "/particle/property/decay/select 1",
    "/particle/property/decay/br 1",
    #dump after
    "/particle/property/decay/dump",
]

SIM.ui.commandsInitialize = []
SIM.ui.commandsPostRun = []
SIM.ui.commandsPreRun = []
SIM.ui.commandsTerminate = []

SIM.numberOfEvents = 1000
SIM.enableGun = True
#SIM.outputFile = "neutron_test.slcio"                                          

SIM.gun.particle = "lambda"       
SIM.gun.momentumMin = 10*GeV
SIM.gun.momentumMax = 275*GeV
SIM.gun.direction = (math.sin(-0.025), 0, math.cos(-0.025))
#############################################


#SIM.part.select = "lambda"
#SIM.part.help

