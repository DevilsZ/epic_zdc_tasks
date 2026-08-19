import ROOT
import random
import math
from os import path
from ROOT import TCanvas, TFile, TPaveText
from ROOT import gROOT, gBenchmark
from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import mm, GeV, MeV, rad, m

CrossingAngle = -0.025 ##-0.025
SIM = DD4hepSimulation()
## The compact XML file, or multiple compact files, if the last one is the closer.
SIM.compactFile = []
## Lorentz boost for the crossing angle, in radian!
SIM.crossingAngleBoost = CrossingAngle * rad
SIM.enableDetailedShowerMode = False
SIM.enableG4GPS = False
SIM.enableG4Gun = False
SIM.enableGun = False
## InputFiles for simulation .stdhep, .slcio, .HEPEvt, .hepevt, .pairs, .hepmc, .hepmc.gz, .hepmc.xz, .hepmc.bz2, .hepmc3, .hepmc3.gz, .hepmc3.xz, .hepmc3.bz2, .hepmc3.tree.root files are supported
SIM.inputFiles = []
## Macro file to execute for runType 'run' or 'vis'
SIM.macroFile = ""
## number of events to simulate, used in batch mode
SIM.numberOfEvents = 10
## Outputfile from the simulation: .slcio, edm4hep.root and .root output files are supported
SIM.outputFile = "dummyOutput_test.root"
## Physics list to use in simulation
SIM.physicsList = None
## Verbosity use integers from 1(most) to 7(least) verbose
## or strings: VERBOSE, DEBUG, INFO, WARNING, ERROR, FATAL, ALWAYS
SIM.printLevel = 3
## The type of action to do in this invocation
## batch: just simulate some events, needs numberOfEvents, and input file or gun
## vis: enable visualisation, run the macroFile if it is set
## qt: enable visualisation in Qt shell, run the macroFile if it is set
## run: run the macroFile and exit
## shell: enable interactive session
SIM.runType = "batch"
## Skip first N events when reading a file
SIM.skipNEvents = 0
## Steering file to change default behaviour
SIM.steeringFile = None
## FourVector of translation for the Smearing of the Vertex position: x y z t
SIM.vertexOffset = [0.0, 0.0, 0.0, 0.0]
## FourVector of the Sigma for the Smearing of the Vertex position: x y z t
SIM.vertexSigma = [0.0, 0.0, 0.0, 0.0]

################################################################################
## Configuration for the GuineaPig InputFiles 
################################################################################

## Set the number of pair particles to simulate per event.
##     Only used if inputFile ends with ".pairs"
##     If "-1" all particles will be simulated in a single event
##     


################################################################################
## Configuration for the DDG4 ParticleGun 
################################################################################

##  direction of the particle gun, 3 vector 
##SIM.gun.direction =(0, 0, 1)

## choose the distribution of the random direction for theta
## 
##     Options for random distributions:
## 
##     'uniform' is the default distribution, flat in theta
##     'cos(theta)' is flat in cos(theta)
##     'eta', or 'pseudorapidity' is flat in pseudorapity
##     'ffbar' is distributed according to 1+cos^2(theta)
## 
##     Setting a distribution will set isotrop = True
##     
SIM.gun.distribution = "uniform"

## Total energy (including mass) for the particle gun.
## 
## If not None, it will overwrite the setting of momentumMin and momentumMax
SIM.gun.energy = None

## Maximal pseudorapidity for random distibution (overrides thetaMin)
SIM.gun.etaMax = None

## Minimal pseudorapidity for random distibution (overrides thetaMax)
SIM.gun.etaMin = None   

##  isotropic distribution for the particle gun
## 
##     use the options phiMin, phiMax, thetaMin, and thetaMax to limit the range of randomly distributed directions
##     if one of these options is not None the random distribution will be set to True and cannot be turned off!
##     
SIM.gun.isotrop = False

## Maximal momentum when using distribution (default = 0.0)
SIM.gun.momentumMax = 200 *GeV

## Minimal momentum when using distribution (default = 0.0)
SIM.gun.momentumMin = 200 *GeV

SIM.gun.multiplicity = 1
SIM.gun.particle = "neutron"

## Maximal azimuthal angle for random distribution
SIM.gun.phiMax = None

## Minimal azimuthal angle for random distribution
SIM.gun.phiMin = None

##  position of the particle gun, 3 vector 
z = 0
x = math.tan(CrossingAngle * rad) * z 
y = math.tan(0.0036 * rad) * z

SIM.gun.position = (x * mm, y * mm, z * mm)

## Maximal polar angle for random distribution
SIM.gun.thetaMax = 0.004 * rad

## Minimal polar angle for random distribution
SIM.gun.thetaMin = 0.0 * rad

# SIM.gun.phiMin = 0.5 * 3.141592 * rad
# SIM.gun.phiMax = 0.5 * 3.141592 * rad




################################################################################
## Configuration for setting commands to run during different phases.
## 
##   In this section, one can configure commands that should be run during the different phases of the Geant4 execution.
## 
##   1. Configuration
##   2. Initialization
##   3. Pre Run
##   4. Post Run
##   5. Terminate / Finalization
## 
##   For example, one can add
## 
##   >>> SIM.ui.commandsConfigure = ['/physics_lists/em/SyncRadiation true']
## 
##   Further details should be taken from the Geant4 documentation.
##    
################################################################################
SIM.ui.commandsConfigure = [
    "/particle/select lambda",                        
    "/particle/property/decay/dump",                 
    "/particle/property/decay/select 0",             
    "/particle/property/decay/br 0",                 
    "/particle/property/decay/select 1",             
    "/particle/property/decay/br 1",                                        
    "/particle/property/decay/dump",                  
    ]
SIM.ui.commandsInitialize = []
SIM.ui.commandsPostRun = []
SIM.ui.commandsPreRun = []
SIM.ui.commandsTerminate=[]
