#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue September 28 
Purpose: Minimize, equilibrate, and produce run type 2 IP3R structure complete with AF by NP

Version 1.5 : 

Surface tension = 0
Total sim time = 500 ns
"""

from sys import stdout

import mdtraj as md
import numpy as np
import simtk.openmm as mm
import simtk.openmm.app as app
from simtk.openmm import LangevinMiddleIntegrator  
from simtk.openmm.app import CharmmPsfFile, PDBFile, CharmmParameterSet
from openmmtools.integrators import LangevinIntegrator
import simtk.unit as unit
import os, time
import bz2

# Load CHARMM files
filePath = 'input/'
psf = CharmmPsfFile(filePath + 'step5_input.psf')
pdb = PDBFile(filePath + 'step5_input.pdb')

param_filenames = ['par_all36m_prot.prm', 'top_all36_prot.rtf',
                       'par_all36_lipid.prm', 'top_all36_lipid.rtf',
                       'toppar_water_ions.str']
param_paths = [os.path.join(filePath, file) for file in param_filenames]
params = CharmmParameterSet(*param_paths)

## X Y and Z values retrieved from the CHARMM-GUI interface
## The barostat will take care of any inaccuracies.
psf.setBox(12.8, 12.8, 11.4) # Change for different charmm output etc

## Setting system parameters we will use later
nonbonded_method = app.PME
constraints = app.HBonds
hydrogen_mass = hydrogenMass=4.0 * unit.amu
temperature = 303.15*unit.kelvin
friction = 1/unit.picosecond
time_step = 0.004*unit.picoseconds
pressure = 1*unit.bar
surface_tension = 0*unit.bar*unit.nanometer

## Setting up the system using the psf file
system = psf.createSystem(params, 
                          nonbondedMethod=nonbonded_method, 
                          constraints=constraints,
                          removeCMMotion=False,
                          hydrogenMass=hydrogen_mass
                         )

## Setting the integrator and barostat

# integrator = LangevinIntegrator(temperature,   # Temperature of head bath
#                                 friction, # Friction coefficient
#                                 time_step) # Time step

integrator = LangevinMiddleIntegrator(temperature,        ## this is new, suggested by John  
                                     friction,
                                     time_step)

barostat = mm.MonteCarloMembraneBarostat(pressure,
                                         surface_tension, 
                                         temperature,
                                         mm.MonteCarloMembraneBarostat.XYIsotropic, 
                                         mm.MonteCarloMembraneBarostat.ZFree
                                        )
barostat.setFrequency(50)    ## for some reason the __init__ won't accept it as an argument, but this works
                            ## the default is 25 timesteps, i've set it for 50
system.addForce(barostat)

simulation = app.Simulation(psf.topology, system, integrator)
simulation.context.setPositions(pdb.positions)

## Run minimization
print(
    "  initial : %8.3f kcal/mol"
    % (
        simulation.context.getState(getEnergy=True).getPotentialEnergy()
        / unit.kilocalories_per_mole
    )
)
simulation.minimizeEnergy()
print(
    "  final : %8.3f kcal/mol"
    % (
        simulation.context.getState(getEnergy=True).getPotentialEnergy()
        / unit.kilocalories_per_mole
    )
)

## Saving xml and pdb files after minimization
## This isn't strictly necessary but it is good to know how to do
## and is useful for saving time while troubleshooting.
output_prefix = 'output/'
integrator_xml_filename = "minimized-integrator.xml"
state_xml_filename = "minimized-state.xml"
system_xml_filename = "minimized-system.xml"
minimized_pdb = "minimized.pdb"

# Save and serialize the minimization state
print("Serializing state to %s" % state_xml_filename)
state = simulation.context.getState(
    getPositions=True,
    getVelocities=True,
    getEnergy=True,
    getForces=True
)
with open(output_prefix + state_xml_filename, "w") as outfile:
    xml = mm.XmlSerializer.serialize(state)
    outfile.write(xml)
    
# Save and serialize system
print("Serializing system to %s" % system_xml_filename)
system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
with open(output_prefix + system_xml_filename, "w") as outfile:
    xml = mm.XmlSerializer.serialize(system)
    outfile.write(xml)
    
# Save integrator
print("Serializing integrator to %s" % integrator_xml_filename)
with open(output_prefix + integrator_xml_filename, "w") as outfile:
    xml = mm.XmlSerializer.serialize(integrator)
    outfile.write(xml)

print("Writing system to %s" % minimized_pdb)
with open(output_prefix+minimized_pdb, "w") as outfile:
    PDBFile.writeFile(
        simulation.topology,
        simulation.context.getState(
            getPositions=True,
            enforcePeriodicBox=True).getPositions(),
            file=outfile,
            keepIds=True
    )
    
## Load System from XML Files
output_prefix = 'output/'
integrator_xml_filename = "minimized-integrator.xml"
state_xml_filename = "minimized-state.xml"
system_xml_filename = "minimized-system.xml"
filePath = 'input/'
psf = CharmmPsfFile(os.path.join(filePath, 'step5_input.psf'))
pdb = PDBFile(os.path.join(filePath, 'step5_input.pdb'))
systemFile = output_prefix+system_xml_filename
intFile = output_prefix+integrator_xml_filename
stateFile = output_prefix+state_xml_filename

sim = app.Simulation(psf.topology, 
                     system=systemFile, 
                     integrator=intFile,
                     state=stateFile
                     )

## Run Equilibration
# Set steps and frequencies
nsteps = 125000000 # 500 ns
report_freq = 25000 # Report every 0.1 ns
chk_freq = 12500000 # Report every 50 ns
traj_freq = 250000  # 100 frames; 1 ns at 4 fs / step

# Set file names
output_prefix = 'output/'
state_data_filename = 'state_data'
integrator_xml_filename = "integrator.xml"
state_xml_filename = "state.xml"
state_pdb_filename = "equilibrated.pdb"
system_xml_filename = "system.xml"
checkpoint_filename = "equilibrated.chk"
traj_output_filename = "equilibrated.dcd"

# set starting velocities:
print("Generating random starting velocities")
sim.context.setVelocitiesToTemperature(temperature)

# write limited state information to standard out:
sim.reporters.append(
    app.StateDataReporter(
        output_prefix + state_data_filename , # Change to write out to a seperate file (as CSV or something else)
        reportInterval=report_freq,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        temperature=True,
        speed=True,
        progress=True,
        remainingTime=True,
        totalSteps=nsteps,
        separator="\t",
    )
)

# Write to checkpoint files regularly:
sim.reporters.append(app.CheckpointReporter(
    file=output_prefix + checkpoint_filename,
    reportInterval=chk_freq
    )
)

# Write out the trajectory

sim.reporters.append(md.reporters.DCDReporter(
    file=output_prefix + traj_output_filename,
    reportInterval=traj_freq
    )
)

# Run NPT dynamics
print("Running dynamics in the NPT ensemble...")
initial_time = time.time()
sim.step(nsteps)
elapsed_time = (time.time() - initial_time) * unit.seconds
simulation_time = nsteps * time_step
print('    Equilibration took %.3f s for %.3f ns (%8.3f ns/day)' % (elapsed_time / unit.seconds, simulation_time / unit.nanoseconds, simulation_time / elapsed_time * unit.day / unit.nanoseconds))

# Save and serialize the final state
print("Serializing state to %s" % state_xml_filename)
state = sim.context.getState(
    getPositions=True,
    getVelocities=True,
    getEnergy=True,
    getForces=True
)
with bz2.open(output_prefix + state_xml_filename, "wt") as outfile:
    xml = mm.XmlSerializer.serialize(state)
    outfile.write(xml)

# Save the final state as a PDB
print("Saving final state as %s" % state_pdb_filename)
with open(output_prefix + state_pdb_filename, "wt") as outfile:
    PDBFile.writeFile(
        sim.topology,
        sim.context.getState(
            getPositions=True,
            enforcePeriodicBox=True).getPositions(),
            file=outfile,
            keepIds=True
    )

# Save and serialize system
print("Serializing system to %s" % system_xml_filename)
system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
with bz2.open(output_prefix + system_xml_filename, "wt") as outfile:
    xml = mm.XmlSerializer.serialize(system)
    outfile.write(xml)

# Save and serialize integrator
print("Serializing integrator to %s" % integrator_xml_filename)
with bz2.open(output_prefix + integrator_xml_filename, "wt") as outfile:
    xml = mm.XmlSerializer.serialize(integrator)
    outfile.write(xml)
