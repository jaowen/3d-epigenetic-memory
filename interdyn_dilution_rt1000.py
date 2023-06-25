from __future__ import absolute_import, division, print_function, unicode_literals
import os,sys
import polychrom
from polychrom import (simulation, starting_conformations,
                       forces, forcekits)
import simtk.openmm as openmm
import simtk.unit
import os
import polychrom.polymerutils as polymerutils
import numpy as np
import random
import math
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
import csv
import networkx as nx
import EoN
import copy

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def simple_marks_SSW(
    sim_object,
    marks,
    repulsionEnergy=60.0,  # base repulsion energy for **all** particles
    repulsionRadius=1.0,
    attractionEnergy=0.0,  # base attraction energy for **all** particles
    attractionRadius=1.5,
    selectiveAttractionEnergy=0.3,  # **extra** attractive energy for **sticky** particles
    name="simple_selective_SSW",
):
    # based on selective_SSW

    energy = (
        "step(REPsigma - r) * Erep + step(r - REPsigma) * Eattr;"
        ""
        "Erep = rsc12 * (rsc2 - 1.0) * REPeTot / emin12 + REPeTot;" 
        "REPeTot = REPe;"
        "rsc12 = rsc4 * rsc4 * rsc4;"
        "rsc4 = rsc2 * rsc2;"
        "rsc2 = rsc * rsc;"
        "rsc = r / REPsigma * rmin12;"
        ""
        "Eattr = - rshft12 * (rshft2 - 1.0) * ATTReTot / emin12 - ATTReTot;"
        "ATTReTot = ATTRe + min((m1+1.0)/2.0, (m2+1.0)/2.0) * ATTReAdd;"
        "rshft12 = rshft4 * rshft4 * rshft4;"
        "rshft4 = rshft2 * rshft2;"
        "rshft2 = rshft * rshft;"
        "rshft = (r - REPsigma - ATTRdelta) / ATTRdelta * rmin12;"
        ""
    )

    force = openmm.CustomNonbondedForce(energy)
    force.name = name

    force.setCutoffDistance(attractionRadius * sim_object.conlen)

    force.addGlobalParameter("REPe", repulsionEnergy * sim_object.kT)
    force.addGlobalParameter("REPsigma", repulsionRadius * sim_object.conlen)

    force.addGlobalParameter("ATTRe", attractionEnergy * sim_object.kT)
    force.addGlobalParameter("ATTReAdd", selectiveAttractionEnergy * sim_object.kT)
    force.addGlobalParameter(
        "ATTRdelta", sim_object.conlen * (attractionRadius - repulsionRadius) / 2.0
    )

    # Coefficients for x^12*(x*x-1)
    force.addGlobalParameter("emin12", 46656.0 / 823543.0)
    force.addGlobalParameter("rmin12", np.sqrt(6.0 / 7.0))

    force.addPerParticleParameter("m")

    for i in range(sim_object.N):
        force.addParticle(([marks[i]]))

    return force


def update_marks_SIS(pts, marks, r, s, ET, p3D, tmax):
    
    # updates marks according to SIS dynamics on network with spread rate s, recovery rate 1
    # network is built from list of points according to radius r 

    n = len(marks)

    tree = cKDTree(pts)
    dists, neighbors = tree.query(pts, n, distance_upper_bound=r)
    adjlist = neighbors.tolist()
    for ind in range(n):
        adjlist[ind] = [val for val in adjlist[ind] if val != n]
        adjlist[ind] = [val for val in adjlist[ind] if val != ind]

    G = nx.Graph()
    for ind in range(n):
        G.add_node(ind)
        for node in adjlist[ind]:
            if (ind == node + 1) or (node == ind + 1):
                G.add_edge(ind, node, weight = 1)
            else:
                G.add_edge(ind, node, weight = p3D)
 

    #G = nx.path_graph(n)

    for _ in range(tmax):

        marked_sites = []
        for i, item in enumerate(marks):
            if item > 0:
                marked_sites.append(i)
        
        ST = 0.001
        for (i, j) in G.edges():
            if marks[i]*marks[j] < 0: ST += 1
        if ST < ET:
            pb = 1.0
        else:
            pb = ET / ST

        msim = EoN.fast_SIS(G, s*pb, 0, initial_infecteds=marked_sites, tmax=1.0, transmission_weight='weight', return_full_data=True)
        #msim = EoN.fast_SIS(G, s*pb, 1.0, initial_infecteds=marked_sites, tmax=1.0, return_full_data=True)
        #get new marks from sim
        stat = msim.get_statuses(time=1.0)
        for key in stat:
            if stat[key] == 'I':
                marks[key] = 1.0
            else:
                marks[key] = -1.0

    degrees = [len(part) for part in adjlist]

    return marks, degrees


def send_marks_to_sim(sim_object, mark_force, marks):

    for i in range(sim_object.N):
        mark_force.setParticleParameters(i, ([marks[i]]))
    
    mark_force.updateParametersInContext(sim_object.context)

def relax_polymer(sim, relaxer, time):
    pts = sim.get_data()
    alphaval = sim.context.getParameter("simple_selective_SSW_ATTReAdd")
    relaxer.set_data(pts)
    relaxer._apply_forces()

    relaxer.state = relaxer.context.getState(
        getPositions=False, getVelocities=False, getEnergy=False
           )
    curtime = relaxer.state.getTime() / simtk.unit.picosecond
    relaxer.integrator.stepTo(curtime + time)
    relaxer.state = relaxer.context.getState(
        getPositions=True, getVelocities=True, getEnergy=True
           )
    #relaxer.data = relaxer.state.getPositions(asNumpy=True)
    relaxer.set_data(relaxer.state.getPositions(asNumpy=True))
    sim.set_data(relaxer.state.getPositions(asNumpy=True))
    #print(relaxer.context.getParameter("simple_selective_SSW_ATTReAdd"))
    sim.reinitialize()
    relaxer.reinitialize()
    relaxer.context.setParameter("simple_selective_SSW_ATTReAdd", alphaval)
    sim.context.setParameter("simple_selective_SSW_ATTReAdd", alphaval)

def run_polymer_dynamics(sim, steps):
    sim._apply_forces()
    sim.integrator.step(steps)
    sim.state = sim.context.getState(
        getPositions=True, getVelocities=True, getEnergy=True
           )
    sim.set_data(sim.state.getPositions(asNumpy=True))

def initial_polymer_state(sim, size):
    polymer = starting_conformations.grow_cubic(size, 200)
    sim.set_data(polymer, center=True)
    sim.local_energy_minimization()

def central_domain(m, totalsize):
    left = totalsize//2 - m//2
    right = totalsize - left - m
    return [-1.0]*left + [1.0]*m + [-1.0]*right

def count_marks(marks):
    count = 0
    for m in marks:
        if m > 0: 
            count += 1
    return count

def main():

    gpuid = sys.argv[1]
    alpha10 = int(sys.argv[2])
    ET = int(sys.argv[3])
    dynspeed = int(sys.argv[4])
    gens = int(sys.argv[5])

    size = 10000
    desired_density = 0.05
    collrate = 2.0
    alpha = 0 / 10.0
    p3D = 100 / 100.0
    rtime = 1000

    r = 1.5
    #tmax = 200

    relaxer = simulation.Simulation(
            platform="CUDA",
            precision="single",
            integrator="variableLangevin",
            GPU = gpuid,
            collision_rate=0.01,
            error_tol=0.0005,
            N = size)

    sim = simulation.Simulation(
            platform="CUDA",
            precision="single",
            integrator="brownian",
            GPU = gpuid,
            collision_rate=collrate,
            timestep = 10,
            N = size)


    # CONFINEMENT
    particle_radius = 0.95 / 2.0
    desired_radius = particle_radius*((size)**(1.0/3.0)) / (desired_density**(1.0/3.0))
    sim.add_force(forces.spherical_confinement(sim, r=desired_radius, k=5))
    relaxer.add_force(forces.spherical_confinement(sim, r=desired_radius, k=5))

    # CHAIN BONDS
    sim.add_force(
    forcekits.polymer_chains(
        sim,
        chains=[(0, None, False)],

        bond_force_func=forces.harmonic_bonds,
        bond_force_kwargs={
            'bondLength':1.0,
            'bondWiggleDistance':0.1, # Bond distance will fluctuate +- 0.05 on average
         },

        angle_force_func=None,
        angle_force_kwargs={},

        nonbonded_force_func=None,
        nonbonded_force_kwargs={},

        except_bonds=True,
        )
     )

    relaxer.add_force(
    forcekits.polymer_chains(
        relaxer,
        chains=[(0, None, False)],

        bond_force_func=forces.harmonic_bonds,
        bond_force_kwargs={
            'bondLength':1.0,
            'bondWiggleDistance':0.1, # Bond distance will fluctuate +- 0.05 on average
         },

        angle_force_func=None,
        angle_force_kwargs={},

        nonbonded_force_func=None,
        nonbonded_force_kwargs={},

        except_bonds=True,
        )
     )


    marks0 = central_domain(1000, size)

   # marks0 = 10000*[-1.0]
   # marks0 = 2000*[-1.0] + 1000*[1.0] + 4000*[-1.0] + 1000*[1.0] + 2000*[-1.0]
   # for i in range(len(marks0)):
   #     if random.random() < 0.5:
   #         marks0[i] = -1.0

    fsim = simple_marks_SSW(sim, marks0, selectiveAttractionEnergy=alpha)
    frel = simple_marks_SSW(sim, marks0, selectiveAttractionEnergy=alpha)
    sim.add_force(fsim)
    relaxer.add_force(frel)

    polymer = starting_conformations.grow_cubic(size, 200)
    sim.set_data(polymer, center=True)
    sim.local_energy_minimization()
    relax_polymer(sim, relaxer, 1)

    # polymerutils.save(sim.get_data(), 'dyntest_a'+str(alpha10)+'_i'+str(0)+'.dat')
    # relax_polymer(sim, relaxer, 1000)
    # polymerutils.save(sim.get_data(), 'dyntest_a'+str(alpha10)+'_i'+str(1)+'.dat')
    # for i in range(2, 201, 1):
    #     print(i)
    #     run_polymer_dynamics(sim, 10000)
    #     polymerutils.save(sim.get_data(), 'dyntest_a'+str(alpha10)+'_i'+str(i)+'.dat')

    # relax_polymer(sim, relaxer, 1000)
    # polymerutils.save(sim.get_data(), 'dyntest_a'+str(alpha10)+'_i'+str(201)+'.dat')

    #s10range = np.linspace(2, 30, 10)
    #ETrange = [50, 500]
    #s100range = [30, 34, 38, 42, 46, 50, 54, 58, 62]
    s100range = [62]
    #alpha10range = [0, 6, 12, 18, 24]
    alpha10range = [alpha10]
    #s10range = [20]
    for alpha10 in alpha10range:
        a = alpha10 / 10.0
        relaxer.context.setParameter("simple_selective_SSW_ATTReAdd", a * relaxer.kT)
        sim.context.setParameter("simple_selective_SSW_ATTReAdd", a * sim.kT)
        for s100 in s100range:
            s = 0.00346574 * s100 / 100.0
            print((a, s))
            csvfile = open('interdyn_dilution_a'+str(alpha10)+'_s'+str(s100)+'_e'+str(ET)+'_dyn'+str(dynspeed)+'_rt'+str(rtime)+'.csv', 'w', newline='')
            writer = csv.writer(csvfile)
            #writer.writerow(s100range)
            marks = copy.copy(marks0)
            writer.writerow(marks)
            polymer = starting_conformations.grow_cubic(size, 200)
            sim.set_data(polymer, center=True)
            sim.local_energy_minimization()
            relax_polymer(sim, relaxer, rtime)
            for i in range(gens):
                #writer.writerow(marks) 
                if ((i+1) % 10 == 0) or (i == 0):
                    csvfile.flush()
                if count_marks(marks) == 0:
                    pass
                else:
                    for j in range(200):
                        #print(j)
                        pts = sim.get_data()
                        #polymerutils.save(pts, 'dynmulti_a'+str(alpha10)+'_s'+str(s100)+'_e'+str(ET)+'_j'+str(j)+'.dat')
                        if j == 100: #replicative dilution
                            for i in range(len(marks)):
                                if random.random() < 0.5:
                                    marks[i] = -1.0
                        marks, degrees = update_marks_SIS(pts, marks, r, s, ET, p3D, 1)
                        send_marks_to_sim(sim, fsim, marks)
                        send_marks_to_sim(relaxer, frel, marks)
                        run_polymer_dynamics(sim, dynspeed)

                    writer.writerow(marks)
                    csvfile.flush()

                    initial_polymer_state(sim, size)
                    relax_polymer(sim, relaxer, rtime)
            
            csvfile.close()

    #         #CREATE AND EXPORT PLOT
    #         data = np.genfromtxt('diag_a'+str(alpha10)+'_s'+str(s100)+'_e'+str(ET)+'_rt'+str(rtime)+'.csv', delimiter=',', skip_header=1)
    #         data = np.rint(data)
    #         cmap = ListedColormap(['#fee090','blue'])
    #         image = cmap(data)
    #         golden = (1 + 5 ** 0.5) / 2
    #         plt.imshow(image, aspect=(10000/len(data))*(1/golden), interpolation='none')
    #         plt.axis('off')
    #         plt.savefig('diag_a'+str(alpha10)+'_s'+str(s100)+'_e'+str(ET)+'_rt'+str(rtime)+'.png', bbox_inches='tight')

    #         marks = copy.copy(marks0)
    #         send_marks_to_sim(sim, fsim, marks)
    #         send_marks_to_sim(relaxer, frel, marks)

    



main()
exit()
