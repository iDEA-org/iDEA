###################################################################################################################################################
####################################################### Unrestricted Hartree-Fock Theory ##########################################################
###################################################################################################################################################

############################################################## Import packages ####################################################################

import numpy as np # Python's numerical library
from numpy import diag
import scipy.integrate as spi
from scipy.linalg import eigh # To solve the ground-state Schroedinger equation (eigenproblem)
from scipy.linalg import expm # For time propagation
import matplotlib.pyplot as plt # For plotting
from scipy.sparse import spdiags

################################################################## Functions #####################################################################

def calculate_current_density(density_array, dt, dx):
    return -spi.cumtrapz((np.gradient(density_array, dt, axis=0, edge_order=2)), dx=dx, axis=1)

def Hamiltonian( spin, perturbing_potential, Kinetic_energy, Externa_potential, occ_channel_1, occ_channel_2, coulomb_potential, potential, psi_channel_1, psi_channel_2, spatial_points, dx, fock_on, time_dependent ): # This function constructs the Hamiltonian
    Hartree_potential = diag( ( np.dot( ( density(psi_channel_1, occ_channel_1) + density(psi_channel_2, occ_channel_2) ), coulomb_potential ) * dx ) ) # Hartree potential
    if spin == 'up':
        occ = occ_channel_1
        psi = psi_channel_1
    else:
        occ = occ_channel_2
        psi = psi_channel_2
    Fock_operator = fock( occ, psi, coulomb_potential )
    if time_dependent == False:
        return Kinetic_energy + Externa_potential + Hartree_potential + ( fock_on * Fock_operator * dx ) # Full Hamiltonian
    elif time_dependent == True:
        return Kinetic_energy + Externa_potential + Hartree_potential + ( fock_on * Fock_operator * dx ) + perturbing_potential # Add perturbing potential to the Hamiltonian

def density( psi, occ ): # Calculate the electron density
    den = np.zeros( psi.shape[0], dtype = "float" )
    for i in range( len(occ) ):
        den += occ[i] * abs( psi[:,i] )**2
    return den

def fock( occ, psi, coulomb_potential ): # Calculate the Fock operator
    F = np.zeros( ( psi.shape[0], psi.shape[0] ), dtype='complex')
    for i in range( np.count_nonzero(occ) ):
        F -= np.tensordot(psi[:,i], psi[:,i].conj(), axes = 0)
    F *= coulomb_potential
    return F

def coulomb( x ): # Construct the Coulomb interaction
    coulomb_interaction = np.zeros( (x.shape[0], x.shape[0]), dtype = "float" )
    for i in range(spatial_points):
        for j in range(spatial_points):
            coulomb_interaction[i,j] = 1.0 / ( abs( x[i] - x[j] ) + 1 )
    return coulomb_interaction

def solve( mixing, Kinetic_energy, Externa_potential, occ_channel_1, occ_channel_2, coulomb_potential, potential, spatial_points, dx, psi_channel_1, psi_channel_2, fock_on ): # Solve the ground-state Hartree-Fock equations self consistently
    convergence = 1
    psi_old_channel_1 = psi_start_channel_1
    psi_old_channel_2 = psi_start_channel_2
    H_old_channel_1 = Hamiltonian( 'up', perturbing_potential, Kinetic_energy, Externa_potential, occ_channel_1, occ_channel_2, coulomb_potential, potential, psi_old_channel_1, psi_old_channel_2, spatial_points, dx, fock_on, False )
    H_old_channel_2 = Hamiltonian( 'down', perturbing_potential, Kinetic_energy, Externa_potential, occ_channel_1, occ_channel_2, coulomb_potential, potential, psi_old_channel_1, psi_old_channel_2, spatial_points, dx, fock_on, False )
    H_old_channel_2 += spdiags( 1e-13*x, np.array([0]), spatial_points, spatial_points ).toarray() # Break the symmertry
    while convergence > 1e-13:
        H_channel_1 = ( 1 - mixing ) * H_old_channel_1 + mixing * Hamiltonian( 'up', perturbing_potential, Kinetic_energy, Externa_potential, occ_channel_1, occ_channel_2, coulomb_potential, potential, psi_old_channel_1, psi_old_channel_2, spatial_points, dx, fock_on, False ) # Construct the Hamiltonian for our system (mixed for stability)
        H_channel_2 = ( 1 - mixing ) * H_old_channel_2 + mixing * Hamiltonian( 'down', perturbing_potential, Kinetic_energy, Externa_potential, occ_channel_1, occ_channel_2, coulomb_potential, potential, psi_old_channel_1, psi_old_channel_2, spatial_points, dx, fock_on, False ) # Construct the Hamiltonian for our system (mixed for stability)
        eigenvalues_channel_1, eigenvectors_channel_1 = eigh( H_channel_1 ) # Solve the ground-state Hartree-Fock equations # Solve the ground-state Hartree-Fock equations
        eigenvalues_channel_2, eigenvectors_channel_2 = eigh( H_channel_2 ) # Solve the ground-state Hartree-Fock equations
        psi_channel_1 = eigenvectors_channel_1[:,:len(occ_channel_1)] # Ground-state wavefunction
        psi_new_channel_1 = psi_channel_1 / np.sqrt( np.trapz( abs( psi_channel_1 )**2, dx = dx, axis = 0 ) ) # Normalise the wavefunction
        psi_channel_2 = eigenvectors_channel_2[:,:len(occ_channel_2)] # Ground-state wavefunction
        psi_new_channel_2 = psi_channel_2 / np.sqrt( np.trapz( abs( psi_channel_2 )**2, dx = dx, axis = 0 ) ) # spatial_pointsormalise the wavefunction
        convergence = np.trapz( abs( density( psi_new_channel_1, channel_1 ) + density( psi_new_channel_2, channel_2 ) - density( psi_old_channel_1, channel_1 ) - density( psi_old_channel_2, channel_2 ) ), dx = dx ) # Compare the new density to the old
        H_old_channel_1 = Hamiltonian( 'up', perturbing_potential, Kinetic_energy, Externa_potential, occ_channel_1, occ_channel_2, coulomb_potential, potential, psi_new_channel_1, psi_new_channel_2, spatial_points, dx, fock_on, False )
        H_old_channel_2 = Hamiltonian( 'down', perturbing_potential, Kinetic_energy, Externa_potential, occ_channel_1, occ_channel_2, coulomb_potential, potential, psi_new_channel_1, psi_new_channel_2, spatial_points, dx, fock_on, False )
        psi_old_channel_1 = psi_new_channel_1
        psi_old_channel_2 = psi_new_channel_2
    return psi_new_channel_1, psi_new_channel_2, eigenvalues_channel_1, eigenvalues_channel_2

def timestep( spin, wavefunctions_channel_1, wavefunctions_channel_2, Kinetic_energy, Externa_potential, perturbing_potential, occ_channel_1, occ_channel_2, dx, dt, fock_on ):
    if spin == 'up':
        occ = occ_channel_1
        wavefunctions = wavefunctions_channel_1
    else:
        occ = occ_channel_2
        wavefunctions = wavefunctions_channel_2
    for i in range( len(occ) ):
        wavefunctions[:,i] = np.dot(expm(-1.0j*dt*Hamiltonian( spin, perturbing_potential, Kinetic_energy, Externa_potential, occ_channel_1, occ_channel_2, coulomb_potential, potential, wavefunctions_channel_1, wavefunctions_channel_2, spatial_points, dx, fock_on, True )), wavefunctions[:,i]) # Propagate the wavefunction through one time step
        #wavefunctions[:,i] /= np.sqrt( np.trapz( abs( wavefunctions[:,i] )**2, dx = dx, axis = 0 ) ) # spatial_pointsormalise the wavefunction
    return wavefunctions

def occupations( spins, restricted ):
    Number_electrons_up = spins.count( 'u' )
    Number_electrons_down = spins.count( 'd' )

    if Number_electrons_up > Number_electrons_down:
        occupations_1 = np.zeros( Number_electrons_up, dtype='int' )
        occupations_2 = np.zeros( Number_electrons_up, dtype='int' )
    else:
        occupations_1 = np.zeros( Number_electrons_down, dtype='int' )
        occupations_2 = np.zeros( Number_electrons_down, dtype='int' )

    for v in range( Number_electrons_up ):
        occupations_1[v] += 1
    if restricted == 'False':
        for v in range( Number_electrons_down ):
            occupations_2[v] += 1
    else:
        for v in range( Number_electrons_down ):
            occupations_1[v] += 1

    return list(occupations_1), list(occupations_2)

def total_energy( channel_1, channel_2, wavefunctions_channel_1, wavefunctions_channel_2, eigenenergies_channel_1, eigenenergies_channel_2, coulomb_potential, dx ):
    E = 0
    for i in range( len(channel_1) ):
        E += channel_1[i] * eigenenergies_channel_1[i]
    for i in range( len(channel_2) ):
        E += channel_2[i] * eigenenergies_channel_2[i]

    # Subtract Hartree energy
    den = density( wavefunctions_channel_1, channel_1 ) + density( wavefunctions_channel_2, channel_2 )
    Hartree_potential = np.dot( ( density(wavefunctions_channel_1, channel_1) + density(wavefunctions_channel_2, channel_2) ), coulomb_potential ) * dx
    E -= 0.5 * np.dot( Hartree_potential, den ) * dx

    # Fock correction
    Fock_1 = fock( channel_1, wavefunctions_channel_1, coulomb_potential )
    Fock_2 = fock( channel_2, wavefunctions_channel_2, coulomb_potential )
    for i in range( len(channel_1) ):
        E -= channel_1[i] * 0.5 * np.dot( (wavefunctions_channel_1[:,i].conj().T), np.dot( Fock_1[:], wavefunctions_channel_1[:,i] )) * dx**2
    for i in range( len(channel_2) ):
        E -= channel_2[i] * 0.5 * np.dot( (wavefunctions_channel_2[:,i].conj().T), np.dot( Fock_2[:], wavefunctions_channel_2[:,i] )) * dx**2

    return E.real

#################################################################################################################################################

L = 40.0 # Length of the real-space grid
dx = 0.2 # Grid spacing
spatial_points = int( L / dx ) # Number of grid points

spins = 'ud'
restricted = 'True'

channel_1, channel_2 = occupations( spins, restricted )

sorted_channel_1 = channel_1.copy()
sorted_channel_1.sort( reverse=True )
sorted_channel_2 = channel_2.copy()
sorted_channel_2.sort( reverse=True )
Number_electrons = sum( channel_1 ) + sum( channel_2 ) # Number of electrons
x = np.linspace( -0.5*dx*spatial_points, 0.5*dx*spatial_points, num = spatial_points ) # Spatial grid
potential = 0.5*( 0.25**2 )*x**2 #-1.0/(abs(x+5)+1) -1.0/(abs(x-5)+1) -3.0/(abs(x+10)+1)    # External potential

# Set these up once and don't need to do it again
Kinetic_energy = spdiags( np.array( [ (1.0/dx**2)*np.ones(spatial_points), (-0.5/dx**2)*np.ones(spatial_points), (-0.5/dx**2)*np.ones(spatial_points)] ), np.array( [0,-1,1] ), spatial_points, spatial_points ) # Kinetic energy
Externa_potential = diag( potential ) # Potential energy
coulomb_potential = coulomb(x) # Calculate once and not again
perturbing_potential = diag( -0.1*x )
psi_start_channel_1 = np.zeros( (spatial_points, len(channel_1) ) )
psi_start_channel_2 = np.zeros( (spatial_points, len(channel_2) ) )
non_interaction_wavefunction_channel_1, non_interaction_wavefunction_channel_2, non_interaction_eigenenergies_channel_1, non_interaction_eigenenergies_channel_2 = solve( 0.5, Kinetic_energy, Externa_potential, sorted_channel_1, sorted_channel_2, (0*coulomb_potential), potential, spatial_points, dx, psi_start_channel_1, psi_start_channel_2, 0 )
hartree_fock_wavefunction_channel_1, hartree_fock_wavefunction_channel_2, hartree_fock_eigenenergies_channel_1, hartree_fock_eigenenergies_channel_2 = solve( 0.5, Kinetic_energy, Externa_potential, sorted_channel_1, sorted_channel_2, coulomb_potential, potential, spatial_points, dx, psi_start_channel_1, psi_start_channel_2, 1 )
#hartree_wavefunction_channel_1, hartree_wavefunction_channel_2 = solve( 0.5, Kinetic_energy, Externa_potential, sorted_channel_1, sorted_channel_2, coulomb_potential, potential, spatial_points, dx, psi_start_channel_1, psi_start_channel_2, 0 )

energy_hartree_fock = total_energy( channel_1, channel_2, hartree_fock_wavefunction_channel_1, hartree_fock_wavefunction_channel_2, hartree_fock_eigenenergies_channel_1, hartree_fock_eigenenergies_channel_2, coulomb_potential, dx )
print( 'Hartree-Fock energy = %s' % energy_hartree_fock )
energy_non_interaction = total_energy( channel_1, channel_2, non_interaction_wavefunction_channel_1, non_interaction_wavefunction_channel_2, non_interaction_eigenenergies_channel_1, non_interaction_eigenenergies_channel_2, (0*coulomb_potential), dx )
print( 'Non-interacting energy = %s' % energy_non_interaction )

############################################################## Plot the result #################################################################

plt.plot( x, density( hartree_fock_wavefunction_channel_1, channel_1 ) + density( hartree_fock_wavefunction_channel_2, channel_2 ), label='Hartree-Fock' )
#plt.plot( x, density( hartree_wavefunction_channel_1, channel_1 ) + density( hartree_wavefunction_channel_2, channel_2 ), label='Hartree' )
plt.plot( x, density( non_interaction_wavefunction_channel_1, channel_1 ) + density( non_interaction_wavefunction_channel_2, channel_2 ), label = 'Non-interacting' )
#plt.plot( x, potential, label = 'Potential' )
plt.ylabel( r'$n(x)$' )
plt.xlabel( r'$x$' )
plt.tight_layout()
plt.legend()
plt.show()

############################################################# Time dependence #################################################################

number_timesteps = 800
simulation_time = 200.
dt = simulation_time / number_timesteps
density_array = np.zeros( (number_timesteps,spatial_points), dtype='float' )
time_dependent_wavefunctions_channel_1 = np.zeros( (number_timesteps, spatial_points, len(sorted_channel_1) ), dtype = "complex")
time_dependent_wavefunctions_channel_2 = np.zeros( (number_timesteps, spatial_points, len(sorted_channel_1) ), dtype = "complex")
time_dependent_wavefunctions_channel_1[0,...] = hartree_fock_wavefunction_channel_1 # Initialise the wavefunction
time_dependent_wavefunctions_channel_2[0,...] = hartree_fock_wavefunction_channel_2 # Initialise the wavefunction
for j in range( number_timesteps-1 ):
    print('Simulation time %.2f atomic units' % ( j*dt ), end='\r')
    time_dependent_wavefunctions_channel_1[j+1,...] = timestep( 'up', time_dependent_wavefunctions_channel_1[j,...], time_dependent_wavefunctions_channel_2[j,...], Kinetic_energy, Externa_potential, perturbing_potential, sorted_channel_1, sorted_channel_2, dx, dt, 1 )
    time_dependent_wavefunctions_channel_2[j+1,...] = timestep( 'down', time_dependent_wavefunctions_channel_1[j,...], time_dependent_wavefunctions_channel_2[j,...], Kinetic_energy, Externa_potential, perturbing_potential, sorted_channel_1, sorted_channel_2, dx, dt, 1 )
    density_array[j,:] = density( time_dependent_wavefunctions_channel_1[j,...], channel_1 ) + density( time_dependent_wavefunctions_channel_2[j,...], channel_2 )
    if ( j * 5 ) % number_timesteps < 5: # Plot sample timesteps throughout the whole simulation
        plt.plot( x, ( density( time_dependent_wavefunctions_channel_1[j,...], channel_1 ) + density( time_dependent_wavefunctions_channel_2[j,...], channel_2 ) ), label = r'$t = %.2f$' % ( j*dt ) )

plt.legend( prop = {"size":11}, frameon = True, ncol = 3 , loc = 'upper center', bbox_to_anchor = ( 0.5, 1.3 ), fancybox = True )
plt.ylabel( r'$n(x,t)$' )
plt.xlabel( r'$x$' )
plt.tight_layout()
plt.show()

current = calculate_current_density(density_array, dt, dx)

for j in range( number_timesteps-1 ):
    if ( j * 5 ) % number_timesteps < 5:
        plt.plot( ( current[j,:] ), label = r'$t = %.2f$' % ( j*dt ) )

plt.legend( prop = {"size":11}, frameon = True, ncol = 3 , loc = 'upper center', bbox_to_anchor = ( 0.5, 1.3 ), fancybox = True )
plt.ylabel( r'$j(x,t)$' )
plt.xlabel( r'$x$' )
plt.tight_layout()
plt.show()

if restricted == 'True':
    channel_2 = [0] * len(channel_1)
    for v in range( len(channel_1) ):
        if channel_1[v] > 1:
            channel_1[v] = 1
            channel_2[v] = 1

if spins.count( 'u' ) != sum(channel_1):
    channel_1, channel_2 = channel_2, channel_1

print()
print(channel_1)
print(channel_2)
