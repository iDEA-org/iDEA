'''
iDEA single particle time evolution Testing Platform 
Version: 1.0

Tests the time dependant functionality of iDEA against analytical results to protect the iDEA code from any bugs/errors introduced during updates to iDEA.  
To run manually, use pytest test_time_evolution.py when in the 'iDEA/tests/time_evolution' directory.

The system consists of one electrons, with either 'u' or 'd' spin in a quantum harmonic oscillator (QHO) potential.
The ability to choose spin has been included for although the spin of the ellectron has no effect on the result
The analytical solution determines the ground state wavefunction for all times from which observables can be calculated
The same system is then solved using iDEA. Observables are then be compared to the results from the analytical solution.

Tests are written with pytest to check for numerical accuracy and correct behaviour by comparison of iDEA and the analytical test system.
The test system is a Quantum harmonic oscillator in a purtibated field potenial. The system and analytical solution is described in: 
"Analytic solution to the time-dependent Schrodinger equation for the one-dimensional quantum harmonic oscillator with an applied uniform field" MJP Hodgson 2021/22
'''

# Dependencies
import math # Used for mathematical operations
import numpy as np # Python's numerical library
from scipy.sparse import spdiags # To construct the Hamiltonian as a matrix
from scipy.sparse.linalg import eigsh # To solve the ground-state Schroedinger equation
#from scipy.integrate import simps
import iDEA
import pytest
from dataclasses import dataclass # Importing the dataclass decorator

@dataclass
class Groundstate():
    r'''
    Contains all properties of the many body groundstate system.

    | Args:
    |     density : np.array, Probability density on a grid of x values.
    |     wavefunction : np.array, Wavefunction on the grid of x values.
    |     total_energy : float, Total energy of the system.
5e-12
    | Methods:
    |       get_analytic_groundstate(params, spin) : Calculates the groundstate for the analytic test system.
    |       get_iDEA_groundstate(params, spin) : Calculates the groundstate for the iDEA test system.
    '''
    density: np.array 
    wavefunction: np.array
    total_energy: float 

    def get_analytic_groundstate(params, spin):
        r'''
        Calculates the groundstate for the iDEA test system.

        | Args:
        |     params : dictionary, Contains initial parameters: num_pointsi_in_x, omega, x_length, t_length, num_points_in_time,
        |              perturbating_field_strengh, Analytic_terms.
        |              spin : string, Determines spin of the electron. 

        | Returns: 
        |     Groundstate : class, Object containing the groundstate properties of the system.
        '''
        
        N = params['num_points_in_x']
        w = params['omega']
        L = params['x_length']
        T = params['t_length']
        timesteps = params['num_points_in_time']
        e = params['perturbating_field_strengh']
        analytic_terms = params['Analytic_terms']
        K = params['State']
         
        # The representation of the potential on the spatial grid    
        def potential(N, dx, system, e, w): 
            return (0.5 * (w**2) * (np.linspace(-0.5 * dx * N, 0.5 * dx * N, num=N))**2 + e * np.linspace(-0.5 * dx * N, 0.5 * dx * N, num=N) + e**2 / (2 * w**2))

        # Function that yields the Hermite polynomials which are used in the analytic solution
        def hermite(s, w, N, dx): 
            if s == 0:
                return 1
            elif s == 1:
                return 2 * np.sqrt(w) * np.linspace(-0.5 * dx * N, 0.5 * dx * N, num=N)
            else:
                return 2 * np.sqrt(w) * np.linspace(-0.5 * dx * N, 0.5 * dx * N, num=N) * hermite(s-1, w, N, dx) - 2 * (s-1) * hermite(s-2, w, N, dx)

        # Function that creates the analytical solution for the QHO for k = 0,1,2 states 
        def analytic_solution(t, e, w, N, hermite_array, factorial_array, analytic_terms, dx, K): # Analytic solution for the SHO
            a = - e / (2 * np.sqrt(w)**3)
            s = np.linspace(0, analytic_terms, (analytic_terms + 1)).astype(int)
            if K == 0:
                return ((w / np.pi)**0.25) * np.exp((-0.5 * w * (np.linspace(-0.5 * dx * N, 0.5 * dx * N, num=N))**2 + 1.j * t) - a**2) * np.sum((1.0 / factorial_array[s]) * (a**s) * hermite_array[s] * np.exp(-1.0j * s * w * t))
            if K == 1:
                return ((1 / np.sqrt(2**K * math.factorial(K))) * (w / np.pi)**0.25) * np.exp(-0.5 * w * ((np.linspace(-0.5 * dx * N, 0.5 * dx * N, num=N))**2 + 1.j * t) - a**2) * np.sum((1.0 / factorial_array[s]) * (s - 2 * a**2) * (a**(s - K)) * hermite_array[s] * np.exp(-1.0j * s * w * t))
            if K == 2:
                return (((1 / np.sqrt(2**K * math.factorial(K))) * (w / np.pi)**0.25) * np.exp(-0.5 * w * ((np.linspace(-0.5 * dx * N, 0.5 * dx * N, num=N))**2 + 1.j * t) - a**2) * np.sum((1.0 / factorial_array[s]) * (s**2 - s - 4 * s * a**2 + 4 * a**4) * (a**(s - K)) * hermite_array[s] * np.exp(-1.0j * s * w * t)))

        # Set boundaries of x space grid
        x_max = L/2 
        x_min = -L/2
        dx = L / N
        
        # Create x space grid
        x = np.linspace(x_min, x_max, N)
        
        # Set boundaries of time space grid
        t_max = T/2 
        t_min = -T/2
        dt = T / timesteps

        # Create time grid
        t = np.linspace(0, T, timesteps)

        # Create arrays to store the wavefunction, psi, and density, n.
        psi_analytic_array = np.zeros((timesteps, N), dtype=complex) # creating array to store all plots
        n_analytic_array = np.zeros((timesteps, N)) # creating array to store all plots
        # Calculate factorial and Hermite array's once
        factorial_array = np.array([math.factorial(s) for s in range(analytic_terms + 1)], dtype=object)
        hermite_array = np.array([hermite(s, w, N, dx) for s in range(analytic_terms + 1)], dtype=object)
        
        # Solving Analytically
        for j in range(timesteps):
            # Find time from time grid
            time_point = t[j]
            
            # Calculate analytic wavefunction
            psi_analytic = analytic_solution(time_point, e, w, N, hermite_array, factorial_array, analytic_terms, dx, K)
            
            # Calculate the density
            n = np.abs(psi_analytic)**2
            
            # Save wavefunction and density to arrays
            psi_analytic_array[j, :] = psi_analytic
            n_analytic_array[j, :] = n
            
            # Setting energy to zero - included to allow future developments
            total_energy_analytic = 0
            
            # Renaming data names for readability 
            density = n_analytic_array
            wavefunction = psi_analytic_array
            total_energy = total_energy_analytic

        return Groundstate(density, wavefunction, total_energy)
    
    def get_iDEA_groundstate(params, spin):
        r'''
        Calculates the groundstate for the iDEA test system.

        | Args:
        |     params : dictionary, Contains initial parameters: num_pointsi_in_x, omega, x_length, t_length, num_points_in_time,
                       perturbating_field_strengh, Analytic_terms.

        | Returns: 
        |     Groundstate : class, Object containing the groundstate properties of the system.
        '''
        # Naming variables within function
        N = params['num_points_in_x']
        w = params['omega']
        L = params['x_length']
        T = params['t_length']
        timesteps = params['num_points_in_time']
        e = params['perturbating_field_strengh']
        analytic_terms = params['Analytic_terms']
        K = params['State']
        
        # Set boundaries of x space grid
        x_max = L/2 
        x_min = -L/2
        
        # Create x space grid
        x = np.linspace(x_min, x_max, N)
        
        # Set boundaries of time space grid
        t_max = T/2 
        t_min = -T/2
        
        # Create time grid
        t = np.linspace(0, T, timesteps)

        # Potentials used in iDEA
        v_ext = 0.5 * w**2 * x**2 + e * x
        v_int = iDEA.interactions.softened_interaction(x)
        v_ptrb = np.zeros((t.shape[0], x.shape[0]))
        for j, ti in enumerate(t):
            v_ptrb[j, :] = -e * x
        
        # Defining the iDEA system
        s_iDEA = iDEA.system.System(x, v_ext, v_int, electrons=spin)
        
        ### Solving using iDEA ###
        
        # Solve for wavefunction for ground state
        ground_state = iDEA.methods.interacting.solve(s_iDEA, k=K)
        
        # Advance the system in time - this is the wavefunction at each time
        evolution = iDEA.methods.interacting.propagate(s_iDEA, ground_state, v_ptrb, t)
        
        # Calculate the electron density as time evolves 
        n = iDEA.observables.density(s_iDEA, evolution=evolution)
        
        # Setting energy to zero - included to allow future developments
        total_energy_idea = 0
        
        # Renaming data names for readability
        density = n
        wavefunction = evolution.td_space
        total_energy = total_energy_idea

        return Groundstate(density, wavefunction, total_energy) #density, wavefunction, total_energy

################################################################## Test Inputs ################################################################## 
# Parameters for short test
short_params = {'num_points_in_x' : 400,
                'omega' : 0.25,
                'x_length' : 40,
                't_length': 50,
                'num_points_in_time' : 500,
                'perturbating_field_strengh' : 0.1,
                'Analytic_terms' : 20,
                'State' : 2
            }

# Fixtures for short test
@pytest.fixture(scope='class')
def analytic_short(spin):
    return Groundstate.get_analytic_groundstate(short_params, spin)

@pytest.fixture(scope='class')
def iDEA_short(spin):
    return Groundstate.get_iDEA_groundstate(short_params, spin)

################################################################## Testing ##################################################################

@pytest.mark.parametrize('spin', ['u'], scope='class')
class TestShort: 
    r'''
        Contains short runtime test functions, invoked using Pytest. 
        Approximate runtime : 11 s
    '''
    
    def test_density(self, analytic_short, iDEA_short):
        r'''
        Tests that the probability density of the iDEA test system is within a specifed tolerance of the analytic solution. 
        '''
        tolerance = 5e-13 ##################################################################

        iDEA_density = iDEA_short.density 
        analytical_density = analytic_short.density

        assert iDEA_density == pytest.approx(analytical_density, abs=tolerance)

    def test_wavefunction(self, analytic_short, iDEA_short):
        r'''
        Tests that the wavefunctions of the iDEA system are within a defined tolerance of eachother.
        Phase independent test
        '''
        tolerance = 1e-7 ##################################################################

        # Setting lengh of system
        length_in_x = short_params['x_length']
        num_points_in_x = short_params['num_points_in_x']
        # Setting Max and Min in x
        x_max = 0.5 * length_in_x 
        x_min = -x_max
        dx = (x_max - x_min) / (num_points_in_x - 1)

        # Ensure that both wavefunctions are on the same grid
        x = np.linspace(x_min, x_max, num_points_in_x)
        analytic_wavefunction = np.interp(x, np.linspace(x_min, x_max, len(analytic_short.wavefunction[0])), analytic_short.wavefunction[0])
        iDEA_wavefunction = iDEA_short.wavefunction[0]

        # Normalizing wavefunctions
        norm_analytic = np.sqrt(np.sum(np.abs(analytic_wavefunction)**2) * dx)
        norm_idea = np.sqrt(np.sum(np.abs(iDEA_wavefunction)**2) * dx)
        analytic_wavefunction /= norm_analytic
        iDEA_wavefunction /= norm_idea

        # Compute phase-invariant metric
        #overlap = np.sum(np.conjugate(analytic_wavefunction) * iDEA_wavefunction * dx)
        #metric = 1 - np.abs(overlap)

        Psi_1_modsq = abs(analytic_wavefunction)**2
        Psi_2_modsq = abs(iDEA_wavefunction)**2
        Psi_1_star = np.conjugate(analytic_wavefunction)
        Psi_2 = iDEA_wavefunction
        # Calculate metric
        metric = np.sqrt(-np.sum(Psi_1_modsq + Psi_2_modsq)*dx + 2 * abs(np.sum(Psi_1_star * Psi_2)*dx))

        # Print value of metric if test fails
        print(f"Metric: {metric}")
        assert metric <= tolerance