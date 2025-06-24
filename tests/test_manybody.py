'''
iDEA Many-Body Testing Platform 
Version: 1.0

Tests the Many-Body functionality of iDEA against analytical results to protect the iDEA code from any bugs/errors introduced during updates to iDEA.  
To run manually, use pytest test_manybody.py when in the 'iDEA/tests/many_body' directory.

The system consists of two electrons with spin = 'ud' or spin = 'uu' in a harmonic oscillator potential with a softened coulomb interaction i.e. Hooke's atom. 
The analytical solution determines the ground state wavefunction for both electrons using separation of variables. 
One component is solved analytically (Phi) and the other is solved numerically (Chi), both components are combined to give the full many-body wavefunction (Psi) from which observables can be calculated.

The same system is then solved using iDEA. Observables can then be compared to the results from the analytical solution. 

Tests are written with pytest to check for numerical accuracy by comparison of iDEA and the analytical test system. 
'''

# Dependencies
import iDEA
from dataclasses import dataclass
import numpy as np 
import scipy.linalg as spla
import pytest 

@dataclass
class Groundstate:
    r'''
    Contains all properties of the many body groundstate system.

    | Args:
    |     density : np.array, Probability density on a grid of x values.
    |     wavefunction : np.array, Wavefunction on the grid of x values.
    |     total_energy : float, Total energy of the system.

    | Methods:
    |       get_analytic_groundstate(params, spin) : Calculates the groundstate for the analytic test system.
    |       get_iDEA_groundstate(params, spin) : Calculates the groundstate for the iDEA test system.
    '''
    density: np.array 
    wavefunction: np.array
    total_energy: float 

    def get_analytic_groundstate(params, spin): 
        r'''
        Calculates the groundstate for the analytic test system.
        | Args:
        |     params : dictionary, Contains initial parameters: num_points, omega, length. 
        |     spin : string, Determines spin of the two electrons. 

        | Returns: 
        |     Groundstate : class, Object containing the groundstate properties of the system.
        '''
        num_points = params['num_points']
        omega = params['omega']
        length = params['length']
        
        def get_kinetic_energy_operator(u,du):
            r'''
            Generates the kinetic energy operator to be used in the Hamiltonian.

            | Args:
            |     u : np.array, Array containing grid positions.  
            |     du : float, Spacing between grid points

            | Returns:
            |     kinetic_energy : np.array, Kinetic energy operator. 
            '''
            sd = (
                1.0
                / 831600.0
                * np.array(
                    [
                        -50,
                        864,
                        -7425,
                        44000,
                        -222750,
                        1425600,
                        -2480478,
                        1425600,
                        -222750,
                        44000,
                        -7425,
                        864,
                        -50,
                    ],
                    dtype=float,
                )
                / du**2
                )
            sdi = (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6)

            second_derivative = np.zeros((u.shape[0], u.shape[0]))
            for i in range(len(sdi)):
                second_derivative += np.diag(
                    np.full(
                        np.diag(np.zeros((u.shape[0], u.shape[0])), k=sdi[i]).shape[0],
                        sd[i],
                    ),
                    k=sdi[i],
                )
            kinetic_energy = -0.5 * second_derivative *2
            return kinetic_energy 
        
        def get_potentials(u, params):
            r'''
            Generates potential matrix for the Hamiltonian.

            | Args:
            |     u : np.array, Array containing grid positions. 
            |     params : dictionary, Contains initial parameters: num_points, omega, length. 

            | Returns: 
            |     V : np.array, Potential matrix containing the sum of external and interaction potentials. 
            '''

            omega = params['omega']

            v_ext = 0.25 * omega**2 * u**2
            v_int = 1/(abs(u) + 1)

            V = v_ext + v_int 
            return V 
        
        def build_hamiltonian(KE, V):
            r'''
            Generates the Hamiltonian operator for the system. 

            | Args:
            |     KE : np.array, Kinetic energy matrix. 
            |     V : np.array, Potential energy matrix.

            | Returns:
            |     H : np.array, Hamiltonian matrix.
            '''
            H = KE + np.diag(V)
            return H
        
        def get_wavefunction(Chi, Phi, params):
            r'''
            Generates the full wavefunction for the system via a tensor product. 
            Coordinate transforms the wavefunction back to realspace (x-grid).

            | Args:
            |     Chi : np.array, Numerical component of wavefunction. 
            |     Phi : np.array, Analytical component of wavefunction. 
            |     params : dictionary, Input parameters for test system, only num_points is used.

            | Returns:
            |     wavefunction : np.array, Full wavefunction in realspace (x-grid) coordinates. 
            '''

            num_points = params['num_points']

            # Combine wavefunction components via tensor product
            wavefunction_vu = np.tensordot(Chi, Phi, 0)

            # CGenerate indices for coordinate transformation using numpy vector operations
            i, j = np.meshgrid(np.arange(num_points), np.arange(num_points), indexing='ij')
            k = j - i + num_points - 1
            l = i + j

            # Coordinate transform back to x-grid
            wavefunction = wavefunction_vu[k,l]

            return wavefunction

        def antisymmetrize(wavefunction):
            r'''
            Antisymmetrizes an input wavefunction. 

            | Args:
            |     wavefunction : np.array, Wavefunction to be antisymmetrized.  

            | Returns:
            |     anti_wavefunction : np.array, Antisymmetrized wavefunction. 
            '''
            grid_size = wavefunction.shape[0]

            # Antisymmetrize
            anti_wavefunction = wavefunction - wavefunction.T 

            # Normalise
            norm = np.sqrt(np.sum(abs(anti_wavefunction**2)) * dx**2)
            anti_wavefunction = anti_wavefunction / norm 

            return anti_wavefunction

        def get_density(wavefunction, dx):
            r'''
            Calculates the probability density of the system.

            | Args:
            |     wavefunction : np.array, Input wavefunction.
            |     dx : float, Spacing between grid points. 

            | Returns:
            |     density : np.array, Probability density of the system. 
            '''
            
            density = 2 * dx * np.sum(wavefunction**2, axis=1)
            return density 

        # Set boundaries of realspace 1D grid
        x_max = length/2 
        x_min = -length/2

        # Boundaries of coordinate conversion space
        u_max = 2 * x_max 
        u_min = 2 * x_min
        v_max = x_max 
        v_min = x_min 
        alt_num_points = 2*(num_points-1)+1 
        
        u = np.linspace(u_min, u_max, alt_num_points)
        v = np.linspace(v_min, v_max, alt_num_points)

        # Calculate derivatives
        dx = (x_max - x_min)/(num_points - 1)
        du = dx 

        # Generate the kinetic energy operator matrix
        kinetic_energy = get_kinetic_energy_operator(u,du)

        # Generate the potential matrix
        V = get_potentials(u, params)

        # Assemble the hamiltonian
        Hamiltonian = build_hamiltonian(kinetic_energy, V)

        # Solve the eigenproblem
        energies, orbital = spla.eigh(Hamiltonian)
        orbital = orbital/np.sqrt(du)

        # Analytical wavefunction
        Phi = np.exp((-1 * omega) * (v**2))

        # Calculate for uu spin
        if spin == 'uu': 
            # Numerical wavefunction
            Chi = orbital[:,1]

            # Combine the equations via tensor product and transform back to x-grid
            wavefunction = get_wavefunction(Chi, Phi, params)
        
            # antisymmetrize
            wavefunction = antisymmetrize(wavefunction)

            # Get total energy 
            Ev = omega * 0.5
            total_energy = energies[1] + Ev 
        
        # Calculate for ud spin
        elif spin =='ud':
            
            Chi = orbital[:,0]

            wavefunction = get_wavefunction(Chi, Phi, params)
            norm = np.sqrt(np.sum(abs(wavefunction**2)) * dx**2)
            wavefunction = wavefunction / norm 

            Ev = omega * 0.5
            total_energy = energies[0] + Ev 

        # Calculate the density
        density = get_density(wavefunction, dx)

        return Groundstate(density, wavefunction, total_energy)
    
    def get_iDEA_groundstate(params, spin):
        r'''
        Calculates the groundstate for the iDEA test system.

        | Args:
        |     params : dictionary, Contains initial parameters: num_points, omega, length. 
        |     spin : string, Determines spin of the two electrons. 

        | Returns: 
        |     Groundstate : class, Object containing the groundstate properties of the system.
        '''

        num_points = params['num_points']
        omega = params['omega']
        length = params['length']

        # Set grid
        x_max = 0.5 * length 
        x_min = -x_max 
        x = np.linspace(x_min, x_max, num_points)

        # Define external potential 
        v_ext = 0.5 * omega**2 * x**2 

        # Define interaction potential 
        v_int = iDEA.interactions.softened_interaction(x)

        # Build system
        system = iDEA.system.System(x, v_ext, v_int, electrons = spin)

        # Solve for ground state 
        ground_state = iDEA.methods.interacting.solve(system, k=0)

        # Get observables 
        density = iDEA.observables.density(system, state = ground_state)

        total_energy = ground_state.energy
        wavefunction = ground_state.space 

        return Groundstate(density, wavefunction, total_energy)

################################################################## Test Inputs ################################################################## 
# Parameters for short test
short_params = {'num_points' : 231,
                'omega' : 0.3275,
                'length' : 20
            }

# Parameters for long test
long_params = {'num_points' : 921,
                'omega' : 0.3275,
                'length' : 40
            }

# Fixtures for short test
@pytest.fixture(scope='class')
def analytic_short(spin):
    return Groundstate.get_analytic_groundstate(short_params, spin)

@pytest.fixture(scope='class')
def iDEA_short(spin):
    return Groundstate.get_iDEA_groundstate(short_params, spin)

# Fixtures for long test
@pytest.fixture(scope='class')
def analytic_long(spin):
    return Groundstate.get_analytic_groundstate(long_params, spin)

@pytest.fixture(scope='class')
def iDEA_long(spin):
    return Groundstate.get_iDEA_groundstate(long_params, spin)

################################################################## Testing ##################################################################

@pytest.mark.parametrize('spin', [('ud'), ('uu')], scope='class')
class TestShort: 
    r'''
        Contains short runtime test functions, invoked using Pytest.
        Approximate runtime : ~ 16.8 seconds for 'uu' and 'ud' systems combined.
    '''
    
    def test_density(self, analytic_short, iDEA_short):
        r'''
        Tests that the probability density of the iDEA test system is within a specifed tolerance of the analytic solution. 
        '''
        tolerance = 1.2e-11

        iDEA_density = iDEA_short.density 
        analytical_density = analytic_short.density

        assert iDEA_density == pytest.approx(analytical_density, abs=tolerance)

    def test_total_energy(self, analytic_short, iDEA_short):
        r'''
        Tests that the total energy of the iDEA test system is within a specifed tolerance of the analytic solution. 
        '''
        tolerance = 4.0e-12

        iDEA_energy = iDEA_short.total_energy
        analytic_energy = analytic_short.total_energy

        assert iDEA_energy == pytest.approx(analytic_energy, abs=tolerance)

    def test_wavefunction(self, analytic_short, iDEA_short, spin):
        r'''
        Tests that the wavefunction of the iDEA test system is within a specifed tolerance of the analytic solution. 
        '''
        # Determine tolerance based on spin configuration
        if spin == 'ud':
            tolerance = 3.8e-7
        elif spin == 'uu':
            tolerance = 9.2e-7

        length = short_params['length']
        num_points = short_params['num_points']
        x_max = 0.5 * length 
        x_min = - x_max 

        dx = (x_max - x_min)/(num_points -1 )

        # Generate metric for wavefunction comparison
        Psi_1_modsq = abs(analytic_short.wavefunction)**2
        Psi_2_modsq = abs(iDEA_short.wavefunction)**2
        Psi_1_star = np.conjugate(analytic_short.wavefunction)
        Psi_2 = iDEA_short.wavefunction 

        metric = np.sqrt(np.sum(Psi_1_modsq + Psi_2_modsq)*dx - 2 * abs(np.sum(Psi_1_star * Psi_2)*dx))

        assert metric <= tolerance 


@pytest.mark.parametrize('spin', [('ud'), ('uu')], scope='class')
class TestLong:
    r'''
        Contains long runtime test functions, invoked using Pytest. 
        Approximate runtime : ~ 18 minutes for 'uu' and 'ud' systems combined.
    '''

    def test_density(self, analytic_long, iDEA_long):
        r'''
        Tests that the probability density of the iDEA test system is within a specifed tolerance of the analytic solution. 
        '''
        tolerance = 2.0e-13

        iDEA_density = iDEA_long.density 
        analytical_density = analytic_long.density

        assert iDEA_density == pytest.approx(analytical_density, abs=tolerance)

    def test_total_energy(self, analytic_long, iDEA_long):
        r'''
        Tests that the total energy of the iDEA test system is within a specifed tolerance of the analytic solution. 
        '''
        tolerance = 2.0e-12

        iDEA_energy = iDEA_long.total_energy
        analytic_energy =analytic_long.total_energy

        assert iDEA_energy == pytest.approx(analytic_energy, abs=tolerance)

    def test_wavefunction(self, analytic_long, iDEA_long, spin):
        r'''
        Tests that the wavefunction of the iDEA test system is within a specifed tolerance of the analytic solution. 
        '''
        
        if spin == 'ud':
            tolerance = 1.0e-12
        elif spin == 'uu':
            tolerance = 1.0e-12

        length = long_params['length']
        num_points = long_params['num_points']
        x_max = 0.5 * length 
        x_min = - x_max 

        dx = (x_max - x_min)/(num_points -1 )

        Psi_1_modsq = abs(analytic_long.wavefunction)**2
        Psi_2_modsq = abs(iDEA_long.wavefunction)**2
        Psi_1_star = np.conjugate(analytic_long.wavefunction)
        Psi_2 = iDEA_long.wavefunction 
        metric = np.sqrt(np.sum(Psi_1_modsq + Psi_2_modsq)*dx - 2 * abs(np.sum(Psi_1_star * Psi_2)*dx))

        assert metric <= tolerance 
