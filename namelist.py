
import math, os
import numpy as np
import scipy.constants as sc
from scipy.special import binom
import h5py
import scipy.constants as cst
import scipy
from scipy.interpolate import RegularGridInterpolator
import cmath
from math import cos, sin



######### Namelist for PALLAS target optimisation #####################
# works with SMILEI version 5.0 - BTIS II ENV
#######################################################################

######### function copied from LPAbrew/plasma.py to avoid #############

def eDensity(Zat,p):
    """
    plasma electron density pressure (fully ionized)
    ---
    pressure: p [mbar]
    atomic number: Z
    electron density: ne [cm^-3]
    """

    return 2.429e16*Zat*p

######################### Units scales ################################
m2cm = 1e2                       # m => cm
m2um = 1e6                       # m => um
cc2pm3  = 1e6 #cm^-3 to m^-3
#########################  Physical constants #########################

lambda_0            = 0.8e-6                    # laser wavelength, m
onel                = lambda_0/(2*math.pi)      # code length unit
c                   = sc.c                      # lightspeed, m/s
omega_0             = 2*math.pi*c/lambda_0     # laser angular frequency, rad/s
eps0                = sc.epsilon_0              # Vacuum permittivity, F/m
e                   = sc.e                      # Elementary charge, C
me                  = sc.m_e                    # Electron mass, kg
ncrit               = eps0*omega_0**2*me/e**2   # Plasma critical number density, m-3
c_over_omega0       = lambda_0/2./math.pi       # converts from c/omega0 units to m
reference_frequency = omega_0                   # reference frequency, s-1
E0                  = me*omega_0*c/e            # reference electric field, V/m

##### Variables used for unit conversions
c_normalized        = 1.                        # speed of light in vacuum in normalized units
um                  = 1.e-6/c_over_omega0       # 1 micron in normalized units
me_over_me          = 1.0                       # normalized electron mass
mp_over_me = sc.proton_mass / sc.electron_mass  # normalized proton mass
mn_over_me = sc.neutron_mass / sc.electron_mass # normalized neutron mass
fs                  = 1.e-15*omega_0            # 1 femtosecond in normalized units
mm_mrad             = um                        # 1 millimeter-milliradians in normalized units
pC                  = 1.e-12/e                  # 1 picoCoulomb in normalized units
#
#########################  Simulation parameters #########################

##### mesh resolution
dx                  = 0.02* um                   # longitudinal mesh resolution
dr                  = 0.7 * um                   # transverse mesh resolution
dt                  = 0.98 * dx/c_normalized      # integration timestep corespond to 0.1 fs

##### simulation window size
nx                  = 5120                       # number of mesh points in the longitudinal direction
nr                  = 256                        # 153 um en r
# number of mesh points in the transverse direction
Lx                  = nx * dx                   # longitudinal size of the simulation window
Lr                  = nr * dr                   # transverse size of the simulation window

##### Total simulation time
N_timesteps         =  4201*um
T_sim               =  N_timesteps                    #density profile 4mm at speed c this is 125 000[dt] in units of dt

##### patches parameters (parallelization)
npatch_x            = 512
npatch_r            = 32

#
ntrans=nr
dtrans=dr
Ltrans = ntrans*dtrans

Lsim=[dx*nx,nr*dr]

Nm=3          # number of AM modes
N_time = 6000 # needed laserFromlasy block 

#laser_fwhm                  = 700.*fs
#t_laser_peak_enters_window  = 1*laser_fwhm
#time_start_moving_window    = t_laser_peak_enters_window #(Lx-1.8*laser_fwhm)+t_laser_peak_enters_window

########################## Main simulation definition block #########################

Main(
    geometry = "AMcylindrical",

    interpolation_order = 2,
    use_BTIS3_interpolation = True,
    timestep = dt,
    number_of_timesteps = T_sim,

    cell_length  = [dx, dr],
    grid_length = [ Lx,  Lr],

   #number_of_AM = 1,
    number_of_AM = 3,

    number_of_patches = [npatch_x,npatch_r],
    EM_boundary_conditions = [
        ["silver-muller","silver-muller"],
        ["buneman","buneman"],
    ],
    #number_of_pml_cells =  [[6,6],[6,6]]

    solve_poisson = False,
    #solve_relativistic_poisson = True,
    print_every = 100,
    reference_angular_frequency_SI = omega_0,
    random_seed = smilei_mpi_rank
)
#
########################## Moving window ###########################################
MovingWindow(
    time_start = Lx, #time_start_moving_window,
    velocity_x = c_normalized # propagation speed of the moving window along the positive x direction, in c units
)
#
######################### Checkpoint for  the simulation #########################

# Path to your Lasy HDF5 output
lasy_file = "laserfileFromlasy.h5"

# Use the class directly (already inside pyprofiles.py)
laser_module = LaserFromLasy(lasy_file, dt, dtrans, Ltrans, ntrans,N_time, Nm)


# Add your laser
#Sim.addLaser(laser_module.get_smilei_laser())

# Run the simulation
#Sim.run()


############################# Diagnostics #############################

list_fields = ['Ex','Ey','Ez','By','Bz']

DiagProbe(
    every = 100,
    origin = [0., 10., 0.],
    corners = [[Lsim[0], 10., 0.]],
    number=[100],
    fields = list_fields,
)


