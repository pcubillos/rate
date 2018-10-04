# Copyright (c) 2018 Patricio Cubillos and contributors.
# rate is open-source software under the MIT license (see LICENSE).

__all__ = ["gRT", "newton_raphson", "bound_nr", "top", "Rate"]

import os
import numpy as np
import scipy.constants   as sc
import scipy.interpolate as si


rootdir = os.path.realpath(os.path.dirname(__file__) + "/../")


class gRT():
  """
  Object to compute the Gibbs free energies from JANAF data.
  Available species are: H2, H2O, CO, CO2, CH4, C2H2, C2H4, HCN, NH3, and N2.
  """
  def __init__(self):
    self.heat = {}
    self.free_energy = {}
    path = rootdir + "/inputs/"
    for filename in os.listdir(path):
      if filename.endswith(".txt"):
        molname = filename.split("_")[0]
        T, G, H = np.loadtxt(path+filename, unpack=True)
        self.heat[molname] = 1000 * H[T==298.15][0] / sc.R
        self.free_energy[molname] = si.UnivariateSpline(T, G/sc.R, s=1)

  def __call__(self, spec, temp):
    return self.eval(spec, temp)

  def eval(self, spec, temp):
    """
    Evaluate the Gibbs for a given species at specified temperature.

    Parameters
    ----------
    spec: String
       Species. Select from: H2, H2O, CO, CO2, CH4, C2H2, C2H4, HCN,
       NH3, and N2.
    temp: Float scalar or 1D ndarray
       Temperature in Kelvin degree.

    Returns
    -------
    g_RT: Float scalar or 1D ndarray
       The Gibbs free energy in J/mol.
    """
    g_RT = -self.free_energy[spec](temp) + self.heat[spec]/temp
    return g_RT


def newton_raphson(A, guess, xtol=1e-8, imax=100, verb=False, degree=None):
  """
  Newton-Raphson algorithm to find polynomial roots, from Section
  9.5.6 of Numerical Recipes.

  Parameters
  ----------
  A: 1D float ndarray
     Polynomial coefficients sorted from lowest to highest degree.
  guess: Float
     Root's guess value.
  xtol: Float
     Accept solution when the fractional improvement of each
     iteration is less than xtol.
  imax: Integer
     Maximum number of iterations.
  verb: Bool
     Verbosity.
  degree: Integer
     The degree of the polynomial.

  Returns
  -------
  xnew: Float
     A real polynomial root of A.
  """
  if degree is None:
    degree = len(A) - 1
  xnew = guess
  x = -1.0
  k = 0
  # Newton-Raphson root finder:
  while np.abs(1.0-x/xnew) > xtol  and k < imax:
    x = xnew
    p  = A[degree]*x + A[degree-1]
    p1 = A[degree]
    i = degree-2
    while i >= 0:
      p1 = p    + p1*x
      p  = A[i] + p*x
      i -= 1
    xnew -= p/p1
    k += 1
  if k == imax and verb:
    print("Max iteration reached.")
  return xnew


def bound_nr(A, guess, vmax=np.inf, xtol=1e-8, imax=100, verb=False,
             degree=None):
  """
  Iterative Newton-Raphson root finder in bounded range (0,vmax).

  Parameters
  ----------
  A: 1D float ndarray
     Polynomial coefficients sorted from lowest to highest degree.
  guess: Float
     Root's guess value.
  vmax: Float
     Upper acceptable boundary for the polynomial root.
  xtol: Float
     Accept solution when the fractional improvement of each
     iteration is less than xtol.
  imax: Integer
     Maximum number of iterations.
  verb: Bool
     Verbosity.
  degree: Integer
     The degree of the polynomial.

  Returns
  -------
  root: Float
     A real polynomial root.
  """
  if degree is None:
    degree = len(A) - 1
  root = -2
  k1, kmax = 0, 10
  while (root < 0 or root > vmax)  and  k1 < kmax:
    root = newton_raphson(A, guess*10**-k1, xtol, imax, verb, degree)
    k1 += 1
  if (root < 0 or root > vmax) and verb:
    print("NR could not find a root bounded within the range "
          "[0, {:.5g}].".format(vmax))
  return root


def top(T, C, N, O):
  """
  Turn-over pressure (bar) where CO- and H2O-dominated chemistry
  flip, for a given temperature and elemental abundances.
  (CO dominates at p < TOP, H2O dominates at p > TOP).

  Parameters
  ----------
  T: Float scalar or ndarray
     Temperature in Kelvin degree.
  C: Float scalar or ndarray
     Carbon elemental abundance.
  N: Float scalar or ndarray
     Nitrogen elemental abundance.
  O: Float scalar or ndarray
     Oxygen elemental abundance.

  Returns
  -------
  TOP: Float scalar or ndarray
     Turn-over pressure in bar.

  Notes
  -----
  Valid domain is:
    200 < T < 6000 (optimized for 200 < T < 3000),
    3e-7 < C < 0.1,
    7e-8 < N < 0.1,
    5e-7 < O < 0.1, and
    C + N + O < 0.1
  Valid image is:
    1e-8 < TOP < 1e3.
  """
  # Polynomial coefficients describing TOP:
  p = [-1.10959270e+03,
        1.25882718e+03, -5.46310342e+02,  1.07687419e+02, -8.10199497e+00,
        1.30521375e+00,  3.16886254e-01,  5.28887371e-02,  3.09279090e-03,
        2.04846075e-02,  8.72496079e-03,  1.27754826e-03,  5.87928942e-05,
       -1.50411039e-01, -4.45199085e-02, -6.05990924e-03, -3.00488516e-04]
  # The polynomial actualy depends on the log10 of the variables.
  logT = np.log10(T)
  logC = np.log10(C)
  logN = np.log10(N)
  logO = np.log10(O)
  # This is log10(top):
  TOP = (p[ 0]
       + p[ 1]*logT + p[ 2]*logT**2 + p[ 3]*logT**3 + p[ 4]*logT**4
       + p[ 5]*logC + p[ 6]*logC**2 + p[ 7]*logC**3 + p[ 8]*logC**4
       + p[ 9]*logN + p[10]*logN**2 + p[11]*logN**3 + p[12]*logN**4
       + p[13]*logO + p[14]*logO**2 + p[15]*logO**3 + p[16]*logO**4)
  return 10**np.clip(TOP, -8.0001, 3.0001)


class Rate():
  """
  Reliable Analytic Thermochemical Equilibrium.
  Cubillos, Blecic, & Dobbs-Dixon (2018), ApJ, XX, YY.

  References
  ----------
    CBD2018:  Cubillos, Blecic, & Dobbs-Dixon (2018), ApJ
    HT2016: Heng & Tsai (2016), ApJ, 829, 104
    HL2016: Heng & Lyons (2016), ApJ, 817, 149
  """
  def __init__(self, C=2.5e-4, N=1.0e-4, O=5.0e-4):
    """
    Class initializer.

    Parameters
    ----------
    C: Float
       Carbon elemental abundance (relative to hydrogen).
    N: Float
       Nitrogen elemental abundance (relative to hydrogen).
    O: Float
       Oxygen elemental abundance (relative to hydrogen).
    """
    # Initialize elemental abundances:
    self.C = C
    self.N = N
    self.O = O

    # Initialize deltaG interpolators:
    self.grt = gRT()
    # Number of species:
    self.nmol = 11


  def kprime0(self, temp, press):
    """
    Compute the zeroth equilibrium constant K0 (Eq. (X) of CBD2018) for
    the reaction: H2 <-> 2*H,
    with k0 = n_H**2.

    Parameters
    ----------
    temp: Float scalar or 1D ndarray
       Temperature in Kelvin degrees.
    press: Float scalar or 1D ndarray
       Pressure in bars.

    Returns
    -------
    k0: Scalar or 1D float ndarray
       Zeroth normalized equilibrium constant (same shape as inputs).
    """
    k0 = np.exp(-( 2*self.grt("H",temp) - self.grt("H2",temp) )) / press
    return k0


  def kprime(self, temp, press):
    """
    Compute the first equilibrium constant K' (Eq. (27) of HL2016) for
    the reaction: CH4 + H2O <-> CO + 3*H2,
    with kp = n_CO / (n_CH4 * n_H2O).

    Parameters
    ----------
    temp: Float scalar or 1D ndarray
       Temperature in Kelvin degrees.
    press: Float scalar or 1D ndarray
       Pressure in bars.

    Returns
    -------
    kp: Scalar or 1D float ndarray
       First normalized equilibrium constant (same shape as inputs).
    """
    kp = np.exp(-( self.grt("CO", temp) + 3*self.grt("H2", temp)
                  -self.grt("CH4",temp) -   self.grt("H2O",temp) )) / press**2
    return kp


  def kprime2(self, temp, press=None):
    """
    Compute second equilibrium constant K2' (Eq. (28) of HL2016) for
    the reaction: CO2 + H2 <-> CO + H2O,
    with k2 = n_CO * n_H2O / n_CO2.

    Parameters
    ----------
    temp: Float scalar or 1D ndarray
       Temperature in Kelvin degrees.

    Returns
    -------
    k2: Float scalar or 1D ndarray
       Second normalized equilibrium constant (same shape as inputs).
    """
    k2 = np.exp(-( self.grt("CO", temp) + self.grt("H2O",temp)
                  -self.grt("CO2",temp) - self.grt("H2", temp) ))
    return k2


  def kprime3(self, temp, press):
    """
    Compute third equilibrium constant K3' (Eq. (29) of HL2016), for
    the reaction: 2*CH4 <-> C2H2 + 3*H2,
    with k3 = n_C2H2 / (n_CH4)**2.

    Parameters
    ----------
    temp: Float scalar or 1D ndarray
       Temperature in Kelvin degrees.
    press: Float scalar or 1D ndarray
       Pressure in bars.

    Returns
    -------
    k3: Float scalar or 1D ndarray
       Third normalized equilibrium constant (same shape as inputs).
    """
    k3 = np.exp(-(   self.grt("C2H2",temp) + 3*self.grt("H2",temp)
                  -2*self.grt("CH4", temp) )) / press**2
    return k3


  def kprime4(self, temp, press):
    """
    Compute fourth equilibrium constant K4 (Eq. (2) of HT2016), for
    the reaction: C2H4 <-> C2H2 + H2,
    with k4 = n_C2H2 / n_CH24.

    Parameters
    ----------
    temp: Float scalar or 1D ndarray
       Temperature in Kelvin degrees.
    press: Float scalar or 1D ndarray
       Pressure in bars.

    Returns
    -------
    k4: Float scalar or 1D ndarray
       Fourth normalized equilibrium constant (same shape as inputs).
    """
    k4 = np.exp(-( self.grt("C2H2",temp) + self.grt("H2",temp)
                  -self.grt("C2H4",temp) )) / press
    return k4


  def kprime5(self, temp, press):
    """
    Compute fifth equilibrium constant K5 (Eq. (2) of HT2016), for
    the reaction: 2*NH3 <-> N2 + 3*H2,
    with k5 = n_N2 / (n_NH3)**2.

    Parameters
    ----------
    temp: Float scalar or 1D ndarray
       Temperature in Kelvin degrees.
    press: Float scalar or 1D ndarray
       Pressure in bars.

    Returns
    -------
    k5: Float scalar or 1D ndarray
       Fifth normalized equilibrium constant (same shape as inputs).
    """
    k5 = np.exp(-(   self.grt("N2", temp) + 3*self.grt("H2",temp)
                  -2*self.grt("NH3",temp) )) / press**2
    return k5


  def kprime6(self, temp, press):
    """
    Compute sixth equilibrium constant K6 (Eq. (2) of HT2016), for
    the reaction: NH3 + CH4 <-> HCN + 3*H2,
    with k6 = n_HCN / (n_NH3 * n_CH4).

    Parameters
    ----------
    temp: Float scalar or 1D ndarray
       Temperature in Kelvin degrees.
    press: Float scalar or 1D ndarray
       Pressure in bars.

    Returns
    -------
    k6: Float scalar or 1D ndarray
       Sixth normalized equilibrium constant (same shape as inputs).
    """
    k6 = np.exp(-( self.grt("HCN",temp) + 3*self.grt("H2", temp)
                  -self.grt("NH3",temp) -   self.grt("CH4",temp) )) / press**2
    return k6


  def HCO_poly6_CO(self, temp, press, f, k1, k2, k3, k4, k5=None, k6=None):
    """
    Compute polynomial coefficients for CO in HCO chemistry considering
    six molecules: H2O, CO, CO2, CH4, C2H2, and C2H4.

    Parameters
    ----------
    temp: Float
       Temperature in Kelvin degrees.
    press: Float
       Pressure in bars.
    f: Float
       Ratio of available hydrogen atoms over molecular-hydrogen particles.
    k1: Float
       First scaled equilibrium constant at input temperature and pressure.
    k2: Float
       Second scaled equilibrium constant at input temperature and pressure.
    k3: Float
       Third scaled equilibrium constant at input temperature and pressure.
    k4: Float
       Fourth scaled equilibrium constant at input temperature and pressure.
    k5: Float
       Fifth scaled equilibrium constant at input temperature and pressure.
    k6: Float
       Sixth scaled equilibrium constant at input temperature and pressure.

    Returns
    -------
    A: List of floats
       CO polynomial coefficients (sorted from lowest to highest degree).
    """
    # Elemental abundances:
    C, O = self.C, self.O
    # Polynomial coefficients sorted from lowest to highest degree:
    A = [-C*O**2*f**3*k1**2*k2**3*k4,
         -C*O**2*f**3*k1**2*k2**2*k4 + 2*C*O*f**2*k1**2*k2**3*k4
           + O**3*f**3*k1**2*k2**2*k4 + O**2*f**2*k1**2*k2**3*k4
           + O*f*k1*k2**3*k4,
         2*C*O*f**2*k1**2*k2**2*k4 - C*f*k1**2*k2**3*k4
           - 2*O**2*f**2*k1**2*k2**2*k4 - 2*O*f*k1**2*k2**3*k4
           + 2*O*f*k1*k2**2*k4 - k1*k2**3*k4 + 2*k2**3*k3*k4 + 2*k2**3*k3,
         -C*f*k1**2*k2**2*k4 + O*f*k1**2*k2**2*k4 + O*f*k1*k2*k4
           + k1**2*k2**3*k4 - 2*k1*k2**2*k4 + 6*k2**2*k3*k4 + 6*k2**2*k3,
         -k1*k2*k4 + 6*k2*k3*k4 + 6*k2*k3,
         2*k3*k4 + 2*k3]
    return A


  def HCO_poly6_H2O(self, temp, press, f, k1, k2, k3, k4, k5=None, k6=None):
    """
    Get polynomial coefficients for H2O in HCO chemistry considering
    six molecules: H2O, CO, CO2, CH4, C2H2, and C2H4.

    Parameters
    ----------
    temp: Float
       Temperature in Kelvin degrees.
    press: Float
       Pressure in bars.
    f: Float
       Ratio of available hydrogen atoms over molecular-hydrogen particles.
    k1: Float
       First scaled equilibrium constant at input temperature and pressure.
    k2: Float
       Second scaled equilibrium constant at input temperature and pressure.
    k3: Float
       Third scaled equilibrium constant at input temperature and pressure.
    k4: Float
       Fourth scaled equilibrium constant at input temperature and pressure.
    k5: Float
       Fifth scaled equilibrium constant at input temperature and pressure.
    k6: Float
       Sixth scaled equilibrium constant at input temperature and pressure.

    Returns
    -------
    A: List of floats
       H2O polynomial coefficients (sorted from lowest to highest degree).
    """
    # Elemental abundances:
    C, O = self.C, self.O
    # Polynomial coefficients sorted from lowest to highest degree:
    A = [2*O**2*f**2*k2**2*k3*k4 + 2*O**2*f**2*k2**2*k3,
         O*f*k1*k2**2*k4 - 4*O*f*k2**2*k3*k4 - 4*O*f*k2**2*k3,
        -C*f*k1**2*k2**2*k4 + O*f*k1**2*k2**2*k4 + O*f*k1*k2*k4
          - k1*k2**2*k4 + 2*k2**2*k3*k4 + 2*k2**2*k3,
        -2*C*f*k1**2*k2*k4 + 2*O*f*k1**2*k2*k4 - k1**2*k2**2*k4 - k1*k2*k4,
        -C*f*k1**2*k4 + O*f*k1**2*k4 - 2*k1**2*k2*k4,
        -k1**2*k4]
    return A


  def HCNO_poly8_CO(self, temp, press, f, k1, k2, k3, k4, k5, k6):
    """
    Get polynomial coefficients for CO in HCNO chemistry considering
    eight molecules: H2O, CO, CH4, C2H2, C2H4, HCN, NH3, and N2.

    Parameters
    ----------
    temp: Float
       Temperature in Kelvin degrees.
    press: Float
       Pressure in bars.
    f: Float
       Ratio of available hydrogen atoms over molecular-hydrogen particles.
    k1: Float
       First scaled equilibrium constant at input temperature and pressure.
    k2: Float
       Second scaled equilibrium constant at input temperature and pressure.
    k3: Float
       Third scaled equilibrium constant at input temperature and pressure.
    k4: Float
       Fourth scaled equilibrium constant at input temperature and pressure.
    k5: Float
       Fifth scaled equilibrium constant at input temperature and pressure.
    k6: Float
       Sixth scaled equilibrium constant at input temperature and pressure.

    Returns
    -------
    A: List of floats
       CO polynomial coefficients (sorted from lowest to highest degree).
    """
    # Elemental abundances:
    C, N, O = self.C, self.N, self.O
    # Polynomial coefficients sorted from lowest to highest degree:
    A = [2*C**2*O**4*f**6*k1**4*k4**2*k5,
        -C*O**3*f**4*k1**3*k4**2*(8*C*f*k1*k5 + 4*O*f*k1*k5 + 4*k5 - k6),
         O**2*f**2*k1**2*k4*(12*C**2*f**2*k1**2*k4*k5 + 16*C*O*f**2*k1**2*k4*k5
                           + 12*C*f*k1*k4*k5 - 3*C*f*k1*k4*k6 - 8*C*f*k3*k4*k5
                           - 8*C*f*k3*k5 + C*f*k4*k6**2 - N*f*k4*k6**2
                           + 2*O**2*f**2*k1**2*k4*k5 + 4*O*f*k1*k4*k5
                           - O*f*k1*k4*k6 + 2*k4*k5 - k4*k6),
        -O*f*k1*k4*(8*C**2*f**2*k1**3*k4*k5 + 24*C*O*f**2*k1**3*k4*k5
                 + 12*C*f*k1**2*k4*k5 - 3*C*f*k1**2*k4*k6 - 16*C*f*k1*k3*k4*k5
                 - 16*C*f*k1*k3*k5 + 2*C*f*k1*k4*k6**2 - 2*N*f*k1*k4*k6**2
                 + 8*O**2*f**2*k1**3*k4*k5 + 12*O*f*k1**2*k4*k5
                 - 3*O*f*k1**2*k4*k6 - 8*O*f*k1*k3*k4*k5 - 8*O*f*k1*k3*k5
                 + O*f*k1*k4*k6**2 + 4*k1*k4*k5 - 2*k1*k4*k6 - 8*k3*k4*k5
                 + 2*k3*k4*k6 - 8*k3*k5 + 2*k3*k6 + k4*k6**2),
         2*C**2*f**2*k1**4*k4**2*k5 + 16*C*O*f**2*k1**4*k4**2*k5
           + 4*C*f*k1**3*k4**2*k5 - C*f*k1**3*k4**2*k6
           - 8*C*f*k1**2*k3*k4**2*k5 - 8*C*f*k1**2*k3*k4*k5
           + C*f*k1**2*k4**2*k6**2 - N*f*k1**2*k4**2*k6**2
           + 12*O**2*f**2*k1**4*k4**2*k5 + 12*O*f*k1**3*k4**2*k5
           - 3*O*f*k1**3*k4**2*k6 - 16*O*f*k1**2*k3*k4**2*k5
           - 16*O*f*k1**2*k3*k4*k5 + 2*O*f*k1**2*k4**2*k6**2 + 2*k1**2*k4**2*k5
           - k1**2*k4**2*k6 - 8*k1*k3*k4**2*k5 + 2*k1*k3*k4**2*k6
           - 8*k1*k3*k4*k5 + 2*k1*k3*k4*k6 + k1*k4**2*k6**2 + 8*k3**2*k4**2*k5
           + 16*k3**2*k4*k5 + 8*k3**2*k5 - 2*k3*k4**2*k6**2 - 2*k3*k4*k6**2,
        -k1**2*k4*(4*C*f*k1**2*k4*k5 + 8*O*f*k1**2*k4*k5 + 4*k1*k4*k5
                   - k1*k4*k6 - 8*k3*k4*k5 - 8*k3*k5 + k4*k6**2),
         2*k1**4*k4**2*k5]
    return A


  def HCNO_poly8_H2O(self, temp, press, f, k1, k2, k3, k4, k5, k6):
    """
    Get polynomial coefficients for H2O in HCNO chemistry considering
    eight molecules: H2O, CO, CH4, C2H2, C2H4, HCN, NH3, and N2.

    Parameters
    ----------
    temp: Float
       Temperature in Kelvin degrees.
    press: Float
       Pressure in bars.
    f: Float
       Ratio of available hydrogen atoms over molecular-hydrogen particles.
    k1: Float
       First scaled equilibrium constant at input temperature and pressure.
    k2: Float
       Second scaled equilibrium constant at input temperature and pressure.
    k3: Float
       Third scaled equilibrium constant at input temperature and pressure.
    k4: Float
       Fourth scaled equilibrium constant at input temperature and pressure.
    k5: Float
       Fifth scaled equilibrium constant at input temperature and pressure.
    k6: Float
       Sixth scaled equilibrium constant at input temperature and pressure.

    Returns
    -------
    A: List of floats
       H2O polynomial coefficients (sorted from lowest to highest degree).
    """
    # Elemental abundances:
    C, N, O = self.C, self.N, self.O
    # Polynomial coefficients sorted from lowest to highest degree:
    A = [2*O**4*f**4*k3*(k4 + 1)*(4*k3*k4*k5 + 4*k3*k5 - k4*k6**2),
         O**3*f**3*(8*k1*k3*k4**2*k5 - 2*k1*k3*k4**2*k6 + 8*k1*k3*k4*k5
             - 2*k1*k3*k4*k6 - k1*k4**2*k6**2 - 32*k3**2*k4**2*k5
             - 64*k3**2*k4*k5 - 32*k3**2*k5 + 8*k3*k4**2*k6**2 + 8*k3*k4*k6**2),
        -O**2*f**2*(8*C*f*k1**2*k3*k4**2*k5 + 8*C*f*k1**2*k3*k4*k5
                  - C*f*k1**2*k4**2*k6**2 + N*f*k1**2*k4**2*k6**2
                  - 8*O*f*k1**2*k3*k4**2*k5 - 8*O*f*k1**2*k3*k4*k5
                  + O*f*k1**2*k4**2*k6**2 - 2*k1**2*k4**2*k5 + k1**2*k4**2*k6
                  + 24*k1*k3*k4**2*k5 - 6*k1*k3*k4**2*k6 + 24*k1*k3*k4*k5
                  - 6*k1*k3*k4*k6 - 3*k1*k4**2*k6**2 - 48*k3**2*k4**2*k5
                  - 96*k3**2*k4*k5 - 48*k3**2*k5 + 12*k3*k4**2*k6**2
                  + 12*k3*k4*k6**2),
        -O*f*(4*C*f*k1**3*k4**2*k5 - C*f*k1**3*k4**2*k6
           - 16*C*f*k1**2*k3*k4**2*k5 - 16*C*f*k1**2*k3*k4*k5
           + 2*C*f*k1**2*k4**2*k6**2 - 2*N*f*k1**2*k4**2*k6**2
           - 4*O*f*k1**3*k4**2*k5 + O*f*k1**3*k4**2*k6
           + 24*O*f*k1**2*k3*k4**2*k5 + 24*O*f*k1**2*k3*k4*k5
           - 3*O*f*k1**2*k4**2*k6**2 + 4*k1**2*k4**2*k5 - 2*k1**2*k4**2*k6
           - 24*k1*k3*k4**2*k5 + 6*k1*k3*k4**2*k6 - 24*k1*k3*k4*k5
           + 6*k1*k3*k4*k6 + 3*k1*k4**2*k6**2 + 32*k3**2*k4**2*k5
           + 64*k3**2*k4*k5 + 32*k3**2*k5 - 8*k3*k4**2*k6**2 - 8*k3*k4*k6**2),
         2*C**2*f**2*k1**4*k4**2*k5 - 4*C*O*f**2*k1**4*k4**2*k5
           + 4*C*f*k1**3*k4**2*k5 - C*f*k1**3*k4**2*k6
           - 8*C*f*k1**2*k3*k4**2*k5 - 8*C*f*k1**2*k3*k4*k5
           + C*f*k1**2*k4**2*k6**2 - N*f*k1**2*k4**2*k6**2
           + 2*O**2*f**2*k1**4*k4**2*k5 - 8*O*f*k1**3*k4**2*k5
           + 2*O*f*k1**3*k4**2*k6 + 24*O*f*k1**2*k3*k4**2*k5
           + 24*O*f*k1**2*k3*k4*k5 - 3*O*f*k1**2*k4**2*k6**2 + 2*k1**2*k4**2*k5
           - k1**2*k4**2*k6 - 8*k1*k3*k4**2*k5 + 2*k1*k3*k4**2*k6
           - 8*k1*k3*k4*k5 + 2*k1*k3*k4*k6 + k1*k4**2*k6**2 + 8*k3**2*k4**2*k5
           + 16*k3**2*k4*k5 + 8*k3**2*k5 - 2*k3*k4**2*k6**2 - 2*k3*k4*k6**2,
         k1**2*k4*(4*C*f*k1**2*k4*k5 - 4*O*f*k1**2*k4*k5 + 4*k1*k4*k5
                - k1*k4*k6 - 8*k3*k4*k5 - 8*k3*k5 + k4*k6**2),
         2*k1**4*k4**2*k5]
    return A


  def solve_rest(self, H2O, CO, f, k1, k2, k3, k4, k5, k6):
    """
    Find abundances for remaining species once H2O and CO are known.
    Note that this also uses self.N.

    Parameters
    ----------
    H2O: Float scalar or 1D ndarray
       Water abundance.
    CO: Float scalar or 1D ndarray
       Carbon monoxide abundance.
    f: Float scalar or 1D ndarray
       Ratio of available hydrogen atoms over molecular-hydrogen particles.
    k1: Float scalar or 1D ndarray
       First scaled equilibrium constant.
    k2: Float scalar or 1D ndarray
       Second scaled equilibrium constant.
    k3: Float scalar or 1D ndarray
       Third scaled equilibrium constant.
    k4: Float scalar or 1D ndarray
       Fourth scaled equilibrium constant.
    k5: Float scalar or 1D ndarray
       Fifth scaled equilibrium constant.
    k6: Float scalar or 1D ndarray
       Sixth scaled equilibrium constant.

    Returns
    -------
    H2O: Float scalar or 1D ndarray
       Water abundance.
    CH4: Float scalar or 1D ndarray
       Methane abundance.
    CO: Float scalar or 1D ndarray
       Carbon monoxide abundance.
    NH3: Float scalar or 1D ndarray
       Ammonia abundance.
    C2H2: Float scalar or 1D ndarray
       Acetylene abundance.
    C2H4: Float scalar or 1D ndarray
       Ethylene abundance.
    HCN: Float scalar or 1D ndarray
       Hydrogen cyanide abundance.
    N2: Float scalar or 1D ndarray
       Molecular nitrogen abundance.
    """
    CH4  = CO / (k1*H2O)
    CO2  = CO * H2O / k2
    C2H2 = k3 * CH4**2
    C2H4 = C2H2 / k4
    # Solve for NH3 from quadratic formula (Eq. (21) of CBD2018):
    b = 1.0 + k6*CH4
    NH3 = (np.sqrt(b**2 + 8*f*k5*self.N) - b) /(4*k5)
    # Use this approximation when 2*K5 << (1+K6*CH4):
    try:
      NH3[8*f*k5*self.N/b**2<1e-6] = f*self.N
    except:
      if 8*f*k5*self.N/b**2 < 1e-6:
        NH3 = f*self.N
    HCN = k6 * NH3 * CH4
    N2  = k5 * NH3**2
    return H2O, CH4, CO, CO2, NH3, C2H2, C2H4, HCN, N2


  def solveH2O(self, poly, temp, press, f, k1, k2, k3, k4, k5, k6, guess=None):
    """
    Wrapper to find root for H2O polynomial for the input poly
    function at given atmospheric properties.

    Parameters
    ----------
    poly: function
       Function to compute polynomial coefficients.
    temp: Float
       Temperature in Kelvin degree.
    press: Float
       Pressure in bar.
    f: Float
       Ratio of available hydrogen atoms over molecular-hydrogen particles.
    k1: Float
       First scaled equilibrium constant.
    k2: Float
       Second scaled equilibrium constant.
    k3: Float
       Third scaled equilibrium constant.
    k4: Float
       Fourth scaled equilibrium constant.
    k5: Float
       Fifth scaled equilibrium constant.
    k6: Float
       Sixth scaled equilibrium constant.
    guess: Float
       Intial guess for Newton-Raphson root finder.

    Returns
    -------
    H2O: Float
       Water abundance.
    CH4: Float
       Methane abundance.
    CO: Float
       Carbon monoxide abundance.
    NH3: Float
       Ammonia abundance.
    C2H2: Float
       Acetylene abundance.
    C2H4: Float
       Ethylene abundance.
    HCN: Float
       Hydrogen cyanide abundance.
    N2: Float
       Molecular nitrogen abundance.
    """
    vmax = f*self.O
    if guess is None:
      guess = 0.99 * vmax
    A   = poly(temp, press, f, k1, k2, k3, k4, k5, k6)
    H2O = bound_nr(A, guess=guess, vmax=vmax)
    CO  = (f*self.O-H2O) / (1.0+2*H2O/k2)
    return np.array(self.solve_rest(H2O, CO, f, k1,k2,k3,k4,k5,k6))


  def solveCO(self, poly, temp, press, f, k1, k2, k3, k4, k5, k6, guess=None):
    """
    Wrapper to find root for CO polynomial for the input poly
    function at given atmospheric properties.

    Parameters
    ----------
    poly: function
       Function to compute polynomial coefficients.
    temp: Float
       Temperature in Kelvin degree.
    press: Float
       Pressure in bar.
    f: Float
       Ratio of available hydrogen atoms over molecular-hydrogen particles.
    k1: Float
       First scaled equilibrium constant.
    k2: Float
       Second scaled equilibrium constant.
    k3: Float
       Third scaled equilibrium constant.
    k4: Float
       Fourth scaled equilibrium constant.
    k5: Float
       Fifth scaled equilibrium constant.
    k6: Float
       Sixth scaled equilibrium constant.
    guess: Float
       Intial guess for Newton-Raphson root finder.

    Returns
    -------
    H2O: Float
       Water abundance.
    CH4: Float
       Methane abundance.
    CO: Float
       Carbon monoxide abundance.
    NH3: Float
       Ammonia abundance.
    C2H2: Float
       Acetylene abundance.
    C2H4: Float
       Ethylene abundance.
    HCN: Float
       Hydrogen cyanide abundance.
    N2: Float
       Molecular nitrogen abundance.
    """
    vmax = f*np.amin((self.C,self.O))
    if guess is None:
      guess = 0.99 * vmax
    A   = poly(temp, press, f, k1, k2, k3, k4, k5, k6)
    CO  = bound_nr(A, guess=guess, vmax=vmax)
    H2O = (f*self.O-CO) / (1.0+2*CO/k2)
    #print("{:.3e}: {:.3e}  {:.3e}  {:.3e} / {:.3e}  {:.3e}"
    #      .format(press, f*self.O, f*self.C, CO, k2, 1.0+2*CO/k2))
    return np.array(self.solve_rest(H2O, CO, f, k1,k2,k3,k4,k5,k6))


  def solve(self, temp, press, C=None, N=None, O=None, poly=None):
    """
    Compute analytic thermochemical equilibrium abundances following
    the prescription of Cubillos, Blecic, & Dobbs-Dixon (2018), ApJ, XX, YY.

    Parameters
    ----------
    temp: 1D float ndarray
       Temperature in Kelvin degree.
    press: 1D float ndarray
       Pressure in bar.
    C: Float
       If not None, update the carbon elemental abundance (C/H).
    N: Float
       If not None, update the nitrogen elemental abundance (N/H).
    O: Float
       If not None, update the oxygen elemental abundance (O/H).
    poly: function
       If not None, enforce poly as the root-finding polynomial.

    Returns
    -------
    Q: 2D float ndarray
       Array of shape (nmol, nlayers) with equilibrium abundances for
       H2O, CH4, CO, CO2, NH3, C2H2, C2H4, HCN, N2, H2, and H.
    """
    nlayers = len(temp)

    # Equilibrium constants:
    k0 = self.kprime0(temp, press)
    k1 = self.kprime(temp, press)
    k2 = self.kprime2(temp)
    k3 = self.kprime3(temp, press)
    k4 = self.kprime4(temp, press)
    k5 = self.kprime5(temp, press)
    k6 = self.kprime6(temp, press)

    # Update elemental abundances if requested:
    if C is not None:
      self.C = C
    if N is not None:
      self.N = N
    if O is not None:
      self.O = O
    C, N, O = self.C, self.N, self.O

    # Hydrogen chemistry:
    Hatom = (-1 + np.sqrt(1+8/k0)) / (4/k0)
    Hmol  = Hatom**2/k0
    f = (Hatom + 2*Hmol) / Hmol

    if poly is not None:
      pass  # TBD

    # Compute reliable abundances:
    Q = np.zeros((self.nmol, nlayers))
    for i in np.arange(nlayers):
      if C/O < 1.0:
        if N/C > 10 and temp[i] > 2000.0:
          if C/O > 0.1:
            Q[:9,i] = self.solveH2O(self.HCNO_poly8_H2O, temp[i], press[i],f[i],
                                   k1[i], k2[i], k3[i], k4[i], k5[i], k6[i])
          else:
            Q[:9,i] = self.solveCO(self.HCNO_poly8_CO, temp[i], press[i], f[i],
                                   k1[i], k2[i], k3[i], k4[i], k5[i], k6[i])
        else:
          Q[:9,i] = self.solveCO(self.HCO_poly6_CO, temp[i], press[i], f[i],
                                   k1[i], k2[i], k3[i], k4[i], k5[i], k6[i])
      else:
        if press[i] > top(temp[i], C, N, O):
          # Lower atmosphere:
          Q[:9,i] = self.solveCO(self.HCO_poly6_CO, temp[i], press[i], f[i],
                                   k1[i], k2[i], k3[i], k4[i], k5[i], k6[i])
        else:
          # Upper atmosphere:
          if N/C > 0.1 and temp[i] > 900.0:
            Q[:9,i] = self.solveH2O(self.HCNO_poly8_H2O, temp[i], press[i],f[i],
                                   k1[i], k2[i], k3[i], k4[i], k5[i], k6[i])
          else:
            Q[:9,i] = self.solveH2O(self.HCO_poly6_H2O, temp[i], press[i], f[i],
                                   k1[i], k2[i], k3[i], k4[i], k5[i], k6[i])

    # De-normalize by H2:
    Q *= Hmol
    # Set hydrogen and helium abundances:
    Q[ 9] = Hmol
    Q[10] = Hatom
    # Get mol mixing fracions:
    Q /= np.sum(Q, axis=0)
    return Q
