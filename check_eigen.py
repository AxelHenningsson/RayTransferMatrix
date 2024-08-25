
import numpy as np
import matplotlib.pyplot as plt


"""Check the accuracy in the eigenvalue decomposition in simons 2017 with regards to the CRL approximation.
"""

def get_MN_eigen_exact(number_of_lenses, focal_length, lens_space):
    N = number_of_lenses
    f = focal_length
    T = lens_space

    t2 = 1 - (T/(2*f))
    t1 = np.sqrt( 1 - (t2)**2 )
    phi = np.arctan( t1 / t2 )

    #print( np.sqrt(T / f), phi )
    c = np.cos(N*phi)
    s = np.sin(N*phi)

    MN = np.array([[c,             f*s*np.sin(phi)],
                   [-s / (np.sin(phi)*f),    c    ]])
    return MN

if __name__ == "__main__":

    number_of_lenses = 50 # N
    lens_space = 2 * 1e-3 # T
    lens_radius = 50 * 1e-6 # R
    refractive_decrement = 1e-5 # delta
    focal_length = lens_radius / ( 2 * refractive_decrement) # f

    MN = get_MN_eigen_exact(number_of_lenses, focal_length, lens_space)

    Mf = np.array([[ 1.,    lens_space/2.],
                   [ 0,       1.]]).astype(np.float64)

    Ml = np.array([[1.,    0 ],
                   [-1./focal_length, 1.]]).astype(np.float64)

    M = Mf @ Ml @ Mf
    Mnumerical = M.copy()
    for i in range(number_of_lenses-1): Mnumerical = Mnumerical @ M

    print(Mnumerical, '\n\n', MN)

