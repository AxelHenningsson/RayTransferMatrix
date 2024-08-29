
import numpy as np
import matplotlib.pyplot as plt


"""Check the accuracy in the eigenvalue decomposition in simons 2017 with regards to the CRL approximation.
"""
def get_K(number_of_lenses, focal_length, lens_space):

    N = number_of_lenses
    f = focal_length
    T = lens_space

    t2 = 1 - (T/(2*f))
    t1 = np.sqrt( 1 - (t2)**2 )
    phi = np.arctan( t1 / t2 )
    c = np.cos(N*phi)
    s = np.sin(N*phi)

    f_N = focal_length_CRL = focal_length * np.sin(phi) * ( 1. / np.tan(number_of_lenses*phi) )

    #phi_approx = np.sqrt( lens_space / focal_length )
    #f_N_approx =  focal_length * phi * ( 1. / np.tan(number_of_lenses*phi_approx) )
    #print(1 - f_N_approx / f_N)
    #raise

    start_x = - 2.0 * focal_length_CRL
    d1 = np.abs(start_x)
    denom = ((1 / f_N) - (1/d1))

    d2 = ( 1 + (np.tan(N * phi)*f * np.sin(phi) / d1) ) / denom
    #d2 = (f*np.sin(phi)*s - d1*c) / (-d1*(s/(np.sin(phi)*f)) + c)
    residual = (1/d1) + (1/d2) - (1/f_N) + ( (f*np.sin(phi)*np.tan(phi*N)) / (d1*d2) )

    res2 = f*np.sin(phi)*s + d1*(c  - d2*(s/(np.sin(phi)*f))) + d2*c

    # f*np.sin(phi)*s + d1*(c  - d2*(s/(np.sin(phi)*f))) + d2*c = 0
    # -d1*d2*(s/(np.sin(phi)*f)) + d2*c = f*np.sin(phi)*s - d1*c
    # d2*(-d1*(s/(np.sin(phi)*f)) + c) = f*np.sin(phi)*s - d1*c
    # d2 = (f*np.sin(phi)*s - d1*c) / (-d1*(s/(np.sin(phi)*f)) + c)

    D1 = np.array([[ 1.,   d1],
            [ 0,        1.     ]])

    D2 = np.array([[ 1.,   d2],
            [ 0,        1.     ]])

    MN = np.array([[c,             f*s*np.sin(phi)],
                   [-s / (np.sin(phi)*f),    c    ]])

    K = D2 @ MN @ D1

    print(residual, res2, K)
    raise

    return K

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
    lens_space = 1.6 * 1e-3 # T
    lens_radius = 50 * 1e-6 # R
    refractive_decrement = 2.359 * 1e-6  # delta
    focal_length = lens_radius / ( 2 * refractive_decrement) # f

    K = get_K(number_of_lenses, focal_length, lens_space)
    raise

    MN = get_MN_eigen_exact(number_of_lenses, focal_length, lens_space)

    Mf = np.array([[ 1.,    lens_space/2.],
                   [ 0,       1.]]).astype(np.float64)

    Ml = np.array([[1.,    0 ],
                   [-1./focal_length, 1.]]).astype(np.float64)

    M = Mf @ Ml @ Mf
    Mnumerical = M.copy()
    for i in range(number_of_lenses-1): Mnumerical = Mnumerical @ M

    print(Mnumerical, '\n\n', MN)

