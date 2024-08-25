import numpy as np
import matplotlib.pyplot as plt


"""check some of the accuracy in the Taylor expansions...
"""

def get_MN_eigen_exact(number_of_lenses, focal_length, lens_space):

    t2 = 1 - (lens_space/(2*focal_length))
    t1 = np.sqrt( 1 - (t2)**2 )
    phi = np.arctan( t1 / t2 )

    c = np.cos(number_of_lenses*phi)
    s = np.sin(number_of_lenses*phi)

    MN = np.array([[c,             focal_length*s*np.sin(phi)],
                   [-s / (np.sin(phi)*focal_length),    c    ]])
    return MN


def get_MN_numerical(distance, focal_length, N):

    Mf = np.array([[ 1.,   distance/2.],
                  [ 0,        1.     ]])

    Ml = np.array([[1.,               0 ],
                  [-1./focal_length, 1.]])

    M = Mf @ Ml @ Mf

    Mnumerical = M.copy()
    for i in range(number_of_lenses-1): Mnumerical = Mnumerical @ M

    return Mnumerical

def free_space_propagate( ray, distance ):
    M = np.array([[ 1.,   distance],
                  [ 0,        1.     ]])
    return M @ ray

def thin_lens_propagate( ray, focal_length ):
    M = np.array([[1.,               0 ],
                  [-1./focal_length, 1.]])
    return M @ ray

if __name__ == "__main__":

    number_of_lenses = 50 # N
    lens_space = 2 * 1e-3 # T
    lens_radius = 50 * 1e-6 # R
    refractive_decrement = 1.65 * 1e-6  # delta
    focal_length = lens_radius / ( 2 * refractive_decrement) # f

    # equation (4), simons 2016
    phi = np.sqrt( lens_space / focal_length ) # phi

    focal_length_CRL = focal_length * phi * ( 1. / np.tan(number_of_lenses*phi) )

    y = lens_radius / 2.
    alpha = 0.

    ray0 = np.array([y, alpha])

    ray = ray0.copy()

    rays_stack = np.zeros((number_of_lenses, 2))
    states = np.array(range(number_of_lenses)).astype(int)
    for i in states:
        ray = free_space_propagate(ray, lens_space / 2.)
        ray = thin_lens_propagate(ray, focal_length)
        ray = free_space_propagate(ray, lens_space / 2.)
        rays_stack[i, :] = ray[0]

    distance = - ray[0] / ray[1]

    MN = get_MN_eigen_exact(number_of_lenses, focal_length, lens_space)
    MNnumerical = get_MN_numerical(lens_space, focal_length, number_of_lenses)
    ray_final = MN @ ray0

    focal_length_CRL_2 = - ray_final[0] / ray_final[1]

    print(number_of_lenses*lens_space/2.)
    print(focal_length_CRL-focal_length_CRL_2)
    print(MN - MNnumerical)

    plt.figure()
    plt.plot(states, rays_stack[:, 0 ]*1e6, 'k-')
    plt.grid()
    plt.ylabel('Ray-offset from optical axis [microns]')
    plt.xlabel('Lens Number')
    plt.show()