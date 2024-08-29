import numpy as np
import matplotlib.pyplot as plt
import vtk
import cProfile
import pstats
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from scipy.spatial.transform import Rotation
from scipy.stats import vonmises_fisher

def load_image(path, shape):
    image = plt.imread('crystals.png')
    n, m, _ = image.shape
    x = np.arange(0, n)
    y = np.arange(0, m)
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    interp_R = RegularGridInterpolator((x, y), R, method='linear')
    interp_G = RegularGridInterpolator((x, y), G, method='linear')
    interp_B = RegularGridInterpolator((x, y), B, method='linear')
    res_n, res_m = shape
    x_new = np.linspace(0, n - 1, res_n)
    y_new = np.linspace(0, m - 1, res_m)
    xv, yv = np.meshgrid(x_new, y_new, indexing='ij')
    coords = np.column_stack((xv.ravel(), yv.ravel()))
    r_vals = interp_R(coords)
    g_vals = interp_G(coords)
    b_vals = interp_B(coords)
    rgb_vals = np.stack([r_vals, g_vals, b_vals], axis=-1)
    interpolated_image = rgb_vals.reshape((res_n, res_m, 3))
    return interpolated_image

def render_image(pixel_size, y, z, rgb_vals):
    D = np.max([np.max(np.abs(y)), np.max(np.abs(z))])
    dg = np.arange(-D-pixel_size, D + 2*pixel_size, pixel_size)
    Y, Z = np.meshgrid(dg, dg, indexing='ij')
    rendered_image = np.zeros((Y.shape[0],Y.shape[1],3))
    counts = np.zeros((Y.shape[0],Y.shape[1]))
    yi = np.round( len(dg) * (y-np.min(dg))/(np.max(dg)-np.min(dg)) ).astype(int)
    zi = np.round( len(dg) * (z-np.min(dg))/(np.max(dg)-np.min(dg)) ).astype(int)
    np.add.at(counts[:,:], (yi, zi), np.ones((len(y),)))
    np.add.at(rendered_image[:,:,0], (yi, zi), rgb_vals[:,0])
    np.add.at(rendered_image[:,:,1], (yi, zi), rgb_vals[:,1])
    np.add.at(rendered_image[:,:,2], (yi, zi), rgb_vals[:,2])

    mask = counts!=0
    rendered_image[mask] = rendered_image[mask] / counts[mask, np.newaxis]

    xi = np.array([Y.flatten(), Z.flatten()]).T
    points = np.array([Y[mask], Z[mask]]).T
    r = rendered_image[mask, 0]
    g = rendered_image[mask, 1]
    b = rendered_image[mask, 2]
    interp_image = rendered_image.copy()*0

    m,n,_ = rendered_image.shape
    interp_image[:, :, 0] = griddata(points, r, xi, method='linear', fill_value=0).reshape(m, n)
    interp_image[:, :, 1] = griddata(points, g, xi, method='linear', fill_value=0).reshape(m, n)
    interp_image[:, :, 2] = griddata(points, b, xi, method='linear', fill_value=0).reshape(m, n)

    interp_counts = griddata(points, counts[mask], xi, method='linear', fill_value=0).reshape(m, n)

    return counts, interp_image

FACELIST = [
    [0, 1, 2, 3],  # Front face
    [4, 5, 6, 7],  # Back face
    [0, 1, 5, 4],  # Bottom face
    [2, 3, 7, 6],  # Top face
    [0, 3, 7, 4],  # Left face
    [1, 2, 6, 5]   # Right face
]

def detector_plane_to_paraview(x, number_of_lenses, lens_space, lens_radius, fname):

    t = number_of_lenses*lens_space / 20.
    w = 0.17*2
    h = 0.17*2

    points = vtk.vtkPoints()
    points.InsertNextPoint(-0+x, -w, -h)  # 0
    points.InsertNextPoint( t+x, -w, -h)  # 1
    points.InsertNextPoint( t+x,  w, -h)  # 2
    points.InsertNextPoint(-0+x,  w, -h)  # 3
    points.InsertNextPoint(-0+x, -w,  h)  # 4
    points.InsertNextPoint( t+x, -w,  h)  # 5
    points.InsertNextPoint( t+x,  w,  h)  # 6
    points.InsertNextPoint(-0+x,  w,  h)  # 7

    faces = vtk.vtkCellArray()
    for facelist in FACELIST:
        faces.InsertNextCell(4)
        for faceindex in facelist:
            faces.InsertCellPoint(faceindex)

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetPolys(faces)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(poly_data)
    writer.Write()

def sample_plane_to_paraview(x, number_of_lenses, lens_space, lens_radius, fname):

    t = number_of_lenses*lens_space / 20.
    w = 0.12
    h = 0.12

    points = vtk.vtkPoints()
    points.InsertNextPoint(-0+x, -w, -h)  # 0
    points.InsertNextPoint( t+x, -w, -h)  # 1
    points.InsertNextPoint( t+x,  w, -h)  # 2
    points.InsertNextPoint(-0+x,  w, -h)  # 3
    points.InsertNextPoint(-0+x, -w,  h)  # 4
    points.InsertNextPoint( t+x, -w,  h)  # 5
    points.InsertNextPoint( t+x,  w,  h)  # 6
    points.InsertNextPoint(-0+x,  w,  h)  # 7

    faces = vtk.vtkCellArray()
    for facelist in FACELIST:
        faces.InsertNextCell(4)
        for faceindex in facelist:
            faces.InsertCellPoint(faceindex)

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetPolys(faces)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(poly_data)
    writer.Write()

def backfocal_to_paraview(x, number_of_lenses, lens_space, lens_radius, fname):
    t = number_of_lenses*lens_space / 20.
    w = 0.05
    h = 0.05

    points = vtk.vtkPoints()
    points.InsertNextPoint(-t+x, -w, -h)  # 0
    points.InsertNextPoint( t+x, -w, -h)  # 1
    points.InsertNextPoint( t+x,  w, -h)  # 2
    points.InsertNextPoint(-t+x,  w, -h)  # 3
    points.InsertNextPoint(-t+x, -w,  h)  # 4
    points.InsertNextPoint( t+x, -w,  h)  # 5
    points.InsertNextPoint( t+x,  w,  h)  # 6
    points.InsertNextPoint(-t+x,  w,  h)  # 7

    faces = vtk.vtkCellArray()
    for facelist in FACELIST:
        faces.InsertNextCell(4)
        for faceindex in facelist:
            faces.InsertCellPoint(faceindex)

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetPolys(faces)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(poly_data)
    writer.Write()

def lens_to_paraview(number_of_lenses, lens_space, lens_radius, fname):

    t = number_of_lenses*lens_space
    w = 2.5*lens_radius * 1e3

    cylinder = vtk.vtkCylinderSource()
    cylinder.SetRadius(w)
    cylinder.SetHeight(t)
    cylinder.SetResolution(50)
    cylinder.SetCenter(0, 0, 0)

    cylinder.CappingOn()
    cylinder.Update()

    transform = vtk.vtkTransform()
    transform.RotateWXYZ(90, 0, 0, 1)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputConnection(cylinder.GetOutputPort())
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(transformFilter.GetOutput())
    writer.Write()


def optical_axis_to_paraview(fname, L, d1):
    optical_axis = np.zeros((1, 2, 3))
    optical_axis[0, 0, 0] = -d1
    optical_axis[0, 1, 0] = L-d1
    ray_data_to_paraview(optical_axis, fname)


def ray_data_to_paraview(ray_data, fname):

    vtk_points = vtk.vtkPoints()
    vtk_lines = vtk.vtkCellArray()

    coordinates = ray_data.copy()
    coordinates[:, :, 1] *=5 * 1e3
    coordinates[:, :, 2] *=5 * 1e3

    for coord in coordinates    :

        point_ids = []

        for c in coord:
            point_ids.append(vtk_points.InsertNextPoint(c))

        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(point_ids))
        for i, pid in enumerate(point_ids):
            line.GetPointIds().SetId(i, pid)
        vtk_lines.InsertNextCell(line)

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(vtk_lines)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(poly_data)
    writer.Write()

def free_space_propagate( rays, distance ):
    M = np.array([[ 1.,   distance],
                  [ 0,        1.     ]])
    return M @ rays

def thin_lens_propagate( rays, focal_length ):
    M = np.array([[1.,               0 ],
                  [-1./focal_length, 1.]])
    return M @ rays


def numpy_to_vtk_image_data(np_array, filename):
    image_data = (np_array*255).astype(np.uint8)
    width, height, _ = image_data.shape

    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(width, height, 1)
    vtk_image.SetSpacing(1.0, 1.0, 1.0)
    vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)

    for i in range(height):
        for j in range(width):
            vtk_image.SetScalarComponentFromDouble(j, i, 0, 0, image_data[i, j, 0])  # Red
            vtk_image.SetScalarComponentFromDouble(j, i, 0, 1, image_data[i, j, 1])  # Green
            vtk_image.SetScalarComponentFromDouble(j, i, 0, 2, image_data[i, j, 2])  # Blue

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(vtk_image)
    writer.Write()

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()

    number_of_lenses = 50 # N
    lens_space = 1.6 * 1e-3 # T
    lens_radius = 50 * 1e-6 # R
    refractive_decrement = 2.359 * 1e-6  # delta
    focal_length = lens_radius / ( 2 * refractive_decrement) # f

    number_of_rays = 5000

    # Approximations, equation (4), simons 2016
    # phi = np.sqrt( lens_space / focal_length ) # phi
    # f_N = focal_length_CRL = focal_length * phi * ( 1. / np.tan(number_of_lenses*phi) )

    # exact equations...
    N = number_of_lenses
    f = focal_length
    T = lens_space
    t2 = 1 - (T/(2*f))
    t1 = np.sqrt( 1 - (t2)**2 )
    phi = np.arctan( t1 / t2 )
    c = np.cos(N*phi)
    s = np.sin(N*phi)
    f_N = focal_length_CRL = focal_length * np.sin(phi) * ( 1. / np.tan(number_of_lenses*phi) )

    #equation (16), simons 2016
    start_x = - 1.4 * focal_length_CRL
    d1 = np.abs(start_x)
    denom = ((1 / f_N) - (1/d1))
    d2 = ( 1 + (np.tan(N * phi)*f * np.sin(phi) / d1) ) / denom # exact with sin(phi)

    L = number_of_lenses*lens_space + d1 + d2

    Magnification = (d2/(f*np.sin(phi)))-c

    print('M   :  ', Magnification)
    print('L   :  ', L)
    print('d1  :  ', d1)
    print('d2  :  ', d2)
    print('fN  :  ', f_N)
    print('NT  :  ', N*T)

    yg = zg = np.linspace(-0.55*lens_radius, 0.55*lens_radius, 512)
    X, Y =np.meshgrid(yg,zg,indexing='ij')

    image = load_image('crystals.png', shape=X.shape)

    angle = np.random.rand(number_of_rays, )*2*np.pi
    r = 0.3*(np.sqrt(np.random.rand(number_of_rays, )))*lens_radius # radius from optical axis

    #r, angle = r*0, angle*0
    #r += lens_radius / 3.

    s, c = np.sin(angle), np.cos(angle)
    #Rx = np.array([[1,0,0],[0,c,-s],[0,s,c]])

    x0, y0, z0 = 0, -s*r, c*r

    yi = np.round( len(yg) * (y0-np.min(yg))/(np.max(yg)-np.min(yg)) ).astype(int)
    zi = np.round( len(zg) * (z0-np.min(zg))/(np.max(zg)-np.min(zg)) ).astype(int)

    image_data = image[yi, zi, :]

    sigma_alpha = (0.01*1e-3)**2
    kappa = 1/sigma_alpha
    vmf = vonmises_fisher(mu=np.array([1, 0, 0]), kappa=kappa)
    alpha_vecs = vmf.rvs(number_of_rays)

    alpha_xy = np.arccos( alpha_vecs[:, (0, 1) ][:, 0] / np.linalg.norm(alpha_vecs[:, (0, 1) ], axis=1) )
    alpha_xy *= np.sign(alpha_vecs[:, 1 ])
    alpha_xz = np.arccos( alpha_vecs[:, (0, 2) ][:, 0] / np.linalg.norm(alpha_vecs[:, (0, 2) ], axis=1) )
    alpha_xz *= np.sign(alpha_vecs[:, 2 ])

    r_xy = y0 * 0
    r_xz = z0*0 + lens_radius/4.

    rays_xy0 = np.array([r_xy, alpha_xy])
    rays_xz0 = np.array([r_xz, alpha_xz])

    ####################################################################################
    # X-Z shadow paths....

    rays = rays_xz0.copy()

    # the three extra (+3) states represents the sample-, bacfocal- and image-planes.
    rays_stack = np.zeros((number_of_rays, number_of_lenses + 3, 3))
    states = np.array(range(number_of_lenses + 3)).astype(int)
    x = start_x
    rays_stack[:, 0, 0] = x
    rays_stack[:, 0, 1:] = rays.T

    rays = free_space_propagate(rays, d1)

    x = x + d1

    M = np.array([[ 1.,   d1],
                [ 0,        1.     ]])
    for i in range(number_of_lenses):
        #if i!=0:

        rays = free_space_propagate(rays, lens_space / 2.)
        x = x + lens_space/2.

        M = np.array([[ 1.,   lens_space / 2.],
                  [ 0,        1.     ]]) @ M

        rays = thin_lens_propagate(rays, focal_length)

        #if i!=number_of_lenses-1:

        M = np.array([[1.,               0 ],
                  [-1./focal_length, 1.]]) @ M

        rays = free_space_propagate(rays, lens_space / 2.)
        x = x + lens_space/2.

        M = np.array([[ 1.,   lens_space / 2.],
                  [ 0,        1.     ]]) @ M

        rays_stack[:, i+1, 1:] = rays.T
        rays_stack[:, i+1, 0] = x
    M = np.array([[ 1.,   d2],
                [ 0,        1.     ]]) @ M
    print(M)

    x = x + focal_length_CRL
    rays = free_space_propagate(rays, focal_length_CRL)
    rays_stack[:, -2, 1:] = rays.T
    rays_stack[:, -2, 0] = x

    x = x + d2 - focal_length_CRL  # the relative image plane coordinate
    rays = free_space_propagate(rays, d2 - focal_length_CRL )
    rays_stack[:, -1, 1:] = rays.T
    rays_stack[:, -1, 0] = x

    ray_xyz_coord = np.zeros((number_of_rays, 3, rays_stack.shape[1]))
    ray_xyz_coord[:, 0, :] = rays_stack[:, :, 0]
    ray_xyz_coord[:, 2, :] = rays_stack[:, :, 1]

    shadow_paths_xz = np.zeros((number_of_rays, number_of_lenses + 3, 3))
    shadow_paths_xz[:, :, 0] = ray_xyz_coord[:, 0, :]
    shadow_paths_xz[:, :, 2] = ray_xyz_coord[:, 2, :]


    # X-Y shadpw paths ####################################################################################

    rays = rays_xy0.copy()

    # the three extra (+3) states represents the sample-, bacfocal- and image-planes.
    rays_stack = np.zeros((number_of_rays, number_of_lenses + 3, 3))
    states = np.array(range(number_of_lenses + 3)).astype(int)
    x = start_x
    rays_stack[:, 0, 0] = x
    rays_stack[:, 0, 1:] = rays.T

    rays = free_space_propagate(rays, d1)

    x = x + d1

    M = np.array([[ 1.,   d1],
                [ 0,        1.     ]])
    for i in range(number_of_lenses):
        #if i!=0:

        rays = free_space_propagate(rays, lens_space / 2.)
        x = x + lens_space/2.

        M = np.array([[ 1.,   lens_space / 2.],
                  [ 0,        1.     ]]) @ M

        rays = thin_lens_propagate(rays, focal_length)

        #if i!=number_of_lenses-1:

        M = np.array([[1.,               0 ],
                  [-1./focal_length, 1.]]) @ M

        rays = free_space_propagate(rays, lens_space / 2.)
        x = x + lens_space/2.

        M = np.array([[ 1.,   lens_space / 2.],
                  [ 0,        1.     ]]) @ M

        rays_stack[:, i+1, 1:] = rays.T
        rays_stack[:, i+1, 0] = x
    M = np.array([[ 1.,   d2],
                [ 0,        1.     ]]) @ M
    print(M)

    x = x + focal_length_CRL
    rays = free_space_propagate(rays, focal_length_CRL)
    rays_stack[:, -2, 1:] = rays.T
    rays_stack[:, -2, 0] = x

    x = x + d2 - focal_length_CRL  # the relative image plane coordinate
    rays = free_space_propagate(rays, d2 - focal_length_CRL )
    rays_stack[:, -1, 1:] = rays.T
    rays_stack[:, -1, 0] = x

    ray_xyz_coord = np.zeros((number_of_rays, 3, rays_stack.shape[1]))
    ray_xyz_coord[:, 0, :] = rays_stack[:, :, 0]
    ray_xyz_coord[:, 2, :] = rays_stack[:, :, 1]

    shadow_paths_xy = np.zeros((number_of_rays, number_of_lenses + 3, 3))
    shadow_paths_xy[:, :, 0] = ray_xyz_coord[:, 0, :]
    shadow_paths_xy[:, :, 2] = ray_xyz_coord[:, 2, :]

    # Reconstruct 3D rays form shadow paths.

    ray_data = np.zeros((number_of_rays, number_of_lenses + 3, 3))
    ray_data[:, :, 0] = shadow_paths_xz[:, :, 0]
    ray_data[:, :, 1] = shadow_paths_xy[:, :, 2]
    ray_data[:, :, 2] = shadow_paths_xz[:, :, 2]

    #ray_data[:, :, 1] = c[:, np.newaxis]*ray_xyz_coord[:, 1, :] - s[:, np.newaxis]*ray_xyz_coord[:, 2, :]
    #ray_data[:, :, 2] = s[:, np.newaxis]*ray_xyz_coord[:, 1, :] + c[:, np.newaxis]*ray_xyz_coord[:, 2, :]

    #plt.figure()
    #dx = lens_radius/200
    #plt.hist(rays[0,:], bins=np.arange(-lens_radius*3, dx + lens_radius*3., dx))
    #plt.xlabel('z dist')
    #plt.show()

    ray_data_to_paraview(ray_data, fname='CRL_simulation.vtk')
    lens_to_paraview(number_of_lenses, lens_space, lens_radius, fname='CRL_lens.vtk')
    x = np.mean(ray_data[:,-2,0])
    backfocal_to_paraview(x, number_of_lenses, lens_space, lens_radius,  fname='backfocal.vtk')
    x = np.mean(ray_data[:,0,0])
    sample_plane_to_paraview(x, number_of_lenses, lens_space, lens_radius, fname='sample_plane.vtk')
    detector_plane_to_paraview(np.mean(ray_data[:,-1,0]), number_of_lenses, lens_space, lens_radius, fname='detector_plane.vtk')
    optical_axis_to_paraview('optical_axis.vtk', L, d1)

    pr.disable()
    pr.dump_stats('tmp_profile_dump')
    ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
    ps.print_stats(15)
    print("")

    if 0:
        ax = plt.figure(figsize=(12,6)).add_subplot(projection='3d')
        for j in range(number_of_rays):
            x,y,z = ray_data[j].T
            ax.plot(x, y*1e6 , z*1e6 , 'k-', alpha = 1 / (0.5*np.sqrt(number_of_rays)), )
        ax.grid()
        ax.set_zlabel('z-coordinate [microns]')
        ax.set_ylabel('y-coordinate [microns]')
        ax.set_xlabel('x-coordinate [m]')

    #plt.ylim([-1.5*lens_radius*1e6, 1.5*lens_radius*1e6])

    x = ray_data[:, -2, 0]*1e6
    y = ray_data[:, -2, 1]*1e6
    z = ray_data[:, -2, 2]*1e6

    alpha0 = rays_stack[:, 0, 2]
    if 0:
        plt.figure()
        plt.scatter(y, z, c=np.abs(alpha0), alpha=0.5, cmap='jet')
        plt.colorbar()

    y = ray_data[:, -1, 1]*1e6
    z = ray_data[:, -1, 2]*1e6

    counts, image = render_image(pixel_size=0.25, y=y, z=z, rgb_vals=image_data)

    numpy_to_vtk_image_data(image, filename='detector_image.vti')

    plt.figure()
    plt.title('Image Plane')
    plt.imshow(image)

    plt.figure()
    plt.title('Image Plane Sampling')
    plt.imshow(counts)

    ymax = np.max(np.abs(y))*1.05
    zmax = np.max(np.abs(z))*1.05

    # plt.figure()
    # plt.title('Detector plane')
    # plt.scatter(y, z, c=image_data, alpha=0.5, cmap='jet')
    # plt.xlim([-ymax, ymax])
    # plt.ylim([-zmax, zmax])
    # plt.colorbar()

    y = ray_data[:, 0, 1]*1e6
    z = ray_data[:, 0, 2]*1e6

    counts, image = render_image(pixel_size=0.1, y=y, z=z, rgb_vals=image_data)
    plt.figure()
    plt.title('Sample Plane')
    plt.imshow(image)

    plt.figure()
    plt.title('Sample Plane Sampling')
    plt.imshow(counts)

    # plt.figure()
    # plt.title('Sample plane')
    # plt.scatter(y, z, c=image_data, alpha=0.5, cmap='jet')
    # plt.xlim([-ymax, ymax])
    # plt.ylim([-zmax, zmax])
    # plt.colorbar()


    plt.show()
