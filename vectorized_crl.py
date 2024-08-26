import numpy as np
import matplotlib.pyplot as plt
import vtk
import cProfile
import pstats

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
    w = 0.14*2
    h = 0.14*2

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
    w = 0.12
    h = 0.12

    points = vtk.vtkPoints()
    points.InsertNextPoint(-0, -w, -h)  # 0
    points.InsertNextPoint( t, -w, -h)  # 1
    points.InsertNextPoint( t,  w, -h)  # 2
    points.InsertNextPoint(-0,  w, -h)  # 3
    points.InsertNextPoint(-0, -w,  h)  # 4
    points.InsertNextPoint( t, -w,  h)  # 5
    points.InsertNextPoint( t,  w,  h)  # 6
    points.InsertNextPoint(-0,  w,  h)  # 7

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


# Convert NumPy array to VTK ImageData
def numpy_to_vtk_image_data(np_array, filename):
    # Create VTK image data object
    vtk_image = vtk.vtkImageData()

    # Set the dimensions of the image data (x, y, z)
    dimensions = np_array.shape
    vtk_image.SetDimensions(dimensions[1], dimensions[0], 1)

    # Set the scalar type and number of scalar components
    vtk_image.AllocateScalars(vtk.VTK_FLOAT, 1)

    # Get pointer to the image data's scalar pointer
    scalars = vtk_image.GetPointData().GetScalars()

    # Copy NumPy array to VTK scalars
    for j in range(dimensions[0]):
        for i in range(dimensions[1]):
            vtk_image.SetScalarComponentFromFloat(i, j, 0, 0, np_array[j, i])

    transform = vtk.vtkTransform()
    transform.Translate((0,0,0))  # Set position in x, y, z
    transform.RotateX(0)  # Set rotation around X-axis
    transform.RotateY(90)  # Set rotation around Y-axis
    transform.RotateZ(0)  # Set rotation around Z-axis

    # Apply the transform to the image data
    transform_filter = vtk.vtkImageReslice()  # Correct class for image data transformation
    transform_filter.SetInputData(vtk_image)
    transform_filter.SetResliceTransform(transform)
    transform_filter.SetInterpolationModeToLinear()
    transform_filter.Update()

    vtk_image = transform_filter.GetOutput()


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

    number_of_rays = 2000
    ray_data = np.zeros((number_of_rays, number_of_lenses + 3, 3))

    # equation (4), simons 2016
    phi = np.sqrt( lens_space / focal_length ) # phi
    focal_length_CRL = focal_length * phi * ( 1. / np.tan(number_of_lenses*phi) ) + lens_space/2. # measured from exit lens surface...
    start_x = -focal_length_CRL


    yg = zg = np.linspace(-0.55*lens_radius, 0.55*lens_radius, 128)
    X, Y =np.meshgrid(yg,zg,indexing='ij')
    image = np.zeros( (128, 128) )
    image[32:64, 16:100] = 1
    image[16:110, 32:54] = 1
    image[100:122, 100:122] = 1

    numpy_to_vtk_image_data(image, filename='test.vti')


    plt.figure()
    plt.imshow(image)

    r = 0.4*(np.random.rand(number_of_rays, )-0.5)*2*lens_radius # radius from optical axis

    angle = np.random.rand(number_of_rays, )*2*np.pi
    s, c = np.sin(angle), np.cos(angle)
    #Rx = np.array([[1,0,0],[0,c,-s],[0,s,c]])

    x0, y0, z0 = 0, -s*r, c*r

    yi = np.round( len(yg) * (y0-np.min(yg))/(np.max(yg)-np.min(yg)) ).astype(int)
    zi = np.round( len(zg) * (z0-np.min(zg))/(np.max(zg)-np.min(zg)) ).astype(int)

    image_data = image[yi, zi]

    alpha = 0 * 0.02 * (np.random.rand(number_of_rays,)-0.5)*2 * 1e-3

    rays0 = np.array([r, alpha])

    rays = rays0.copy()

    # the three extra (+3) states represents the sample-, bacfocal- and image-planes.
    rays_stack = np.zeros((number_of_rays, number_of_lenses + 3, 3))
    states = np.array(range(number_of_lenses + 3)).astype(int)
    x = start_x
    rays_stack[:, 0, 0] = x
    rays_stack[:, 0, 1:] = rays.T

    ray = free_space_propagate(rays, np.abs(start_x))

    x = x + np.abs(start_x)
    for i in range(number_of_lenses):
        if i!=0:
            rays = free_space_propagate(rays, lens_space / 2.)
            x = x + lens_space/2.

        rays = thin_lens_propagate(rays, focal_length)

        if i!=number_of_lenses-1:
            rays = free_space_propagate(rays, lens_space / 2.)
            x = x + lens_space/2.

        rays_stack[:, i+1, 1:] = rays.T
        rays_stack[:, i+1, 0] = x

    x = x + focal_length_CRL
    rays = free_space_propagate(rays, focal_length_CRL)
    rays_stack[:, -2, 1:] = rays.T
    rays_stack[:, -2, 0] = x

    x = x + 3*focal_length_CRL
    rays = free_space_propagate(rays, 3*focal_length_CRL)
    rays_stack[:, -1, 1:] = rays.T
    rays_stack[:, -1, 0] = x

    ray_xyz_coord = np.zeros((number_of_rays, 3, rays_stack.shape[1]))
    ray_xyz_coord[:, 0, :] = rays_stack[:, :, 0]
    ray_xyz_coord[:, 2, :] = rays_stack[:, :, 1]

    #Rx = [[1,0,0],[0,c,-s],[0,s,c]]
    #ray_xyz_coord = Rx @ ray_xyz_coord

    # np.zeros((number_of_rays, number_of_lenses + 3, 3))
    ray_data[:, :, 0] = ray_xyz_coord[:, 0, :]
    ray_data[:, :, 1] = c[:, np.newaxis]*ray_xyz_coord[:, 1, :] - s[:, np.newaxis]*ray_xyz_coord[:, 2, :]
    ray_data[:, :, 2] = s[:, np.newaxis]*ray_xyz_coord[:, 1, :] + c[:, np.newaxis]*ray_xyz_coord[:, 2, :]

    ray_data_to_paraview(ray_data, fname='CRL_simulation.vtk')
    lens_to_paraview(number_of_lenses, lens_space, lens_radius, fname='CRL_lens.vtk')
    x = np.mean(ray_data[:,-2,0])
    backfocal_to_paraview(x, number_of_lenses, lens_space, lens_radius,  fname='backfocal.vtk')
    x = np.mean(ray_data[:,0,0])
    sample_plane_to_paraview(x, number_of_lenses, lens_space, lens_radius, fname='sample_plane.vtk')
    detector_plane_to_paraview(np.mean(ray_data[:,-1,0]), number_of_lenses, lens_space, lens_radius, fname='detector_plane.vtk')

    pr.disable()
    pr.dump_stats('tmp_profile_dump')
    ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
    ps.print_stats(15)
    print("")

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

    plt.figure()
    alpha0 = rays_stack[:, 0, 2]
    plt.scatter(y, z, c=np.abs(alpha0), alpha=0.5, cmap='jet')
    plt.colorbar()

    y = ray_data[:, -1, 1]*1e6
    z = ray_data[:, -1, 2]*1e6

    ymax = np.max(np.abs(y))*1.05
    zmax = np.max(np.abs(z))*1.05

    plt.figure()
    plt.title('Detector plane')
    plt.scatter(y, z, c=image_data, alpha=0.5, cmap='jet')
    plt.xlim([-ymax, ymax])
    plt.ylim([-zmax, zmax])
    plt.colorbar()

    y = ray_data[:, 0, 1]*1e6
    z = ray_data[:, 0, 2]*1e6

    plt.figure()
    plt.title('Sample plane')
    plt.scatter(y, z, c=image_data, alpha=0.5, cmap='jet')
    plt.xlim([-ymax, ymax])
    plt.ylim([-zmax, zmax])
    plt.colorbar()



    #plt.show()
