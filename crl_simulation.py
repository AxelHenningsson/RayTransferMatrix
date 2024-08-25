import numpy as np
import matplotlib.pyplot as plt
import vtk

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
    t = number_of_lenses*lens_space / 10.
    w = 0.05
    h = 0.05

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
    lens_space = 1.6 * 1e-3 # T
    lens_radius = 50 * 1e-6 # R
    refractive_decrement = 2.359 * 1e-6  # delta
    focal_length = lens_radius / ( 2 * refractive_decrement) # f

    number_of_rays = 1500
    ray_data = np.zeros((number_of_rays, number_of_lenses + 3, 3))

    # equation (4), simons 2016
    phi = np.sqrt( lens_space / focal_length ) # phi
    focal_length_CRL = focal_length * phi * ( 1. / np.tan(number_of_lenses*phi) )
    start_x = -focal_length_CRL

    alpha0 = np.zeros((ray_data.shape[0], ))

    #xg = yg = np.linspace(-lens_radius,lens_radius,128)*0.5
    #X,Y =np.meshgrid(yg,zg,indexing='ij')

    for j in range(number_of_rays):
        r = 0.4*(np.random.rand()-0.5)*2*lens_radius # radius from optical axis

        angle = np.random.rand()*2*np.pi
        s,c = np.sin(angle), np.cos(angle)
        Rx = np.array([[1,0,0],[0,c,-s],[0,s,c]])

        alpha = 0.02 * (np.random.rand()-0.5)*2 * 1e-3
        alpha0[j] = alpha
        #print(alpha)
        #raise
        ray0 = np.array([r, alpha])

        ray = ray0.copy()

        rays_stack = np.zeros((number_of_lenses + 3, 3))
        states = np.array(range(number_of_lenses + 3)).astype(int)
        x = start_x
        rays_stack[0, 1:] = ray
        rays_stack[0, 0] = x
        ray = free_space_propagate(ray, np.abs(start_x))
        #print(start_x)
        #raise
        x = x + np.abs(start_x)
        for i in states[1:]:
            ray = free_space_propagate(ray, lens_space / 2.)
            ray = thin_lens_propagate(ray, focal_length)
            ray = free_space_propagate(ray, lens_space / 2.)
            rays_stack[i, 1:] = ray
            x = x + lens_space
            rays_stack[i, 0] = x

        CRL_focal_distance = focal_length_CRL - (lens_space / 2.)#- ray[0] / ray[1]
        x = x + CRL_focal_distance
        ray = free_space_propagate(ray, CRL_focal_distance)
        rays_stack[-2, 1:] = ray
        rays_stack[-2, 0] = x

        x = x + 3*CRL_focal_distance
        ray = free_space_propagate(ray, 3*CRL_focal_distance)
        rays_stack[-1, 1:] = ray
        rays_stack[-1, 0] = x

        ray_xyz_coord = np.zeros((3, rays_stack.shape[0]))
        ray_xyz_coord[0,:] = rays_stack[:, 0 ]
        ray_xyz_coord[2,:] = rays_stack[:, 1 ]
        ray_xyz_coord = Rx @ ray_xyz_coord

        ray_data[j, :, :] = ray_xyz_coord[:, :].T

    ray_data_to_paraview(ray_data, fname='CRL_simulation.vtk')
    lens_to_paraview(number_of_lenses, lens_space, lens_radius, fname='CRL_lens.vtk')
    x = np.mean(ray_data[:,-2,0])
    backfocal_to_paraview(x, number_of_lenses, lens_space, lens_radius,  fname='backfocal.vtk')
    x = np.mean(ray_data[:,0,0])
    sample_plane_to_paraview(x, number_of_lenses, lens_space, lens_radius, fname='sample_plane.vtk')
    detector_plane_to_paraview(np.mean(ray_data[:,-1,0]), number_of_lenses, lens_space, lens_radius, fname='detector_plane.vtk')

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
    plt.scatter(y, z, c=np.abs(alpha0), alpha=0.5, cmap='jet')
    plt.colorbar()
    #plt.show()
