from tvtk.api import tvtk
from tvtk.array_handler import *
from tvtk.api import write_data
from vtk import *
import numpy as np
import h5py
import os, errno

from pprint import pprint

def __mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def __export_bench_bar(basedir, basename) :
    h5name = '{}.h5'.format(basename)
    vtname = '{}.vtu'.format(basename)

    with h5py.File(h5name, 'r') as f :
        dof = f['solution/vector_0'].shape[0]
    print("DOF = {}".format(dof))

    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtname)
    reader.Update()

    grid = reader.GetOutput()
    grid.GetCellData().RemoveArray("jacobian")

    # reflection
    refl = vtkReflectionFilter()
    refl.SetPlaneToYMax()
    refl.SetInputData(grid)
    refl.Update()

    # warped geometry
    refl.GetOutput().GetPointData().SetActiveVectors('displacement')
    warp = vtkWarpVector()
    warp.SetInputConnection(refl.GetOutputPort())
    warp.Update()

    warp.GetOutput().GetPointData().RemoveArray('displacement')
    refl.GetOutput().GetPointData().RemoveArray('displacement')

    # writer
    __mkdir_p('{}/{}'.format(basedir, dof))
    writer = vtkUnstructuredGridWriter()
    writer.SetInputConnection(warp.GetOutputPort())
    writer.SetFileName('{}/{}/deformed.vtk'.format(basedir, dof))
    writer.Write()

    writer.SetInputConnection(refl.GetOutputPort())
    writer.SetFileName('{}/{}/undeformed.vtk'.format(basedir, dof))
    writer.Write()

def __export_bench_axisym(basedir, basename, nslices=64) :
    h5name = '{}.h5'.format(basename)
    vtname = '{}.vtu'.format(basename)

    with h5py.File(h5name, 'r') as f :
        dof = f['solution/vector_0'].shape[0]
    print("DOF = {}".format(dof))

    # read the axisymmetric solution
    reader = tvtk.XMLUnstructuredGridReader(file_name=vtname)
    reader.update()
    grid = reader.get_output()
    X = grid.points.to_array()
    elm = grid.get_cells().to_array().reshape((-1,4))[:,1:].astype(np.intc)

    # fields
    E = grid.cell_data.get_array('strain').to_array()
    u = grid.point_data.get_array('displacement').to_array()
    Exx, Exy, Exz = E[:,0], E[:,1], E[:,2]
    Eyy, Eyz, Ezz = E[:,4], E[:,5], E[:,8]

    # deformed pts
    x = X + u

    # rotational extrusion
    # Xnew = Ry(pi/2) Rx(pi/2) Rx(theta)
    # NOTE: we don't care about degenerate wedges
    Xnew = [np.column_stack([X[:,1], -X[:,2], -X[:,0]])]
    xnew = [np.column_stack([x[:,1], -x[:,2], -x[:,0]])]
    elmnew = []
    Exxnew, Exynew, Exznew = [], [], []
    Eyynew, Eyznew, Ezznew = [], [], []
    for i in xrange(1, nslices) :
        theta = 2.0*i/nslices * np.pi
        off1 = (i-1)*len(X)
        off2 = i*len(X)
        X1rot = +X[:,1]*np.cos(theta) - X[:,2]*np.sin(theta)
        X2rot = -X[:,1]*np.sin(theta) - X[:,2]*np.cos(theta)
        X3rot = -X[:,0]
        x1rot = +x[:,1]*np.cos(theta) - x[:,2]*np.sin(theta)
        x2rot = -x[:,1]*np.sin(theta) - x[:,2]*np.cos(theta)
        x3rot = -x[:,0]
        # new points
        Xnew += [np.column_stack([X1rot, X2rot, X3rot])]
        xnew += [np.column_stack([x1rot, x2rot, x3rot])]
        elmnew += [np.column_stack([elm + off1, elm + off2])]
        # strain (evaluated in the middle point)
        tmid = 2.0*i/nslices * np.pi + 1.0/nslices * np.pi
        Exxnew += [Ezz*np.sin(tmid)**2 + Eyy*(1-np.sin(tmid)**2)]
        Exynew += [-0.5*np.sin(2*tmid)*(Eyy-Ezz)]
        Exznew += [-Exy*np.cos(tmid)]
        Eyynew += [Eyy*np.sin(tmid)**2 + Ezz*(1-np.sin(tmid)**2)]
        Eyznew += [Exy*np.sin(tmid)]
        Ezznew += [Exx]

    # final slice
    tmid = 1.0/nslices * np.pi
    elmnew += [np.column_stack([elm + off2, elm])]
    Exxnew += [Ezz*np.sin(tmid)**2 + Eyy*(1-np.sin(tmid)**2)]
    Exynew += [-0.5*np.sin(2*tmid)*(Eyy-Ezz)]
    Exznew += [-Exy*np.cos(tmid)]
    Eyynew += [Eyy*np.sin(tmid)**2 + Ezz*(1-np.sin(tmid)**2)]
    Eyznew += [Exy*np.sin(tmid)]
    Ezznew += [Exx]

    # to numpy array
    Xnew = np.vstack(Xnew)
    xnew = np.vstack(xnew)
    elmnew = np.vstack(elmnew)
    Exxnew = np.hstack(Exxnew)
    Exynew = np.hstack(Exynew)
    Exznew = np.hstack(Exznew)
    Eyynew = np.hstack(Eyynew)
    Eyznew = np.hstack(Eyznew)
    Ezznew = np.hstack(Ezznew)
    Enew = np.vstack([Exxnew, Exynew, Exznew,
                      Exynew, Eyynew, Eyznew,
                      Exznew, Eyznew, Ezznew]).transpose()

    ug = tvtk.UnstructuredGrid(points=Xnew)
    ug.set_cells(tvtk.Wedge().cell_type, elmnew.astype('f'))
    ug.cell_data.tensors = Enew
    ug.cell_data.tensors.name = 'E'

    ugdef = tvtk.UnstructuredGrid(points=xnew)
    ugdef.set_cells(tvtk.Wedge().cell_type, elmnew.astype('f'))
    ugdef.cell_data.tensors = Enew
    ugdef.cell_data.tensors.name = 'E'

    # writer
    __mkdir_p('{}/{}'.format(basedir, dof))
    write_data(ug, '{}/{}/undeformed.vtk'.format(basedir, dof))
    write_data(ugdef, '{}/{}/deformed.vtk'.format(basedir, dof))

if __name__ == "__main__" :
    try :
        os.makedirs('./results_out')
    except OSError :
        pass

    # problem 1
    basedir = './results_out/problem 1'
    __mkdir_p(basedir)
    for ndiv in [ 1, 2, 4, 8 ] :
        print("EXPORT PROBLEM 1 -- NDIV = {}".format(ndiv))
        basename = 'results/P1_unstruct_r01.0_n{:03d}_p2_q4'.format(ndiv)
        __export_bench_bar(basedir, basename)

    # problem 2
    basedir = './results_out/problem 2'
    __mkdir_p(basedir)
    for ndiv in [ 1, 2, 4, 8 ] :
        print("EXPORT PROBLEM 2 -- NDIV = {}".format(ndiv))
        basename = 'results/P2_axisym_n{:03d}_p2_q6'.format(ndiv)
        __export_bench_axisym(basedir, basename)

    # problem 3
    basedir = './results_out/problem 3'
    __mkdir_p(basedir)
    for ndiv in [ 1, 2, 4, 8 ] :
        print("EXPORT PROBLEM 3 -- NDIV = {}".format(ndiv))
        basename = 'results/P3_axisym_n{:03d}_p2_q6'.format(ndiv)
        __export_bench_axisym(basedir, basename)

