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

def __export_bench_bar(basedir, basename):
    """
    Symmetrization of the mesh wrt the plane x1=x1_max.
    """
    h5name = '{}.h5'.format(basename)
    vtname = '{}.vtu'.format(basename)

    with h5py.File(h5name, 'r') as f:
        dof = f['solution/vector_0'].shape[0]
    print("DOF = {}".format(dof))

    # read the solution
    reader = tvtk.XMLUnstructuredGridReader(file_name=vtname)
    reader.update()
    grid = reader.get_output()
    X = grid.points.to_array()
    ncells = grid.number_of_cells
    npts = X.shape[0]
    elm = grid.get_cells().to_array().reshape((ncells,-1))[:,1:].astype(np.intc)
    isquad = grid.get_cell_type(0) == tvtk.QuadraticTetra().cell_type

    # fields
    E = grid.point_data.get_array('strain').to_array()
    u = grid.point_data.get_array('displacement').to_array()

    # deformed geometry
    x = X + u

    # first we need points ids not on the symmetry plane
    ptssym  = np.where(np.abs(X[:,1]-max(X[:,1])) <= 1e-16)[0]
    ptsnsym = np.where(np.abs(X[:,1]-max(X[:,1]))  > 1e-16)[0]
    # new points are organized so that we first have
    # original points not on the symmetry plane, then those
    # on the symmetry plane, and eventually the symmetrized points.
    perm = dict(zip(np.hstack([ptsnsym, ptssym]), np.arange(npts)))
    elm = np.vectorize(perm.get, otypes=[np.intc])(elm)
    X = X[np.hstack([ptsnsym, ptssym]),:]
    x = x[np.hstack([ptsnsym, ptssym]),:]
    E = E[np.hstack([ptsnsym, ptssym]),:]

    Xsym = X[0:len(ptsnsym),:].copy()
    Xsym[:,1] = 2.0*max(X[:,1])-Xsym[:,1]
    Xnew = np.vstack([X, Xsym])

    xsym = x[0:len(ptsnsym),:].copy()
    xsym[:,1] = 2.0*max(x[:,1])-xsym[:,1]
    xnew = np.vstack([x, xsym])

    # new elements
    elmsym = elm.copy() + npts
    # shared points needs to be restored
    sharedids = elmsym >= Xnew.shape[0]
    elmsym[sharedids] = elm[sharedids]
    # correct permutation of dofs
    elmsym = elmsym[:,[0,2,1,3,6,5,4,7,9,8]]
    # full list of elements
    elmnew = np.vstack([elm, elmsym])

    # strain field
    Enew = np.vstack([E, E[0:len(ptsnsym),:]])

    ug = tvtk.UnstructuredGrid(points=Xnew)
    ug.set_cells(tvtk.QuadraticTetra().cell_type, elmnew)
    ug.point_data.tensors = Enew
    ug.point_data.tensors.name = 'E'

    ugdef = tvtk.UnstructuredGrid(points=xnew)
    ugdef.set_cells(tvtk.QuadraticTetra().cell_type, elmnew)
    ugdef.point_data.tensors = Enew
    ugdef.point_data.tensors.name = 'E'

    __mkdir_p('{}/{}'.format(basedir, dof))
    write_data(ug, '{}/{}/undeformed.vtk'.format(basedir, dof))
    write_data(ugdef, '{}/{}/deformed.vtk'.format(basedir, dof))


def __export_bench_axisym(basedir, basename, nsectors=8):
    h5name = '{}.h5'.format(basename)
    vtname = '{}.vtu'.format(basename)

    with h5py.File(h5name, 'r') as f:
        dof = f['solution/vector_0'].shape[0]
    print("DOF = {}".format(dof))

    # read the axisymmetric solution
    reader = tvtk.XMLUnstructuredGridReader(file_name=vtname)
    reader.update()
    grid = reader.get_output()
    X = grid.points.to_array()
    ncells = grid.number_of_cells
    elm = grid.get_cells().to_array().reshape((ncells,-1))[:,1:].astype(np.intc)
    isquad = grid.get_cell_type(0) == tvtk.QuadraticTriangle().cell_type

    # fields
    E = grid.point_data.get_array('strain').to_array()
    u = grid.point_data.get_array('displacement').to_array()
    E = E.reshape((-1,3,3))

    # deformed pts
    x = X + u

    # rotational extrusion
    nslices = 2*nsectors if isquad else nsectors
    npts = X.shape[0]
    # points along the rotation axis require special treatment
    ptsdeg  = np.where(np.abs(X[:,1])  < 1e-16)[0]
    ptsndeg = np.where(np.abs(X[:,1]) >= 1e-16)[0]
    # reorder the nodes to have degenerate vertices at the bottom
    perm = dict(zip(np.hstack([ptsndeg, ptsdeg]), np.arange(npts)))
    elm = np.vectorize(perm.get, otypes=[np.intc])(elm)
    X = X[np.hstack([ptsndeg, ptsdeg]),:]
    x = x[np.hstack([ptsndeg, ptsdeg]),:]
    E = E[np.hstack([ptsndeg, ptsdeg]),:,:]

    Xrot = np.zeros((npts+len(ptsndeg)*(nslices-1),3))
    xrot = np.zeros((npts+len(ptsndeg)*(nslices-1),3))
    Erot = np.zeros((npts+len(ptsndeg)*(nslices-1),3,3))
    # first the initial slice (with degenerate points)
    theta = 0.0
    R = np.array([[ +0.0, +np.cos(theta), -np.sin(theta) ],
                  [ +0.0, -np.sin(theta), -np.cos(theta) ],
                  [ -1.0,            0.0,            0.0 ]])
    Xrot[0:npts,:] = np.dot(X, R.T)
    xrot[0:npts,:] = np.dot(x, R.T)
    Erot[0:npts,:,:] = np.swapaxes(np.dot(R, np.dot(E, R.T)), 0, 1)
    # rotational extrusion
    for i in xrange(1, nslices):
        # R = Ry(pi/2) Rx(pi/2) Rx(theta)
        theta = 2.0*i*np.pi/nslices
        R = np.array([[ +0.0, +np.cos(theta), -np.sin(theta) ],
                      [ +0.0, -np.sin(theta), -np.cos(theta) ],
                      [ -1.0,            0.0,            0.0 ]])
        off1 = npts + len(ptsndeg)*(i-1)
        off2 = npts + len(ptsndeg)*i
        Xrot[off1:off2,:] = np.dot(X[0:len(ptsndeg),:], R.T)
        xrot[off1:off2,:] = np.dot(x[0:len(ptsndeg),:], R.T)
        # strain
        Erot[off1:off2,:,:] = np.swapaxes(np.dot(R, np.dot(E[0:len(ptsndeg),:,:], R.T)), 0, 1)

    # elements, 3 categories:
    # - 0 points on the axis ~> wedge
    # - 1 points on the axis ~> pyramid
    # - 2 points on the axis ~> tetrahedron
    elm0, elm1, elm2 = [], [], []
    for e in elm:
        ncount = np.count_nonzero(e[0:3] >= len(ptsndeg))
        { 0: elm0, 1: elm1, 2: elm2 }[ncount].append(e)
    elm0 = np.array(elm0)
    elm1 = np.array(elm1)
    elm2 = np.array(elm2)

    # add normal elements
    elm0rot = []
    elm1rot = []
    elm2rot = []
    for i in xrange(nsectors):
        if isquad :
            # back face
            off0 = npts + len(ptsndeg)*(2*i-1) if i>0 else 0
            # mid face
            off1 = npts + len(ptsndeg)*(2*i) if i>0 else npts
            # front face
            off2 = off1 + len(ptsndeg) if i<nsectors-1 else 0
            # wedge elements
            enew = np.column_stack([elm0[:,[0,1,2]]+off0,
                                    elm0[:,[0,1,2]]+off2,
                                    elm0[:,[3,4,5]]+off0,
                                    elm0[:,[3,4,5]]+off2,
                                    elm0[:,[0,1,2]]+off1])
            elm0rot += [ enew ]

            # pyramid elements
            for e in elm1 :
                # this is the point on the axis
                idx = np.where(e[0:3] >= len(ptsndeg))[0][0]
                # new pyramid
                e1, e2, e3 = e[idx], e[(idx+1)%3], e[(idx+2)%3]
                e4, e5, e6 = e[idx+3], e[(idx+1)%3+3], e[(idx+2)%3+3]
                enew = [ e2+off0, e3+off0, e3+off2, e2+off2, e1,
                         e5+off0, e3+off1, e5+off2, e2+off1,
                         e4+off0, e6+off0, e6+off2, e4+off2 ]
                elm1rot += [ enew ]

            # tetrahedral elements
            for e in elm2 :
                # this is the point off the axis
                idx = np.where(e[0:3] < len(ptsndeg))[0][0]
                # new tetrahedron
                e1, e2, e3 = e[idx], e[(idx+1)%3], e[(idx+2)%3]
                e4, e5, e6 = e[idx+3], e[(idx+1)%3+3], e[(idx+2)%3+3]
                enew = [ e1+off0, e2, e3, e1+off2,
                         e4+off0, e5, e6+off0,
                         e1+off1, e4+off2, e6+off2 ]
                elm2rot += [ enew ]
        else:
            raise NotImplementedError

    elm0rot = np.vstack(elm0rot)
    elm1rot = np.vstack(elm1rot)
    elm2rot = np.vstack(elm2rot)

    elm0rot = np.hstack([15*np.ones((elm0rot.shape[0],1),dtype=np.intc), elm0rot])
    elm1rot = np.hstack([13*np.ones((elm1rot.shape[0],1),dtype=np.intc), elm1rot])
    elm2rot = np.hstack([10*np.ones((elm2rot.shape[0],1),dtype=np.intc), elm2rot])
    elmrot = np.hstack([elm0rot.ravel(), elm1rot.ravel(), elm2rot.ravel()])

    elmtypes  = [tvtk.QuadraticWedge().cell_type]*elm0rot.shape[0]
    elmtypes += [tvtk.QuadraticPyramid().cell_type]*elm1rot.shape[0]
    elmtypes += [tvtk.QuadraticTetra().cell_type]*elm2rot.shape[0]
    elmtypes = np.array(elmtypes)

    elmoff = np.arange(0,15*elm0rot.shape[0],15)
    elmoff = np.hstack([elmoff, np.arange(13,13*elm1rot.shape[0],13) + elmoff[-1]])
    elmoff = np.hstack([elmoff, np.arange(10,10*elm2rot.shape[0],10) + elmoff[-1]])

    cells = tvtk.CellArray()
    cells.set_cells(elm0rot.shape[0]+elm1rot.shape[0]+elm2rot.shape[0], elmrot)

    ug = tvtk.UnstructuredGrid(points=Xrot)
    ug.set_cells(elmtypes, elmoff, cells)
    ug.point_data.tensors = Erot.reshape((-1,9))
    ug.point_data.tensors.name = 'E'

    ugdef = tvtk.UnstructuredGrid(points=xrot)
    ugdef.set_cells(elmtypes, elmoff, cells)
    ugdef.point_data.tensors = Erot.reshape((-1,9))
    ugdef.point_data.tensors.name = 'E'

    # writer
    __mkdir_p('{}/{}'.format(basedir, dof))
    write_data(ug, '{}/{}/undeformed.vtk'.format(basedir, dof))
    write_data(ugdef, '{}/{}/deformed.vtk'.format(basedir, dof))


if __name__ == "__main__":
    try :
        os.makedirs('./results_out')
    except OSError :
        pass

    # problem 1
    basedir = './results_out/problem 1'
    __mkdir_p(basedir)
    for ndiv in [1, 2, 4, 8]:
        print("EXPORT PROBLEM 1 -- NDIV = {}".format(ndiv))
        basename = 'results/pb1/struct_n{:03d}_p2_r16.0_q6'.format(ndiv)
        __export_bench_bar(basedir, basename)

    # problem 2
    basedir = './results_out/problem 2'
    __mkdir_p(basedir)
    for ndiv in [1, 2, 4, 8]:
        print("EXPORT PROBLEM 2 -- NDIV = {}".format(ndiv))
        basename = 'results/pb2/axisym_n{:03d}_p2_q6'.format(ndiv)
        __export_bench_axisym(basedir, basename)

    # problem 3
    basedir = './results_out/problem 3'
    __mkdir_p(basedir)
    for ndiv in [1, 2, 4, 8]:
        print("EXPORT PROBLEM 3 -- NDIV = {}".format(ndiv))
        basename = 'results/pb3/axisym_n{:03d}_p2_q6'.format(ndiv)
        __export_bench_axisym(basedir, basename)

