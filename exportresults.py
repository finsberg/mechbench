from vtk import *
import h5py
import os, errno

def __mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def __export_bench_format(basedir, basename, transform) :
    h5name = '{}.h5'.format(basename)
    vtname = '{}.vtu'.format(basename)

    with h5py.File(h5name, 'r') as f :
        dof = f['solution/vector_0'].shape[0]

    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtname)
    reader.Update()

    grid = reader.GetOutput()
    grid.GetCellData().RemoveArray("jacobian")

    # warped geometry
    grid.GetPointData().SetActiveVectors('displacement')
    warp = vtkWarpVector()
    warp.SetInputData(grid)
    warp.Update()

    warp.GetOutput().GetPointData().RemoveArray('displacement')
    grid.GetPointData().RemoveArray('displacement')

    if transform == 'axisym' :
        # rotate to have z as major axis
        trans = vtkTransform()
        trans.RotateY(90)
        tfilter1 = vtkTransformFilter()
        tfilter1.SetTransform(trans)
        tfilter1.SetInputData(grid)
        tfilter1.Update()
        # rotational extrusion
        geom = vtkGeometryFilter()
        geom.SetInputConnection(tfilter1.GetOutputPort())
        extr = vtkRotationalExtrusionFilter()
        extr.SetResolution(6)
        extr.SetAngle(90)
        extr.SetInputConnection(geom.GetOutputPort())
        extr.Update()
        ogrid = extr.GetOutput()
        writer = vtkPolyDataWriter()
        writer.SetInputData(extr.GetOutput())
        writer.SetFileName('{}/{}/deformed.vtk'.format(basedir, dof))
        writer.Write()
        #print "aaa"
        exit(0)
        trans = vtkTransform()
        trans.RotateY(90)
        tfilter1 = vtkTransformFilter()
        tfilter1.SetTransform(trans)
        tfilter1.SetInputData(grid)
        tfilter1.Update()
        ogrid = tfilter1.GetOutput()
        tfilter2 = vtkTransformFilter()
        tfilter2.SetTransform(trans)
        tfilter2.SetInputConnection(warp.GetOutputPort())
        tfilter2.Update()
        owarp = tfilter2
    elif transform == 'sym' :
        ogrid = grid
        owarp = warp

    # writer
    __mkdir_p('{}/{}'.format(basedir, dof))
    writer = vtkUnstructuredGridWriter()
    writer.SetInputConnection(owarp.GetOutputPort())
    writer.SetFileName('{}/{}/deformed.vtk'.format(basedir, dof))
    writer.Write()

    writer.SetInputData(ogrid)
    writer.SetFileName('{}/{}/undeformed.vtk'.format(basedir, dof))
    writer.Write()

if __name__ == "__main__" :
    try :
        os.makedirs('./results_out')
    except OSError :
        pass

    # problem 1
    #basedir = './results_out/problem 1'
    #__mkdir_p(basedir)
    #for ndiv in [ 1, 2, 4, 8, 16 ] :
    #    print("EXPORT PROBLEM 1 -- NDIV = {}".format(ndiv))
    #    basename = 'results/P1_unstruct_r01.0_n{:03d}_p2_q4'.format(ndiv)
    #    __export_bench_format(basedir, basename, transform='sym')

    # problem 2
    #basedir = './results_out/problem 2'
    #__mkdir_p(basedir)
    #for ndiv in [ 1, 2, 4, 8, 16, 32, 64 ] :
    #    print("EXPORT PROBLEM 2 -- NDIV = {}".format(ndiv))
    #    basename = 'results/P2_axisym_n{:03d}_p2_q6'.format(ndiv)
    #    __export_bench_format(basedir, basename, transform='axisym')

    # problem 3
    basedir = './results_out/problem 3'
    __mkdir_p(basedir)
    for ndiv in [ 1 ]: #, 2, 4, 8, 16, 32, 64 ] :
        print("EXPORT PROBLEM 3 -- NDIV = {}".format(ndiv))
        basename = 'results/P3_axisym_n{:03d}_p2_q6'.format(ndiv)
        __export_bench_format(basedir, basename, transform='axisym')

