from fenics import *
from bargeometry import BarGeometry
from guccionematerial import GuccioneMaterial
from barproblem import BarProblem
from contsolver import ContinuationSolver
from postprocess import *
from fenicshotools.vtkutils import *

import os

class Problem1(object) :
    def __init__(self, comm, postprocess=True, **params) :
        # parameters
        p = self.default_parameters()
        p.update(params)

        # global parameters
        parameters["form_compiler"]["representation"] = "uflacs"
        parameters["allow_extrapolation"] = True
        parameters['form_compiler']['quadrature_degree'] = p['quad']

        # filename
        gtype = ("" if p['structured'] else "un") + "struct"
        fname = '{outdir}/pb1/{gtype}_n{ndiv:03d}_p{order}'\
                '_r{reduction_factor:04.1f}_q{quad}.h5'\
                .format(gtype=gtype, **p)
        if not os.path.isfile(fname) and postprocess :
            postprocess = False
        if os.path.isfile(fname) and not postprocess and MPI.rank(comm) == 0 :
            os.remove(fname)
        MPI.barrier(comm)

        # geometry
        geoparam = BarGeometry.default_parameters()
        geoparam['mesh_generation']['structured'] = p['structured']
        geoparam['mesh_generation']['reduction_factor'] = p['reduction_factor']
        geoparam['mesh_generation']['ndiv'] = p['ndiv']
        geo = BarGeometry(comm, fname, **geoparam)

        # material
        mat = GuccioneMaterial(e1=geo.f0, e2=geo.s0, e3=geo.n0,
                               C=2.0, bf=8, bt=2, bfs=4)

        self.problem = BarProblem(geo, mat, order=p['order'])
        self.fname = fname
        self.comm  = comm
        self.postprocess = postprocess

    @staticmethod
    def default_parameters() :
        p = { 'structured' : False,
              'reduction_factor' : 4.0,
              'ndiv' : 1,
              'order' : 2,
              'quad' : 4,
              'outdir' : './results' }
        return p

    def run(self) :
        pb = self.problem
        # load solution
        if self.postprocess :
            with HDF5File(self.comm, self.fname, 'r') as f :
                f.read(pb.state, '/solution')
        else :
            # solve
            solver = ContinuationSolver(self.problem, symmetric=False,
                                        backend='mumps')
            solver.solve(pbottom=0.004, step=0.004)

            # save full solution
            with HDF5File(self.comm, self.fname, 'a') as f :
                f.write(self.problem.state, '/solution')

        # postprocess if in serial
        if MPI.size(self.comm) == 1 :
            vname = os.path.splitext(self.fname)[0] + ".vtu"
            domain, u, E = compute_postprocessed_quantities(pb)
            grid = dolfin2vtk(domain, u.function_space())
            vtk_add_field(grid, E)
            vtk_add_field(grid, u)
            vtk_write_file(grid, vname)

if __name__ == "__main__" :
    comm = mpi_comm_world()
    if MPI.rank(comm) == 0 :
        set_log_level(PROGRESS)
    else :
        set_log_level(ERROR)

    post = False
    pb = Problem1(comm, post, ndiv=1, order=2, quad=6,
                  reduction_factor=16.0, structured=True)
    pb.run()

    info("pos = {}".format(pb.problem.get_point_position()))
    info("dof = {}".format(pb.problem.state.function_space().dim()))

