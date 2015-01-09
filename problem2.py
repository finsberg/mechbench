from fenics import *
from ellipsoidgeometry import EllipsoidGeometry
from guccionematerial import GuccioneMaterial
from lvproblem import LVProblem
from contsolver import ContinuationSolver
from postprocess import *
from fenicshotools.vtkutils import *

import os

class Problem2(object) :
    def __init__(self, comm, postprocess=True, **params) :
        # parameters
        p = self.default_parameters()
        p.update(params)

        # global parameters
        parameters["form_compiler"]["representation"] = "uflacs"
        parameters["allow_extrapolation"] = True
        parameters['form_compiler']['quadrature_degree'] = p['quad']

        # filename
        gtype = "axisym" if p['axisymmetric'] else "full3d"
        fname = '{outdir}/pb2/{gtype}_n{ndiv:03d}_p{order}_q{quad}.h5'\
                .format(gtype=gtype, **p)
        if not os.path.isfile(fname) and postprocess :
            postprocess = False
        if os.path.isfile(fname) and not postprocess and MPI.rank(comm) == 0 :
            os.remove(fname)
        MPI.barrier(comm)

        # geometry
        geoparam = EllipsoidGeometry.default_parameters()
        geoparam['axisymmetric'] = p['axisymmetric']
        geoparam['mesh_generation']['order'] = p['order']
        geoparam['mesh_generation']['ndiv'] = p['ndiv']
        geoparam['microstructure']['function_space'] = ''
        geo = EllipsoidGeometry(comm, fname, **geoparam)

        # material
        matparam = GuccioneMaterial.default_parameters()
        matparam['Tactive'] = None
        matparam['C'] = 10.0
        matparam['bf'] = 1.0
        matparam['bt'] = 1.0
        matparam['bfs'] = 1.0
        mat = GuccioneMaterial(**matparam)

        self.problem = LVProblem(geo, mat)
        self.fname = fname
        self.comm  = comm
        self.postprocess = postprocess

    @staticmethod
    def default_parameters() :
        p = { 'axisymmetric' : True,
              'ndiv' : 1,
              'order' : 1,
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
            solver = ContinuationSolver(self.problem, symmetric=True,
                                        backend='mumps')
            solver.solve(pendo=10.0, step=0.25)

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
    pb = Problem2(comm, post, ndiv=1, order=2, quad=6, axisymmetric=True)
    pb.run()

    info("vol = {}".format(pb.problem.get_Vendo()))
    info("dof = {}".format(pb.problem.state.function_space().dim()))

