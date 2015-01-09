from fenics import *
from ellipsoidgeometry import *
from guccionematerial import *
from lvproblem import *
from contsolver import *
from postprocess import *
from fenicshotools.vtkutils import *

import os

class Problem3(object):
    def __init__(self, comm, postprocess=True, **params):
        # parameters
        p = self.default_parameters()
        p.update(params)

        # global parameters
        parameters["form_compiler"]["representation"] = "uflacs"
        parameters["allow_extrapolation"] = True
        parameters['form_compiler']['quadrature_degree'] = p['quad']

        # filename
        gtype = "axisym" if p['axisymmetric'] else "full3d"
        fname = '{outdir}/pb3/{gtype}_n{ndiv:03d}_p{order}_q{quad}.h5'\
                .format(gtype=gtype, **p)
        if not os.path.isfile(fname) and postprocess:
            raise RuntimeError
        if os.path.isfile(fname) and not postprocess and MPI.rank(comm) == 0:
            os.remove(fname)
        MPI.barrier(comm)

        # quadrature order for all forms
        fspace = 'Quadrature_{:d}'.format(p['quad'])

        # geometry
        geoparam = EllipsoidGeometry.default_parameters()
        geoparam['axisymmetric'] = p['axisymmetric']
        geoparam['mesh_generation']['order'] = p['order']
        geoparam['mesh_generation']['ndiv'] = p['ndiv']
        geoparam['microstructure']['alpha_endo'] = +90.0
        geoparam['microstructure']['alpha_epi']  = -90.0
        geoparam['microstructure']['function_space'] = fspace
        geo = EllipsoidGeometry(comm, fname, **geoparam)

        # material
        matparam = GuccioneMaterial.default_parameters()
        matparam['Tactive'] = 0.0
        matparam['C'] = 2.0
        matparam['bf'] = 8.0
        matparam['bt'] = 2.0
        matparam['bfs'] = 4.0
        matparam['e1'] = geo.f0
        matparam['e2'] = geo.s0
        matparam['e3'] = geo.n0
        mat = GuccioneMaterial(**matparam)

        self.problem = LVProblem(geo, mat)
        self.fname = fname
        self.comm  = comm
        self.postprocess = postprocess

    @staticmethod
    def default_parameters():
        p = { 'axisymmetric' : True,
              'ndiv' : 1,
              'order' : 1,
              'quad' : 4,
              'outdir' : './results' }
        return p

    def run(self):
        pb = self.problem
        # load solution
        if self.postprocess:
            with HDF5File(self.comm, self.fname, 'r') as f:
                f.read(pb.state, '/solution')
        else:
            # solve
            solver = ContinuationSolver(self.problem, symmetric=True,
                                        backend='mumps')
            solver.solve(pendo=15.0, Tactive=60.0, step=0.25)

            # save full solution
            with HDF5File(self.comm, self.fname, 'a') as f:
                f.write(self.problem.state, '/solution')

        # postprocess if in serial
        if MPI.size(self.comm) == 1:
            vname = os.path.splitext(self.fname)[0] + ".vtu"
            domain, u, E = compute_postprocessed_quantities(pb)
            grid = dolfin2vtk(domain, u.function_space())
            vtk_add_field(grid, E)
            vtk_add_field(grid, u)
            vtk_write_file(grid, vname)

if __name__ == "__main__":
    comm = mpi_comm_world()
    if MPI.rank(comm) == 0:
        set_log_level(PROGRESS)
    else :
        set_log_level(ERROR)

    post = False
    pb = Problem3(comm, post, ndiv=1, order=2, quad=6, axisymmetric=True)
    pb.run()

    info("vol = {}".format(pb.problem.get_Vendo()))
    info("dof = {}".format(pb.problem.state.function_space().dim()))

