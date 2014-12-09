from fenics import *
from ellipsoidgeometry import EllipsoidGeometry
from guccionematerial import GuccioneMaterial
from lvproblem import LVProblem
from contsolver import ContinuationSolver

import os

parameters["form_compiler"]["representation"] = "uflacs"
parameters["allow_extrapolation"] = True

class Problem2(object) :
    def __init__(self, comm, postprocess=True, **params) :
        # parameters
        p = self.default_parameters()
        p.update(params)

        # filename
        gtype = "axisym" if p['axisymmetric'] else "full3d"
        fname = '{outdir}/P2_{gtype}_n{ndiv:03d}_p{order}_q{quad}.h5'\
                .format(gtype=gtype, **p)
        if not os.path.isfile(fname) and postprocess :
            raise RuntimeError
        if os.path.isfile(fname) and not postprocess and MPI.rank(comm) == 0 :
            os.remove(fname)
        MPI.barrier(comm)

        # quadrature order for all forms
        parameters['form_compiler']['quadrature_degree'] = p['quad']

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

        # load solution
        if postprocess :
            with HDF5File(self.comm, self.fname, 'r') as f :
                f.read(self.problem.state, '/solution')

    @staticmethod
    def default_parameters() :
        p = { 'axisymmetric' : True,
              'ndiv' : 1,
              'order' : 1,
              'quad' : 4,
              'outdir' : './results' }
        return p

    def run(self) :
        if self.postprocess :
            raise RuntimeError

        # solve
        solver = ContinuationSolver(self.problem)
        solver.solve(pendo=10.0, step=0.25)

        # save full solution
        with HDF5File(self.comm, self.fname, 'a') as f :
            f.write(self.problem.state, '/solution')

