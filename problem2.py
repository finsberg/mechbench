from fenics import *
from ellipsoidgeometry import EllipsoidGeometry

# from guccionematerial import GuccioneMaterial
from guccione import Guccione as GuccioneMaterial
from lvproblem import LVProblem
from contsolver import ContinuationSolver
from postprocess import *

# from fenicshotools.vtkutils import *

import os


class Problem2(object):
    def __init__(self, comm, postprocess=True, **params):
        # parameters
        p = self.default_parameters()
        p.update(params)

        # global parameters
        parameters["form_compiler"]["representation"] = "uflacs"
        parameters["allow_extrapolation"] = True
        parameters["form_compiler"]["quadrature_degree"] = p["quad"]

        # filename
        gtype = "axisym" if p["axisymmetric"] else "full3d"
        fname = "{outdir}/pb2/{gtype}_n{ndiv:03d}_p{order}_q{quad}.h5".format(
            gtype=gtype, **p
        )
        if not os.path.isfile(fname) and postprocess:
            postprocess = False
        if os.path.isfile(fname) and not postprocess and MPI.rank(comm) == 0:
            os.remove(fname)
        MPI.barrier(comm)

        # geometry
        geoparam = EllipsoidGeometry.default_parameters()
        geoparam["axisymmetric"] = p["axisymmetric"]
        geoparam["mesh_generation"]["order"] = p["order"]
        geoparam["mesh_generation"]["ndiv"] = p["ndiv"]
        geoparam["microstructure"]["function_space"] = ""
        geo = EllipsoidGeometry(comm, fname, **geoparam)

        # material
        matparam = GuccioneMaterial.default_parameters()
        matparam["Tactive"] = None
        matparam["C"] = 10.0
        matparam["bf"] = 1.0
        matparam["bt"] = 1.0
        matparam["bfs"] = 1.0
        mat = GuccioneMaterial(**matparam)

        self.problem = LVProblem(geo, mat)
        self.fname = fname
        self.comm = comm
        self.postprocess = postprocess

    @staticmethod
    def default_parameters():
        p = {
            "axisymmetric": True,
            "ndiv": 1,
            "order": 1,
            "quad": 4,
            "outdir": "./results",
        }
        return p

    def run(self):
        pb = self.problem
        # load solution
        if self.postprocess:
            with HDF5File(self.comm, self.fname, "r") as f:
                f.read(pb.state, "/solution")
        else:
            # solve
            solver = ContinuationSolver(self.problem, symmetric=True, backend="mumps")
            solver.solve(pendo=10.0, step=0.25)

            # save full solution
            # with HDF5File(self.comm, self.fname, "w") as f:
            #     f.write(self.problem.state, "/solution")

        # breakpoint()
        u = pb.state.split()[0]
        geometry = pb.geo
        endo_apex_marker = geometry.markers["ENDOPT"][0]
        endo_apex_idx = geometry.vfun.array().tolist().index(endo_apex_marker)
        endo_apex = geometry.mesh.coordinates()[endo_apex_idx, :]
        endo_apex_pos = endo_apex + u(endo_apex)

        print(
            ("\n\nGet longitudinal position of endocardial apex: {:.4f} mm" "").format(
                endo_apex_pos[0],
            ),
        )

        epi_apex_marker = geometry.markers["EPIPT"][0]
        epi_apex_idx = geometry.vfun.array().tolist().index(epi_apex_marker)
        epi_apex = geometry.mesh.coordinates()[epi_apex_idx, :]
        epi_apex_pos = epi_apex + u(epi_apex)

        print(
            ("\n\nGet longitudinal position of epicardial apex: {:.4f} mm" "").format(
                epi_apex_pos[0],
            ),
        )
        # postprocess if in serial
        # if MPI.size(self.comm) == 1:
        #     vname = os.path.splitext(self.fname)[0] + ".vtu"
        #     domain, u, E = compute_postprocessed_quantities(pb)
        #     grid = dolfin2vtk(domain, u.function_space())
        #     vtk_add_field(grid, E)
        #     vtk_add_field(grid, u)
        #     vtk_write_file(grid, vname)


if __name__ == "__main__":
    comm = MPI.comm_world
    if MPI.rank(comm) == 0:
        set_log_level(20)
    else:
        set_log_level(30)

    post = False
    pb = Problem2(comm, post, ndiv=1, order=2, quad=6, axisymmetric=True)
    pb.run()

    info("vol = {}".format(pb.problem.get_Vendo()))
    info("dof = {}".format(pb.problem.state.function_space().dim()))
