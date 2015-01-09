from fenics import *
from fenicshotools.gmsh import geo2dolfin
from fenicshotools.geometry import *

from textwrap import dedent
import os.path

__all__ = [ "BarGeometry" ]

class BarGeometry(object) :
    """
    Simple geometry for a bar.
    """

    def __init__(self, comm=None, h5name='', h5group='', **params) :
        """
        Keyword arguments:
        comm    -- MPI communicator (if None, MPI_COMM_WORLD is used)
        h5name  -- Filename (if empty, mesh is generated into memory)
        h5group -- HDF5 group for the geometry
        params  -- Geometry parameters (see default_parameters())
        """

        # mpi communicator
        comm = comm or mpi_comm_world()

        # set up parameters
        params = params or {}
        self._parameters = self.default_parameters()
        self._parameters.update(params)

        # try to load the mesh
        regen = True
        if not self._parameters['mesh_generation']['force_generation'] :
            if os.path.isfile(h5name) :
                # load the geometry
                info('Load geometry from file')
                domain, markers = load_geometry(comm, h5name, h5group)
                # load the parameters
                p = { k: v for k, v in self._parameters.items() 
                         if not isinstance(v, dict) }
                with HDF5File(comm, h5name, 'a') as f :
                    ggroup = '{}/geometry'.format(h5group)
                    for k, v in p.items() :
                        self._parameters[k] = f.attributes(ggroup)[k]
                regen = False

        # regenerate the mesh if necessary
        if regen :
            info('Generate geometry')
            code = self._compile_geo_code()
            domain, markers = geo2dolfin(code, 3, 3, comm)

        # save the mesh
        if regen and h5name != '' :
            info('Save geometry to file')
            save_geometry(comm, domain, h5name, h5group, markers)
            # save parameters
            p = { k: v for k, v in self._parameters.items() 
                       if not isinstance(v, dict) }
            with HDF5File(comm, h5name, 'a') as f :
                ggroup = '{}/geometry'.format(h5group)
                for k, v in p.items() :
                    f.attributes(ggroup)[k] = v

        # markers
        mesh = domain.data()
        bfun = MeshFunction("size_t", mesh, 2, mesh.domains())
        for marker, (name, _) in markers.items() :
            setattr(self, name, marker)

        # microstructure
        self.f0 = as_vector([ 1.0, 0.0, 0.0 ])
        self.s0 = as_vector([ 0.0, 1.0, 0.0 ])
        self.n0 = as_vector([ 0.0, 0.0, 1.0 ])

        # exposing data
        self.domain = domain
        self.bfun = bfun

    @staticmethod
    def default_parameters() :
        p = { 'lx' : 10.0, 'ly' : 1.0, 'lz' : 1.0,
              'mesh_generation' : {
                  'force_generation' : False,
                  'structured' : False,
                  'reduction_factor' : 16.0,
                  'nx' : 4, 'ny' : 1, 'nz' : 1,
                  'psize' : 1.0,
                  'ndiv' : 1 } }
        return p

    def is_axisymmetric(self) :
        return False

    def get_point_position(self, u=None) :
        """
        Return the position of the top-right-back point.
        Keyword arguments:
        u    -- displacement field
        """
        mesh = self.domain.data()
        marker = { k: v for v, k in mesh.domains().markers(0).items() }
        sid = self.TOPLEFTBACK
        comm = mesh.mpi_comm()
        if sid in marker :
            idx = marker[sid]
            X = mesh.coordinates()[idx, :]
            if u :
                pos1 = X[0] + u(X)[0]
                pos2 = X[1] + u(X)[1]
                pos3 = X[2] + u(X)[2]
            else :
                pos1 = X[0]
                pos2 = X[1]
                pos3 = X[2]
        else :
            pos1, pos2, pos3 = 0.0, 0.0, 0.0

        # we return the symmetric point
        # wrt to plane y=0.5.
        pos1 = MPI.sum(comm, pos1)
        pos2 = MPI.sum(comm, pos2)
        pos3 = MPI.sum(comm, pos3)

        return [ pos1, 1.0-pos2, pos3 ]

    def _compile_geo_code(self) :
        """
        Geo code for the geometry.
        """
        code = dedent(\
        """\
        // we do not use extrude to have
        // easier control on physical markers
        lx = {lx};
        ly = {ly} / 2.0;
        lz = {lz};
        ndiv = {mesh_generation[ndiv]};
        nx = {mesh_generation[nx]} * ndiv;
        ny = {mesh_generation[ny]} * ndiv;
        nz = {mesh_generation[nz]} * ndiv;
        psize = {mesh_generation[psize]} / ndiv;
        psize_low  = psize / {mesh_generation[reduction_factor]};
        psize_high = psize;
        pp = Exp(1/(nx-2)*Log({mesh_generation[reduction_factor]}));

        Geometry.CopyMeshingMethod = 1;

        If ({mesh_generation[structured]:^} == 0)
            Point(1) = {{ 0.0, 0.0, 0.0, psize_low }};
            Point(2) = {{ 0.0,  ly, 0.0, psize_low }};
            Point(3) = {{ 0.0,  ly,  lz, psize_low }};
            Point(4) = {{ 0.0, 0.0,  lz, psize_low }};

            Point(5) = {{  lx, 0.0, 0.0, psize_high }};
            Point(6) = {{  lx,  ly, 0.0, psize_high }};
            Point(7) = {{  lx,  ly,  lz, psize_high }};
            Point(8) = {{  lx, 0.0,  lz, psize_high }};
       
            // edges
            Line(1)  = {{ 1, 2 }};
            Line(2)  = {{ 2, 3 }};
            Line(3)  = {{ 3, 4 }};
            Line(4)  = {{ 4, 1 }};
            Line(5)  = {{ 5, 6 }};
            Line(6)  = {{ 6, 7 }};
            Line(7)  = {{ 7, 8 }};
            Line(8)  = {{ 8, 5 }};
            Line(9)  = {{ 1, 5 }};
            Line(10) = {{ 6, 2 }};
            Line(11) = {{ 4, 8 }};
            Line(12) = {{ 7, 3 }};

            // left face
            Line Loop(1) = {{ 1, 2, 3, 4 }};
            Plane Surface(1) = {{ 1 }};
            // right face
            Line Loop(2) = {{ 5, 6, 7, 8 }};
            Plane Surface(2) = {{ 2 }};
            // front face
            Line Loop(3) = {{ 9, -8, -11, 4 }};
            Plane Surface(3) = {{ 3 }};
            // back face
            Line Loop(4) = {{ 10, 2, -12, -6 }};
            Plane Surface(4) = {{ 4 }};
            // bottom face
            Line Loop(5) = {{ 9, 5, 10, -1 }};
            Plane Surface(5) = {{ 5 }};
            // top face
            Line Loop(6) = {{ 11, -7, 12, 3 }};
            Plane Surface(6) = {{ 6 }};

            // volume
            Surface Loop(1) = {{ 1, 2, 3, 4, 5, 6 }};
            Volume(1) = {{ 1 }};

            Mesh.Optimize = 1;
            Mesh.OptimizeNetgen = 1;

            // markers
            Physical Volume("BAR") = {{ 1 }};
            Physical Surface("LEFT") = {{ 1 }};
            Physical Surface("BOTTOM") = {{ 5 }};
            Physical Surface("BACK") = {{ 4 }};
            Physical Point("TOPLEFTBACK") = {{ 8 }};
        EndIf

        If ({mesh_generation[structured]:^} == 1)
            Point(1) = {{ 0.0, 0.0, 0.0 }};
            Extrude {{ 0.0,  ly, 0.0 }} {{ Point{{1}}; Layers{{ny}}; }}
            Extrude {{ 0.0, 0.0,  lz }} {{  Line{{1}}; Layers{{nz}}; }}

            a = 1/nx;
            If (pp != 1)
                a = (pp-1)/(pp^nx-1);
            EndIf
            one[0] = 1;
            layer[0] = a;
            For i In {{1:nx-1}}
                one[i] = 1;
                layer[i] = layer[i-1] + a*pp^i;
            EndFor

            Extrude {{ lx, 0.0, 0.0 }}
                    {{ Surface{{5}}; Layers{{one[], layer[]}}; }}

            // markers
            Physical Volume("BAR") = {{ 1 }};
            Physical Surface("LEFT") = {{ 5 }};
            Physical Surface("BOTTOM") = {{ 14 }};
            Physical Surface("BACK") = {{ 18 }};
            Physical Point("TOPLEFTBACK") = {{ 14 }};
        EndIf
        """).format(**self._parameters)

        return code

if __name__ == '__main__' :
    comm = mpi_comm_world()
    if MPI.rank(comm) != 0 :
        set_log_level(WARNING)

    geo = BarGeometry()
    info("position = {}".format(geo.get_point_position()))

