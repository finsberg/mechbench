from fenics import *
from fenicshotools.gmsh import geo2dolfin
from fenicshotools.geometry import *

from textwrap import dedent
import os.path
import itertools

__all__ = [ 'EllipsoidGeometry' ]

class EllipsoidGeometry(object) :
    """
    Truncated ellipsoidal geometry, defined through the coordinates:

    X1 = Rl(t) cos(mu)
    X2 = Rs(t) sin(mu) cos(theta)
    X3 = Rs(t) sin(mu) sin(theta)

    for t in [0, 1], mu in [0, mu_base] and theta in [0, 2pi).
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
        dim = 2 if self.is_axisymmetric() else 3
        if regen :
            info('Generate geometry')
            code = self._compile_geo_code()
            domain, markers = geo2dolfin(code, dim, dim, comm)

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
        bfun = MeshFunction("size_t", mesh, dim-1, mesh.domains())
        for marker, (name, _) in markers.items() :
            setattr(self, name, marker)

        # microstructure
        mspace = self._parameters['microstructure']['function_space']
        if mspace != '' :
            info('Creating microstructure')
            # coordinate mapping
            coordscode = self._compile_cart2coords_code()
            coords = Expression(cppcode=coordscode)
            if domain.coordinates()  :
                coords.coords = domain.coordinates()
            # local coordinate base
            localbasecode = self._compile_localbase_code()
            localbase = Expression(cppcode=localbasecode)
            localbase.cart2coords = coords
            # function space
            family, degree = mspace.split("_")
            degree = int(degree)
            V = TensorFunctionSpace(domain, family, degree, shape=(3,3))
            # microstructure expression
            microcode = self._compile_microstructure_code()
            micro = Expression(cppcode=microcode)
            micro.cart2coords = coords
            micro.localbase = localbase
            micro.alpha_endo = self._parameters['microstructure']['alpha_endo']
            micro.alpha_epi = self._parameters['microstructure']['alpha_epi']
            # interpolation
            microinterp = interpolate(micro, V)
            self.s0 = microinterp[0, :]
            self.n0 = microinterp[1, :]
            self.f0 = microinterp[2, :]
        else :
            # microstructure not necessary
            self.f0 = None
            self.s0 = None
            self.n0 = None

        # exposing data
        self.domain = domain
        self.bfun = bfun

    @staticmethod
    def default_parameters() :
        p = { 'axisymmetric' : False,
              'mesh_generation' : {
                  'force_generation' : False,
                  'order' : 1,
                  'psize' : 3.0,
                  'ndiv'  : 1 },
              'microstructure' : {
                  'function_space' : 'Quadrature_4',
                  'alpha_endo' : +90.0,
                  'alpha_epi'  : -90.0 },
              'r_short_endo' : 7.0,
              'r_short_epi'  : 10.0,
              'r_long_endo'  : 17.0,
              'r_long_epi'   : 20.0,
              'quota_base'   : -5.0 }
        return p

    def is_axisymmetric(self) :
        """
        Check if the mesh is axisymmetric.
        """
        return self._parameters['axisymmetric']

    def get_apex_position(self, surf, u=None) :
        """
        Return the apex position.
        Keyword arguments:
        surf -- surface, can be 'endocardium' or 'epicardium'
        u    -- displacement field
        """
        mesh = self.domain.data()
        sid = { 'endocardium': self.ENDOPT, 'epicardium': self.EPIPT}[surf]
        marker = { k: v for v, k in mesh.domains().markers(0).items() }
        comm = mesh.mpi_comm()
        if sid in marker :
            idx = marker[sid]
            X = mesh.coordinates()[idx, :]
            if u :
                pos = X[0] + u(X)[0]
            else :
                pos = X[0]
        else :
            pos = 0.0

        return MPI.sum(comm, pos)

    def inner_volume_form(self, u = None) :
        dom = self.domain
        quota = self._parameters['quota_base']
        X = SpatialCoordinate(dom)
        N = FacetNormal(dom)

        if self.is_axisymmetric() :
            xshift = Constant((quota, 0.0))

            X = X - xshift
            N = as_vector([ N[0], N[1], 0.0 ])

            u = u or Constant((0.0, 0.0, 0.0), cell=dom)

            # Theta is always zero
            Z, R, T = X[0], X[1], 0.0
            z, r = Z + u[0], R + u[1]
            zZ, zR = z.dx(0), z.dx(1)
            rZ, rR = r.dx(0), r.dx(1)
            if u.ufl_shape[0] == 2 :
                t = 0.0;
                tZ, tR = 0.0, 0.0
            else :
                t = u[2]
                tZ, tR = t.dx(0), t.dx(1)

            # jacobian of the map
            Jgeo = 2.0*DOLFIN_PI*R
            # deformation gradient tensor
            F = as_tensor([[ zZ, zR, 0.0 ],
                           [ rZ, rR, 0.0 ],
                           [ tZ, tR, 1.0 ]])
            # normalization of components
            F = diag(as_vector([ 1, 1, r ])) * F
            F = F * diag(as_vector([ 1, 1, 1/R ]))

            x = as_vector([z, r, t])
            n = cofac(F)*as_vector([N[0], N[1], 0.0])
        else :
            xshift = Constant((quota, 0.0, 0.0))

            u = u or Constant((0.0, 0.0, 0.0))

            x = X + u - xshift
            F = grad(x)
            n = cofac(F) * N

            Jgeo = 1.0

        return -1.0/3.0 * Jgeo * inner(x, n)

    def inner_volume(self, u = None) :
        """
        Compute the inner volume.
        """
        ds_endo = ds(self.ENDO, subdomain_data = self.bfun)
        Vendo_form = self.inner_volume_form(u) * ds_endo
        V = assemble(Vendo_form)

        return V

    def _compile_geo_code(self) :
        """
        Geo code for the geometry.
        """

        code = dedent(\
        """\
        r_short_endo = {r_short_endo};
        r_short_epi  = {r_short_epi};
        r_long_endo  = {r_long_endo};
        r_long_epi   = {r_long_epi};
        quota_base = {quota_base};

        mu_base = Acos(quota_base / r_long_endo);

        psize_ref = {mesh_generation[psize]} / {mesh_generation[ndiv]};
        axisymmetric = {axisymmetric:^};

        Geometry.CopyMeshingMethod = 1;
        Mesh.ElementOrder = {mesh_generation[order]};
        Mesh.Optimize = 1;
        Mesh.OptimizeNetgen = 1;
        Mesh.HighOrderOptimize = 1;

        Function EllipsoidPoint
            Point(id) = {{ r_long  * Cos(mu),
                           r_short * Sin(mu) * Cos(theta),
                           r_short * Sin(mu) * Sin(theta), psize }};
        Return

        center = newp; Point(center) = {{ 0.0, 0.0, 0.0 }};

        theta = 0.0;

        r_short = r_short_endo; r_long = r_long_endo;
        mu = 0.0;
        psize = psize_ref / 2.0;
        apex_endo = newp; id = apex_endo; Call EllipsoidPoint;
        mu = mu_base;
        psize = psize_ref;
        base_endo = newp; id = base_endo; Call EllipsoidPoint;

        r_short = r_short_epi; r_long = r_long_epi;
        mu = 0.0;
        psize = psize_ref / 2.0;
        apex_epi = newp; id = apex_epi; Call EllipsoidPoint;
        mu = Acos(r_long_endo / r_long_epi * Cos(mu_base));
        psize = psize_ref;
        base_epi = newp; id = base_epi; Call EllipsoidPoint;

        apex = newl; Line(apex) = {{ apex_endo, apex_epi }};
        base = newl; Line(base) = {{ base_endo, base_epi }};
        endo = newl; Ellipse(endo) = {{ apex_endo, center, apex_endo, base_endo }};
        epi  = newl; Ellipse(epi) = {{ apex_epi, center, apex_epi, base_epi }};

        ll1 = newll; Line Loop(ll1) = {{ apex, epi, -base, -endo }};
        s1 = news; Plane Surface(s1) = {{ ll1 }};

        If (axisymmetric == 0)
            sendoringlist[] = {{ }};
            sepiringlist[]  = {{ }};
            sendolist[] = {{ }};
            sepilist[]  = {{ }};
            sbaselist[] = {{ }};
            vlist[] = {{ }};
            
            sold = s1;
            For i In {{ 0 : 3 }}
                out[] = Extrude {{ {{ 1.0, 0.0, 0.0 }}, {{ 0.0, 0.0, 0.0 }}, Pi/2 }}
                                {{ Surface{{sold}}; }};
                sendolist[i] = out[4];
                sepilist[i]  = out[2];
                sbaselist[i] = out[3];
                vlist[i] = out[1];
                bout[] = Boundary{{ Surface{{ sbaselist[i] }}; }};
                sendoringlist[i] = bout[1];
                sepiringlist[i] = bout[3];
                sold = out[0];
            EndFor

            Physical Volume("MYOCARDIUM") = {{ vlist[] }};
            Physical Surface("ENDO") = {{ sendolist[] }};
            Physical Surface("EPI") = {{ sepilist[] }};
            Physical Surface("BASE") = {{ sbaselist[] }};
            Physical Line("ENDORING") = {{ sendoringlist[] }};
            Physical Line("EPIRING") = {{ sepiringlist[] }};
        EndIf

        If (axisymmetric == 1)
            Physical Surface("MYOCARDIUM") = {{ s1 }};
            Physical Line("ENDO") = {{ endo }};
            Physical Line("EPI") = {{ epi }};
            Physical Line("BASE") = {{ base }};
            Physical Line("APEX") = {{ apex }};
            Physical Point("ENDORING") = {{ base_endo }};
            Physical Point("EPIRING") = {{ base_epi }};
        EndIf

        Physical Point("ENDOPT") = {{ apex_endo }};
        Physical Point("EPIPT") = {{ apex_epi }};
        """).format(**self._parameters)

        return code

    def _compile_cart2coords_code(self) :
        code = dedent(\
        """\
        #include <boost/math/tools/roots.hpp>

        namespace dolfin
        {{

        using boost::math::tools::newton_raphson_iterate;

        class SimpleEllipsoidCart2Coords : public Expression
        {{
        public :

            std::shared_ptr<dolfin::GenericFunction> coords;

            SimpleEllipsoidCart2Coords() : Expression(3)
            {{}}

            void eval(dolfin::Array<double>& values,
                      const dolfin::Array<double>& raw_x,
                      const ufc::cell& cell) const
            {{
                // coordinate mapping
                const std::size_t value_size = {axisymmetric:^} ? 2 : 3;
                dolfin::Array<double> x_tmp(value_size);

                if (this->coords)
                    coords->eval(x_tmp, raw_x, cell);
                else
                    std::copy(raw_x.data(), raw_x.data() + value_size, x_tmp.data());

                dolfin::Array<double> x(3);
                x[0] = x_tmp[0];
                x[1] = x_tmp[1];

                if ({axisymmetric:^} == 1) x[2] = 0.0;
                else x[2] = x_tmp[2];

                // constants
                const double r_short_endo = {r_short_endo};
                const double r_short_epi  = {r_short_epi};
                const double r_long_endo  = {r_long_endo};
                const double r_long_epi   = {r_long_epi};

                // to find the transmural position we have to solve a
                // 4th order equation. It is easier to apply bisection
                // in the interval of interest [0, 1]
                auto fun = [&](double t)
                {{
                    double rs = r_short_endo + (r_short_epi - r_short_endo) * t;
                    double rl = r_long_endo + (r_long_epi - r_long_endo) * t;
                    double a2 = x[1]*x[1] + x[2]*x[2];
                    double b2 = x[0]*x[0];
                    double rs2 = rs*rs;
                    double rl2 = rl*rl;
                    double drs = (r_short_epi - r_short_endo) * t;
                    double drl = (r_long_epi - r_long_endo) * t;

                    double f  = a2 * rl2 + b2 * rs2 - rs2 * rl2;
                    double df = 2.0 * (a2 * rl * drl + b2 * rs * drs
                                - rs * drs * rl2 - rs2 * rl * drl);

                    return boost::math::make_tuple(f, df);
                }};

                int digits = std::numeric_limits<double>::digits;
                double t = newton_raphson_iterate(fun, 0.5, 0.0, 1.0, digits);
                values[0] = t;

                double r_short = r_short_endo * (1-t) + r_short_epi * t;
                double r_long  = r_long_endo  * (1-t) + r_long_epi  * t;

                double a = std::sqrt(x[1]*x[1] + x[2]*x[2]) / r_short;
                double b = x[0] / r_long;

                // mu
                values[1] = std::atan2(a, b);

                // theta
                values[2] = (values[1] < DOLFIN_EPS)
                          ? 0.0
                          : M_PI - std::atan2(x[2], -x[1]);
            }}
        }};

        }};
        """).format(**self._parameters)

        return code

    def _compile_localbase_code(self) :
        code = dedent(\
        """\
        #include <Eigen/Dense>

        namespace dolfin
        {{

        class SimpleEllipsoidLocalCoords : public Expression
        {{
        public :

            typedef Eigen::Vector3d vec_type;
            typedef Eigen::Matrix3d mat_type;
            std::shared_ptr<dolfin::Expression> cart2coords;

            SimpleEllipsoidLocalCoords() : Expression(3, 3)
            {{}}

            void eval(dolfin::Array<double>& values,
                      const dolfin::Array<double>& raw_x,
                      const ufc::cell& cell) const
            {{
                // check if coordinates are ok
                assert(this->cart2coords);

                // first find (lambda, mu, theta) from (x0, x1, x2)
                // axisymmetric case has theta = 0
                dolfin::Array<double> coords(3);
                this->cart2coords->eval(coords, raw_x, cell);

                double t = coords[0];
                double mu = coords[1];
                double theta = coords[2];

                // (e_1, e_2, e_3) = G (e_t, e_mu, e_theta)
                const double r_short_endo = {r_short_endo};
                const double r_short_epi  = {r_short_epi};
                const double r_long_endo  = {r_long_endo};
                const double r_long_epi   = {r_long_epi};
                
                double rs = r_short_endo + (r_short_epi - r_short_endo) * t;
                double rl = r_long_endo + (r_long_epi - r_long_endo) * t;
                double drs = r_short_epi - r_short_endo;
                double drl = r_long_epi - r_long_endo;

                double sin_m = std::sin(mu);
                double cos_m = std::cos(mu);
                double sin_t = std::sin(theta);
                double cos_t = std::cos(theta);

                mat_type base;
                base << drl*cos_m,       -rl*sin_m,        0.0,
                        drs*sin_m*cos_t,  rs*cos_m*cos_t, -rs*sin_m*sin_t,
                        drs*sin_m*sin_t,  rs*cos_m*sin_t,  rs*sin_m*cos_t;
                if (mu < DOLFIN_EPS)
                {{
                    // apex, e_mu and e_theta not defined
                    // --> random, but orthonormal
                    base << 1, 0, 0,
                            0, 1, 0,
                            0, 0, 1;
                }}
                base = base.colwise().normalized();

                // in general this base is not orthonormal, unless
                //   d/dt ( rs^2(t) - rl^2(t) ) = 0
                bool enforce_orthonormal_base = true;
                if (enforce_orthonormal_base)
                {{
                    base.col(0) = base.col(1).cross(base.col(2));
                }}

                Eigen::Map<mat_type>(values.data()) = base;
            }}
        }};

        }};
        """).format(**self._parameters)

        return code

    def _compile_microstructure_code(self) :
        """
        C++ code for analytic fiber and sheet.
        """
        code = dedent(\
        """\
        #include <Eigen/Dense>

        class EllipsoidMicrostructure : public Expression
        {{
        public :

            typedef Eigen::Vector3d vec_type;
            typedef Eigen::Matrix3d mat_type;

            std::shared_ptr<dolfin::Expression> cart2coords;
            std::shared_ptr<dolfin::Expression> localbase;

            double alpha_epi, alpha_endo;

            EllipsoidMicrostructure() : Expression(3, 3),
                alpha_epi(0.0), alpha_endo(0.0)
            {{}}

            void eval(dolfin::Array<double>& values,
                      const dolfin::Array<double>& raw_x,
                      const ufc::cell& cell) const
            {{
                // check if coordinates are ok
                assert(this->localbase);
                assert(this->cart2coords);

                // first find (lambda, mu, theta) from (x0, x1, x2)
                dolfin::Array<double> coords(3);
                this->cart2coords->eval(coords, raw_x, cell);

                // then evaluate the local basis
                dolfin::Array<double> base(9);
                this->localbase->eval(base, raw_x, cell);

                // transmural position
                double pos = 0.0;
                pos = coords[0];

                // angles
                double alpha = (alpha_epi - alpha_endo) * pos + alpha_endo;
                alpha = alpha / 180.0 * M_PI;

                // Each column is a basis vector
                // --> [ e_lambda, e_mu, e_theta ]
                mat_type S = Eigen::Map<mat_type>(base.data());

                // Rotation around e_lambda of angle alpha
                Eigen::AngleAxisd rot1(alpha, S.col(0));
                S = rot1 * S;
                // --> [ n0, s0, f0 ]

                // Return the values
                Eigen::Map<mat_type>(values.data()) = S;
            }}
        }};
        """).format()

        return code

if __name__ == '__main__' :
    # test on different geometries
    parameters["form_compiler"]["representation"] = "uflacs"
    parameters["form_compiler"]["quadrature_degree"] = 4

    comm = mpi_comm_world()
    if MPI.rank(comm) != 0 :
        set_log_level(WARNING)

    axisym = True
    if axisym :
        ndivs  = [ 1, 2, 4, 8, 16, 32, 64 ]
        orders = [ 1, 2, 3 ]
    else :
        ndivs  = [ 1, 2, 4 ]
        orders = [ 1, 2 ]

    for ndiv, order in itertools.product(ndivs, orders) :
        info("===================")
        info("n = {}, order = {}".format(ndiv, order))
        info("===================")
        geoparam = EllipsoidGeometry.default_parameters()
        geoparam['axisymmetric'] = axisym
        geoparam['mesh_generation']['force_generation'] = True
        geoparam['mesh_generation']['order'] = order
        geoparam['mesh_generation']['ndiv'] = ndiv
        geoparam['microstructure']['function_space'] = ''
        geo = EllipsoidGeometry(comm, '', **geoparam)

        volume   = geo.inner_volume()
        apexendo = geo.get_apex_position('endocardium')
        apexepi  = geo.get_apex_position('epicardium')

        info("Volume    = {:.2f} mm3".format(volume))
        info("apex endo = {:.2f} mm".format(apexendo))
        info("apex epi  = {:.2f} mm".format(apexepi))
        info("")

