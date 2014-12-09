from fenics import *

__all__ = [ "LVProblem" ]

class LVProblem(object) :
    def __init__(self, geo, mat) :
        """
        Initialise the problem.
        Keyword arguments:
        geo -- geometry
        mat -- material
        """

        # set up of the problem
        domain = geo.domain

        # isoparametric elements, so we extract the
        # space from the coordinates
        order = 1
        if domain.coordinates() :
            order = domain.coordinates().ufl_element().degree()
        # for axisymmetric and isotropic deformation, we expect
        # no torsion, so when neglect the azimuthal deformation
        vdim = 2 if geo.is_axisymmetric() and mat.is_isotropic() else 3
        V = VectorFunctionSpace(domain, 'P', max(order, 2), vdim)

        # we deal with strict incompressibility by means of a
        # Lagrangian multiplier in a mixed formulation, with
        # Taylor-Hood element (P_{k+1} / P_{k}, k>=1)
        if mat.is_incompressible() :
            Q = FunctionSpace(domain, 'P', max(order-1, 1))
            M = MixedFunctionSpace([V, Q])
        else :
            M = V

        # the state of the system
        state = Function(M)
        u = split(state)[0] if mat.is_incompressible() else state
        p = split(state)[1] if mat.is_incompressible() else None

        # the deformation gradient tensor
        if geo.is_axisymmetric() :
            # Theta is always zero
            Z, R = SpatialCoordinate(domain)
            z, r = Z + u[0], R + u[1]
            zZ, zR, zT = z.dx(0), z.dx(1), 0.0
            rZ, rR, rT = r.dx(0), r.dx(1), 0.0
            if mat.is_isotropic() :
                tZ, tR, tT = 0.0, 0.0, 1.0
            else :
                t = u[2]
                tZ, tR, tT = t.dx(0), t.dx(1), 1.0
            # the tensor
            F = as_tensor([[ zZ, zR, zT ],
                           [ rZ, rR, rT ],
                           [ tZ, tR, tT ]])
            # normalization of components
            F = diag(as_vector([ 1, 1, r ])) * F
            F = F * diag(as_vector([ 1, 1, 1/R ]))
            # jacobian of the map
            Jgeo = 2.0*DOLFIN_PI*R
        else :
            F = Identity(3) + grad(u)
            Jgeo = 1.0

        # strain energy
        Lint = mat.strain_energy(F, p) * Jgeo * dx

        # inner pressure
        pendo = Constant(0.0, name='pendo')
        ds_endo = ds(geo.ENDO, subdomain_data = geo.bfun)
        Lext = - pendo * geo.inner_volume_form(u) * ds_endo

        # total energy
        L = Lint + Lext

        # boundary conditions
        Vu = M.sub(0) if mat.is_incompressible() else M
        czero = Constant(tuple([0.0]*vdim))
        # fixed base
        bcs = [ DirichletBC(Vu, czero, geo.bfun, geo.BASE) ]
        # apex symmetry
        if geo.is_axisymmetric() :
            zero = Constant(0.0)
            bcs += [ DirichletBC(Vu.sub(1), zero, geo.bfun, geo.APEX) ]

        # target problem
        G  = derivative(L, state, TestFunction(M))
        dG = derivative(G, state, TrialFunction(M))

        self.state = state
        self.pendo = pendo
        self.G  = G
        self.dG = dG
        self.bcs = bcs

        self.geo = geo
        self.mat = mat

        self.F = F
        self.Jgeo = Jgeo

    def set_pendo(self, value) :
        self.pendo.assign(value)

    def get_pendo(self) :
        return float(self.pendo)

    def get_Vendo(self) :
        if self.mat.is_incompressible() :
            u = split(self.state)[0]
        else :
            u = self.state
        return self.geo.inner_volume(u)

    def get_apex_position(self, surf) :
        inc = self.mat.is_incompressible()
        u = self.state.split(True)[0] if inc else self.state
        return self.geo.get_apex_position(surf, u)

    def get_control_parameters(self, controls) :
        vals = []
        for c in controls :
            if c == 'pendo' :
                vals += [ self.get_pendo() ]
            else :
                vals += self.mat.get_control_parameters([c])
        return dict(zip(controls, vals))

    def set_control_parameters(self, **controls) :
        vals = []
        for c, v in controls.items() :
            if c == 'pendo' :
                self.set_pendo(v)
            else :
                self.mat.set_control_parameters(**{c: v})
        return vals

