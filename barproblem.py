from fenics import *

__all__ = [ "BarProblem" ]

class BarProblem(object) :
    def __init__(self, geo, mat, order=2) :
        """
        Initialise the problem.
        Keyword arguments:
        geo -- geometry
        mat -- material
        """

        # set up of the problem
        domain = geo.domain

        V = VectorFunctionSpace(domain, 'P', order)
        # we deal with strict incompressibility by means of a
        # Lagrangian multiplier in a mixed formulation, with
        # Taylor-Hood element (P_{k+1} / P_{k}, k>=1)
        if mat.is_incompressible() :
            Q = FunctionSpace(domain, 'P', order-1)
            M = MixedFunctionSpace([V, Q])
        else :
            M = V

        # the state of the system
        state = Function(M)
        u = split(state)[0] if mat.is_incompressible() else state
        p = split(state)[1] if mat.is_incompressible() else None

        # deformation gradient
        F = Identity(3) + grad(u)
        Jgeo = 1.0

        # strain energy
        Lint = mat.strain_energy(F, p) * Jgeo * dx

        # pressure at the bottom
        stest = TestFunction(M)
        utest = split(stest)[0] if mat.is_incompressible() else stest

        N = FacetNormal(domain)
        pbottom = Constant(0.0, name='pbottom', cell=domain)
        ds_bottom = ds(geo.BOTTOM, subdomain_data=geo.bfun)
        Gext = pbottom * inner(utest, cofac(F)*N) * ds_bottom

        # total energy
        L = Lint

        # target problem
        G  = derivative(L, state, stest) + Gext
        dG = derivative(G, state, TrialFunction(M))

        # boundary condition
        Vu = M.sub(0) if mat.is_incompressible() else M
        czero = Constant((0.0, 0.0, 0.0))
        # fixed left face
        bcs  = [ DirichletBC(Vu, czero, geo.bfun, geo.LEFT) ]
        # we solve only half of the bar
        bcs += [ DirichletBC(Vu.sub(1), Constant(0.0), geo.bfun, geo.BACK) ]

        self.state = state
        self.pbottom = pbottom
        self.G  = G
        self.dG = dG
        self.bcs = bcs

        self.geo = geo
        self.mat = mat

        self.F = F
        self.Jgeo = Jgeo

    def get_point_position(self) :
        inc = self.mat.is_incompressible()
        u = self.state.split(True)[0] if inc else self.state
        return self.geo.get_point_position(u)

    def get_control_parameters(self, controls) :
        vals = []
        for c in controls :
            if c == 'pbottom' :
                vals += [ float(self.pbottom) ]
            else :
                vals += self.mat.get_control_parameters([c])
        return dict(zip(controls, vals))

    def set_control_parameters(self, **controls) :
        vals = []
        for c, v in controls.items() :
            if c == 'pbottom' :
                self.pbottom.assign(v)
            else :
                self.mat.set_control_parameters(**{c: v})
        return vals

