from fenics import *
from fenicshotools.linearizedomain import *
from fenicshotools.vtkutils import *

__all__ = [ "compute_strain" ]

def compute_strain(problem, ndiv=0) :
    # first we linearize the domain
    #ucoarse = problem.state.sub(0, deepcopy=True)
    domain = problem.geo.domain
    #mesh, u = linearize_domain_and_fields(domain, ucoarse, ndiv=ndiv)
    #domain = mesh.ufl_domain()
    u = split(problem.state)[0]

    # compute the strain
    I = Identity(3)
    gradu = grad(u)

    if problem.geo.is_axisymmetric() :
        X = SpatialCoordinate(domain)
        Z, R, T = X[0], X[1], 0.0
        z, r, t = Z + u[0], R + u[1], T
        zZ, zR, zT = z.dx(0), z.dx(1), 0.0
        rZ, rR, rT = r.dx(0), r.dx(1), 0.0
        if problem.mat.is_isotropic() :
            tZ, tR, tT = 0.0, 0.0, 1.0
        else :
            t += u[2]
            tZ, tR, tT = t.dx(0), t.dx(1), 1.0
        # the tensor
        Fax = as_tensor([[ zZ, zR, zT ],
                         [ rZ, rR, rT ],
                         [ tZ, tR, tT ]])
        M1 = as_tensor([[ 1.0,    0.0,       0.0 ],
                        [ 0.0, cos(t), -r*sin(t) ],
                        [ 0.0, sin(t),  r*cos(t) ]])
        M2 = as_tensor([[ 1.0,    0.0,       0.0 ],
                        [ 0.0, cos(T), -R*sin(T) ],
                        [ 0.0, sin(T),  R*cos(T) ]])
        F = M1*Fax*inv(M2)
        # jacobian of the map
        Jgeo = 2.0*DOLFIN_PI*R
    else :
        F = I + gradu
        Jgeo = 1.0

    Jm23 = pow(det(F), -float(2)/3)
    C = Jm23 * F.T*F
    E = 0.5*(C - I)

    V = TensorFunctionSpace(domain, 'DG', 0, shape=(3,3))
    a = Form(inner(TestFunction(V), TrialFunction(V))*Jgeo*dx)
    L = Form(inner(TestFunction(V), E)*Jgeo*dx)
    Eout = Function(V, name='strain')
    lsolver = LocalSolver()
    lsolver.solve(Eout.vector(), a, L)

    return Eout

def compute_errorL2() :
    # load most refined solution
    pbref = Problem2(comm, True, order=2, ndiv=32, quad=4)
    pbref.run()
    uref = pbref.problem.state.sub(0, deepcopy=True)
    errL2 = []
    ndivs = [ 1, 2, 4, 8 ]
    for ndiv in ndivs :
        pb = Problem2(comm, False, order=2, ndiv=ndiv, quad=6)
        pb.run()
        u = pb.problem.state.sub(0, deepcopy=True)
        # interpolate on the refined mesh
        V = VectorFunctionSpace(pbref.problem.geo.domain, 'P', 2)
        uint = Function(V)
        I = LagrangeInterpolator()
        I.interpolate(uint, u)
        uuref = split(pbref.problem.state)[0]
        errL2.append(sqrt(assemble(inner(uuref-uint,uuref-uint)*pbref.problem.Jgeo*dx)))

    print np.array(errL2)
    import matplotlib.pyplot as plt
    hh = 1./np.array(ndivs)
    plt.loglog(hh, errL2, linewidth=2)
    plt.loglog(hh[0:2], hh[0:2]**2, 'k')
    plt.loglog(hh[0:2], hh[0:2], 'k--')
    plt.grid(True)
    plt.show()

def compute_axisymmetric_to_cartesian_displacement(problem) :
    # displacement reordered by components
    ufun = Function(problem.state.sub(0, deepcopy=True), name="displacement")
    if not problem.geo._parameters['axisymmetric'] :
        return ufun

    V = ufun.function_space()
    idxV = np.column_stack([ V.sub(i).dofmap().dofs()
                for i in xrange(0, V.num_sub_spaces()) ])
    uX, uR, uT = np.hsplit(ufun.vector().array()[idxV], 3)

    # extract dof coordinates
    domain = problem.geo.domain
    if domain.coordinates() :
        phi = domain.coordinates()
        VD = phi.function_space()
        idxD = np.column_stack([ VD.sub(i).dofmap().dofs()
                    for i in xrange(0, VD.num_sub_spaces()) ])
        X, R = np.hsplit(phi.vector().array()[idxD], 2)
    else :
        coords = V.dofmap().tabulate_all_coordinates(V.mesh()).reshape((-1,2))
        X, R = np.hsplit(coords[idxV[:,0]], 2)
    T = np.zeros(shape=X.shape)

    # displacement in cartesian coordinates
    x, r, t = X + uX, R + uR, T + uT
    Y, Z = R * np.cos(T), R * np.sin(T)
    y, z = r * np.cos(t), r * np.sin(t)
    uY, uZ = y - Y, z - Z

    ufun.vector()[idxV[:,0]] = uX.flatten().astype(np.float)
    ufun.vector()[idxV[:,1]] = uY.flatten().astype(np.float)
    ufun.vector()[idxV[:,2]] = uZ.flatten().astype(np.float)

    return ufun

