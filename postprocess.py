from fenics import *
from fenicshotools.linearizedomain import *
from fenicshotools.vtkutils import *
import numpy as np

__all__ = [ "compute_postprocessed_quantities" ]

def __extract_displacement_with_domain(problem, isoparam=True) :
    # we reconstruct the function space to keep track of
    # the domain (bug in DOLFIN 1.5)
    # extract the displacement
    if problem.mat.is_incompressible() :
        uold = problem.state.sub(0)
        Vold = problem.state.function_space().sub(0)
    else :
        uold = problem.state
        Vold = problem.state.function_space()
    # it might be that the domain is linear but the field is not.
    # In this case we construct isoparametric element when requested
    elm = Vold.ufl_element()
    family, degree, shape = elm.family(), elm.degree(), elm.value_shape()[0]
    D = problem.geo.domain
    if isoparam and not D.coordinates() :
        mesh = D.data()
        gdim = D.geometric_dimension()
        Vlin = VectorFunctionSpace(mesh, family, degree, shape)
        e = Expression(tuple("x[{}]".format(i) for i in range(gdim)),
                       element=Vlin.ufl_element())
        coords = Function(Vlin)
        coords.interpolate(e)
        D = Domain(coords)

    V = VectorFunctionSpace(D, family, degree, shape)
    u = Function(V)
    assign(u, uold)

    return u, D

def __axisymmetric_to_cartesian_displacement(ufun, domain) :
    V = ufun.function_space()
    idxV = np.column_stack([ V.sub(i).dofmap().dofs()
                for i in xrange(0, V.num_sub_spaces()) ])
    if idxV.shape[1] == 2 :
        uX, uR = np.hsplit(ufun.vector().array()[idxV], 2)
        uT = np.zeros(shape=uX.shape)
    else :
        uX, uR, uT = np.hsplit(ufun.vector().array()[idxV], 3)

    # extract dof coordinates
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

    ufun.vector()[idxV[:,0].copy()] = uX.flatten().astype(np.float)
    ufun.vector()[idxV[:,1].copy()] = uY.flatten().astype(np.float)
    if idxV.shape[1] == 3 :
        ufun.vector()[idxV[:,2].copy()] = uZ.flatten().astype(np.float)

def compute_postprocessed_quantities(problem, ndiv=-1) :
    # extract the displacement
    u, domain = __extract_displacement_with_domain(problem)
    # linearize the domain if requested
    if ndiv >= 0 :
        mesh, u = linearize_domain_and_fields(domain, u, ndiv=ndiv)
        domain = mesh.ufl_domain()
    u.rename("displacement", "displacement")

    # compute the strain and the cartesian displacement
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

    J = det(F)
    Jm23 = pow(J, -float(2)/3)
    C = Jm23 * F.T*F
    E = 0.5*(C - I)

    if ndiv >=0 :
        V = TensorFunctionSpace(domain, 'DG', 0, shape=(3,3))
        a = Form(inner(TestFunction(V), TrialFunction(V))*Jgeo*dx)
        L = Form(inner(TestFunction(V), E)*Jgeo*dx)
        Eout = Function(V, name='strain')
        lsolver = LocalSolver()
        lsolver.solve(Eout.vector(), a, L)
    else :
        # in this case we keep the high-order domain unchanged
        # and we perform a global L2 projection to recover a
        # continuous strain
        V = TensorFunctionSpace(domain, 'CG', 2, shape=(3,3))
        a = inner(TestFunction(V), TrialFunction(V))*Jgeo*dx
        L = inner(TestFunction(V), E)*Jgeo*dx
        Eout = Function(V, name='strain')
        solve(a == L, Eout, solver_parameters={"linear_solver":"mumps"})

    # displacement in cartesian coordinates
    if problem.geo.is_axisymmetric() :
        __axisymmetric_to_cartesian_displacement(u, domain)
    
    return domain, u, Eout

