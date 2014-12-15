import sys, petsc4py
from fenics import *
from collections import deque
import numpy as np

__all__ = [ 'ContinuationSolver' ]
class ContinuationProblem(NonlinearProblem) :
    def __init__(self, problem) :
        self.problem = problem
        super(ContinuationProblem, self).__init__()

        self.fres = deque(maxlen=2)

        self.first_call = True
        self.skipF = False
        
        self._assemble_jacobian = True

    def form(self, A, b, x) :
        pb = self.problem
        if self._assemble_jacobian :
            assemble_system(pb.dG, pb.G, pb.bcs,
                    A_tensor = A, b_tensor = b)
        else :
            assemble(pb.G, tensor=b)
            if pb.bcs : 
                for bc in pb.bcs :
                    bc.apply(b)
        self._assemble_jacobian = not self._assemble_jacobian

        return
        Timer("ContinuationSolver: form")
        pb = self.problem

        # check if we need to assemble the jacobian
        if self.first_call :
            reset_jacobian = True
            self.first_call = False
            self.skipF = True
        else :
            reset_jacobian = b.empty() and not A.empty()
            self.skipF = reset_jacobian

            if len(self.fres) == 2 and reset_jacobian :
                if self.fres[1] < 0.1*self.fres[0] :
                    debug("REUSE J")
                    reset_jacobian = False

        if reset_jacobian :
            # requested J, assemble both
            debug("ASSEMBLE J")
            assemble_system(pb.dG, pb.G, pb.bcs, x0=x,
                            A_tensor=A, b_tensor=b)

    def J(self, A, x) :
        pass

    def F(self, b, x) :
        return
        if self.skipF : return
        pb = self.problem
        assemble(pb.G, tensor=b)
        for bc in pb.bcs : bc.apply(b)
        self.fres.append(b.norm('l2'))


class ContinuationSolver(object) :
    def __init__(self, problem, symmetric=True, backend='mumps') :
        # initialize the nonlinear problem
        self.problem = problem
        self.nlproblem = ContinuationProblem(problem)

        # initialize the solver
        solver = PETScSNESSolver()
        solver.parameters["linear_solver"] = "gmres"
        solver.parameters["report"] = False
        PETScOptions.set('snes_monitor')
        PETScOptions.set('ksp_type', 'preonly')
        if symmetric :
            PETScOptions.set('pc_type', 'cholesky')
        else :
            PETScOptions.set('pc_type', 'lu')
        if backend == 'pastix' :
            PETScOptions.set('pc_factor_mat_solver_package', 'pastix')
            PETScOptions.set("mat_pastix_threadnbr", 4)
            PETScOptions.set("mat_pastix_verbose", 0)
        elif backend == 'mumps' :
            PETScOptions.set('pc_factor_mat_solver_package', 'mumps')
            PETScOptions.set("mat_mumps_icntl_7", 6)

        self.solver = solver

    def solve(self, step=0.5, **controls) :
        # the problem
        pb = self.problem
        nlpb = self.nlproblem
        solver = self.solver

        # target parameters
        pnames  = controls.keys()
        ptarget = np.array(controls.values())
        pstart  = np.array(pb.get_control_parameters(pnames).values())
        pdir = (ptarget - pstart) / np.linalg.norm(ptarget - pstart)

        # initial pseudo-timestep
        pp = pstart
        if (pp + step*pdir < ptarget).all() :
            dt = step
        else :
            dt = np.linalg.norm(ptarget - pp)
        tt = 0.0

        titer = 0
        while (ptarget > pp).any() :
            titer += 1
            # save old solution
            uold = pb.state.copy(True)
            told = tt
            pold = pp.copy()

            # tentative step
            pp += dt*pdir
            info("")
            info("SOLVING FOR {}".format(", ".join(
                 "{} = {:.4f}".format(n, v) for n, v in zip(pnames, pp))))
            pb.set_control_parameters(**dict(zip(pnames, pp)))

            try :
                nliter, nlconv = solver.solve(nlpb, pb.state.vector())
                nlpb.first_call = True
            except RuntimeError :
                info("CONVERGENCE FAILURE, reducing time-step")
                tt = told
                pp[:] = pold
                pb.state.assign(uold)
                pb.set_control_parameters(**dict(zip(pnames, pp)))
                dt *= 0.5
                continue

            # adapting
            if nliter < 6 :
                dt *= 1.5

            # check for last step
            if (ptarget - pp - dt*pdir < 0.0).any() :
                dt = np.linalg.norm(ptarget - pp)

        info("TARGET REACHED in {} steps".format(titer))

