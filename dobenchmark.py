#!/usr/bin/env python
from fenics import *
from problem1 import Problem1
import argparse

def get_args() :
    descr = 'Run the benchmark for the 3 problems.'
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('--postprocess',
            default=False,
            action='store_true',
            help='Postprocess mode (only serial).')
    return parser.parse_args()

if __name__ == "__main__" :

    # mpi communicator
    comm = mpi_comm_world()
    if MPI.rank(comm) == 0 :
        set_log_level(PROGRESS)
    else :
        set_log_level(ERROR)

    # arguments
    args = get_args()
    post = args.postprocess

    # run the simulation
    quad  = 4
    order = 2
    fact  = 1.0
    ndivs = [ 1, 2, 4, 8, 16 ]
    dofs = []
    poss = []
    for ndiv in ndivs :
        pb = Problem1(comm, post, ndiv=ndiv, order=order, quad=quad,
                      reduction_factor=fact, structured=False)
        pb.run()
        dofs.append(pb.problem.state.function_space().dim())
        poss.append(pb.problem.get_point_position())

    info("order  ndiv   fact     dofs    posx    posy    posz")
    for ndiv, dof, pos in zip(ndivs, dofs, poss) :
        info("{:5d}  {:4d}  {:5.1f}  {:7d}  {:.4f}  {:.4f}  {:.4f}"\
             .format(order, ndiv, fact, dof, pos[0], pos[1], pos[2]))

