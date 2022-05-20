import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from distutils.util import strtobool
from datetime import datetime

from force_qp import solve_force_qp
from contact_qp import solve_contact_qp

from constants import *
from draw import animate_traj
from generate_reference import generate_reference
from utils import extract_X_from_XR, produce_XR_from_X

plt.style.use("seaborn")


def calc_consensus(X, X_prev):
    N = X.shape[1] - 1  # number of time intervals
    l = X[6:9, :]
    l_prev = X_prev[6:9, :]
    consensus = np.linalg.norm(l - l_prev) ** 2 / N
    return consensus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--display",
        help="toggle whether to display animation window",
        type=strtobool,
        default=1,
    )
    parser.add_argument("-n", "--name", help="experiment name", type=str, default=None)
    parser.add_argument(
        "-s", "--save", help="toggle whether to save video", type=strtobool, default=0
    )

    # parse and post processing
    args = parser.parse_args()
    args.display = bool(args.display)
    args.save = bool(args.save)
    if args.name is None:
        args.name = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    start_time = time.time()
    XR_ref, dt = generate_reference()
    X_ref = extract_X_from_XR(XR_ref)
    X = X_ref.copy()

    consensus_arr = []
    obj_arr = []
    time_arr = []

    X_prev = X
    for iter in range(5):
        X, info_fqp = solve_force_qp(X, XR_ref, dt)
        X, info_cqp = solve_contact_qp(X, dt)

        obj_arr.append(info_fqp.obj_val)
        time_arr.append(info_fqp.run_time)
        time_arr.append(info_cqp.run_time)

        consensus = calc_consensus(X, X_prev)
        print("iteration {}, consensus = {:.3}".format(iter, consensus))
        consensus_arr.append(consensus)

        X_prev = X
    X, info_fqp = solve_force_qp(X, XR_ref, dt)
    time_arr.append(info_fqp.run_time)

    consensus_arr = np.array(consensus_arr)
    obj_arr = np.array(obj_arr)
    time_arr = np.array(time_arr)

    print("\ntotal time used in OSQP: {:.3} seconds".format(np.sum(time_arr)))
    print("total time used in program: {:.3} seconds".format(time.time() - start_time))

    fontsize = 15
    plt.subplot(2, 1, 1)
    plt.plot(obj_arr, "-o")
    plt.ylabel("Force QP Objective", size=fontsize)
    plt.tick_params(axis="x", labelsize=fontsize)
    plt.tick_params(axis="y", labelsize=fontsize)
    plt.subplot(2, 1, 2)
    plt.plot(consensus_arr, "-o")
    plt.yscale("log")
    plt.xlabel("BCD Iterations", size=fontsize)
    plt.ylabel("Consensus Parameter", size=fontsize)
    plt.tick_params(axis="x", labelsize=fontsize)
    plt.tick_params(axis="y", labelsize=fontsize)
    plt.show()

    # optionally display/save solution animation
    if args.save:
        fname = args.name
    else:
        fname = None

    R_init_flat = XR_ref[-9:, 0]
    R_init = np.reshape(R_init_flat, (3, 3), order="F")
    XR_sol = produce_XR_from_X(X, dt, R_init)
    animate_traj(XR_sol, dt, fname=fname, display=args.display, repeat=False)
