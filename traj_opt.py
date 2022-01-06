from common import legs
import numpy as np
import casadi as ca

tf = 5
N = int(tf*4)
epsilon = 1e-6


def extract_state(X, U, k):
    p = X[:3, k]
    R_flat = X[3:12, k]
    R = ca.reshape(R_flat, 3, 3)
    pdot = X[12:15, k]
    omega = X[15:18, k]
    p_i = {}
    f_i = {}
    for leg in legs:
        p_i[leg] = U[3*leg.value: leg.value*3+3, k]
        f_i[leg] = U[12+3*leg.value: 12+leg.value*3+3, k]
    return p, R, pdot, omega, p_i, f_i


if __name__ == "__main__":
    X = ca.SX.sym('X', 18, N+1)
    U = ca.SX.sym('U', 24, N+1)
    p, R, pdot, omega, p_i, f_i = extract_state(X, U, k=0)

    print(p)
    print(R)
    print(pdot)
    print(omega)
    for leg in legs:
        print(p_i[leg])
    for leg in legs:
        print(f_i[leg])
    import ipdb
    ipdb.set_trace()
