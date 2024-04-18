"""
Copied from
https://github.com/zhegan27/TSBN_code_NIPS2015/blob/master/bouncing_balls/data/data_handler_bouncing_balls.py
who said:

This script comes from the RTRBM code by Ilya Sutskever from
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.T_totalar
"""

import argparse
import json
from numpy import *
import os



FRICTION = False  # whether there is friction in the system
SIZE = 10


def norm(x): return sqrt((x**2).sum())
def sigmoid(x): return 1./(1.+exp(-x))


def new_speeds(m1, m2, v1, v2):
    new_v2 = (2*m1*v1 + v2*(m2-m1))/(m1+m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2

# size of bounding box: SIZE X SIZE.


def bounce_n(T=128, n=2, r=None, m=None, color_period=None):
    if r is None:
        r = array([4.0]*n)
    if m is None:
        m = array([1]*n)

    # r is to be rather small.
    X = zeros((T, n, 2), dtype='float')
    V = zeros((T, n, 2), dtype='float')
    if color_period is None:
        C = None
    else:
        C = zeros((T, n, 3), dtype='float')
        C[:, :, 0] = 1.0

    v = random.randn(n, 2)
    v = (v / norm(v)*.5)*1.0
    good_config = False
    while not good_config:
        x = 2+random.rand(n, 2)*8
        good_config = True
        for i in range(n):
            for z in range(2):
                if x[i][z]-r[i] < 0:
                    good_config = False
                if x[i][z]+r[i] > SIZE:
                    good_config = False

        # that's the main part.
        for i in range(n):
            for j in range(i):
                if norm(x[i]-x[j]) < r[i]+r[j]:
                    good_config = False

    eps = .5
    for t in range(T):
        # for how long do we show small simulation

        for i in range(n):
            X[t, i] = x[i]
            V[t, i] = v[i]

            if C is not None and t > 0:
                C[t, i, :] = C[t-1, i, :]
                if t % color_period == 0:
                    # Generates 0 1 0 1 0 1 0 1 0 1 0 1 0 1 and so on.
                    C[t, i, 1] = 1 if C[t, i, 1] == 0 else 0
                    # Generates 0 1 0 0 0 1 0 0 0 1 0 0 0 1 and so on.
                    last = C[max(0, t-1), i, 2]
                    lastlast = C[max(0, t-color_period-1), i, 2]
                    lastlastlast = C[max(0, t-color_period*2-1), i, 2]
                    if last == 0 and lastlast == 0 and lastlastlast == 0:
                        C[t, i, 2] = 1
                    else:
                        C[t, i, 2] = 0

        for mu in range(int(1/eps)):

            for i in range(n):
                # x[i]+=eps*v[i]
                x[i] += .5*v[i]

            # gravity and drag
            if FRICTION:
                for i in range(n):
                    if (x[i][1]+r[i] < SIZE):
                        v[i, 1] += .003
                    v[i] += -(.005*v[i])

            for i in range(n):
                for z in range(2):
                    if x[i][z]-r[i] < 0:
                        v[i][z] = abs(v[i][z])  # want positive
                    if x[i][z]+r[i] > SIZE:
                        v[i][z] = -abs(v[i][z])  # want negative
            for i in range(n):
                for j in range(i):
                    if norm(x[i]-x[j]) < r[i]+r[j]:
                        # if (x[i][0] > 0) and (x[i][0] < size) and (x[i][1] > 0) and (x[i][1] < size):
                        #  if (x[i][0] > 0) and (x[i][0] < size) and (x[i][1] > 0) and (x[i][1] < size):
                        # the bouncing off part:
                        w = x[i]-x[j]
                        w = w / norm(w)

                        v_i = dot(w.transpose(), v[i])
                        v_j = dot(w.transpose(), v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)

                        v[i] += w*(new_v_i - v_i)
                        v[j] += w*(new_v_j - v_j)

    return X, V, C


def ar(x, y, z):
    return z/2+arange(x, y, z, dtype='float')


def matricize(X, V, res, chunk_size, save_dir, r=None, C=None):
    T, n = shape(X)[0:2]
    if r is None:
        r = array([4.0]*n)

    [I, J] = meshgrid(ar(0, 1, 1./res)*SIZE, ar(0, 1, 1./res)*SIZE)
    if C is None:
        # When the color schedule is not specified, we use the velocity to determine the color
        C = zeros((T, n, 3), dtype='float')
        for i in range(n):
            C[:, i, 0] += 1.0 * (V[:, i, 0] + .5)
            C[:, i, 1] += 1.0
            C[:, i, 2] += 1.0 * (V[:, i, 1] + .5)

    n_chunks = T // chunk_size
    for chunk_idx in range(n_chunks):

        chunk_start = chunk_idx * chunk_size
        chunk_end = (chunk_idx + 1) * chunk_size

        A_c = zeros((chunk_size, res, res, 3), dtype='float')
        C_c = C[chunk_start:chunk_end]
        X_c = X[chunk_start:chunk_end]

        for t in range(chunk_size):
            for i in range(n):
                gaussian_bump = exp(-(((I-X_c[t, i, 0])**2+(J-X_c[t, i, 1])**2)/(r[i]**2))**4)
                A_c[t, :, :, 0] += C_c[t, i, 0] * gaussian_bump
                A_c[t, :, :, 1] += C_c[t, i, 1] * gaussian_bump
                A_c[t, :, :, 2] += C_c[t, i, 2] * gaussian_bump

            A_c[t, :, :, 0][A_c[t, :, :, 0] > 1] = 1
            A_c[t, :, :, 1][A_c[t, :, :, 1] > 1] = 1
            A_c[t, :, :, 2][A_c[t, :, :, 2] > 1] = 1

        if save_dir is not None:
            with open(f"{save_dir}/{chunk_idx}.npy", 'wb') as f:
                save(f, A_c)


def bounce_vec(save_dir, res, n=2, T=128, chunk_size=100, color_period=None, r=None, m=None):
    if r is None:
        r = array([1.2]*n)
    x, v, c = bounce_n(T, n, r, m, color_period)  # x: <seq_len x num_balls x 2>
    return matricize(x, v, res, chunk_size, save_dir, r, c)


def unsigmoid(x): return log(x) - log(1-x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="datasets/ball")
    parser.add_argument("--T_total", type=int, default=100000)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_balls", type=int, default=1)
    parser.add_argument("--color_period", type=int, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    train_path, test_path = f"{args.save_dir}/train", f"{args.save_dir}/test"
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    assert args.T_total % args.chunk_size == 0

    with open(f"{args.save_dir}/config.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # save train data
    bounce_vec(train_path, args.resolution, args.num_balls, args.T_total, args.chunk_size, args.color_period)

    # save test data
    bounce_vec(test_path, args.resolution, args.num_balls, args.T_total, args.chunk_size, args.color_period)
