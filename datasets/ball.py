"""
Copied from
https://github.com/zhegan27/TSBN_code_NIPS2015/blob/master/bouncing_balls/data/data_handler_bouncing_balls.py
who said:

This script comes from the RTRBM code by Ilya Sutskever from
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.T_totalar
"""

import argparse
import json
import os
from numpy import *



FRICTION = False  # whether there is friction in the system
SIZE = 10
COLORS = dict(red=(1, 0, 0), yellow=(1, 1, 0), green=(0, 1, 0))
COLOR_TRANSITION = {('green','red'): 'yellow', ('yellow','red'): 'green',
                    ('red','green'): 'red', ('red','yellow'): 'red',
                    (None, 'red'): 'yellow', (None, 'green'): 'red', (None, 'yellow'): 'red'}
BALL_POSITIONS_FILENAME = "ball_positions.npy"


def norm(x):
    return sqrt((x**2).sum())
def sigmoid(x):
    return 1./(1.+exp(-x))


def new_speeds(m1, m2, v1, v2):
    new_v2 = (2*m1*v1 + v2*(m2-m1))/(m1+m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2

# size of bounding box: SIZE X SIZE.


def direction_changed(v1, v2):
    v1_x, v1_y = v1
    v2_x, v2_y = v2
    return (v1_x != v2_x) or (v1_y != v2_y)
    # return ((v1_x > 0) != (v2_x > 0)) or ((v1_y > 0) != (v2_y > 0))


def bounce_n(T=128, n=2, color_shift=None, r=None, m=None):
    if r is None:
        r = array([4.0]*n)
    if m is None:
        m = array([1]*n)

    # r is to be rather small.
    X = zeros((T, n, 2), dtype='float')
    V = zeros((T, n, 2), dtype='float')
    C = zeros((T, n, 3), dtype='float')

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
    curr_colors = ['red'] * n
    prev_colors = [None] * n
    for t in range(T):
        # for how long do we show small simulation

        for i in range(n):
            curr_color = curr_colors[i]
            prev_color = prev_colors[i]

            X[t, i] = x[i]
            V[t, i] = v[i]

            if color_shift:
                # Slowly add blue to all the ball colors
                adjusted_color = (*COLORS[curr_color][:-1], COLORS[curr_color][-1]+t/T)
            else:
                adjusted_color = COLORS[curr_color]
            C[t, i] = adjusted_color

            if t>0 and direction_changed(V[t-1,i], V[t,i]):
                next_color = COLOR_TRANSITION[(prev_color, curr_color)]
                prev_colors[i] = curr_color
                curr_colors[i] = next_color

        for mu in range(int(1/eps)):

            for i in range(n):
                x[i] += eps*v[i]

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
                        # if (x[i][0] > 0) and (x[i][0] < size) and (x[i][1] > 0) and (x[i][1] < size):
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


def bounce_vec(save_dir, res, n=2, color_shift=False, T=128, chunk_size=100, r=None, m=None):
    if r is None:
        r = array([1.2]*n)
    x, v, c = bounce_n(T, n, color_shift, r, m)  # x: <seq_len x num_balls x 2>
    if save_dir is not None:
        with open(f"{save_dir}/{BALL_POSITIONS_FILENAME}", 'wb') as f:
            save(f, x)
    return matricize(x, v, res, chunk_size, save_dir, r, c)


def unsigmoid(x): return log(x) - log(1-x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="datasets/ball_stn")
    parser.add_argument("--T_total", type=int, default=1000000)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_balls", type=int, default=2)
    parser.add_argument("--color_shift", action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    train_path, test_path = f"{args.save_dir}/train/{args.seed}", f"{args.save_dir}/test/{args.seed}"
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    assert args.T_total % args.chunk_size == 0

    with open(f"{args.save_dir}/config.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # save train data
    bounce_vec(train_path, args.resolution, args.num_balls, args.color_shift,
               args.T_total, args.chunk_size)

    # save test data
    bounce_vec(test_path, args.resolution, args.num_balls, args.color_shift,
               args.T_total, args.chunk_size)
