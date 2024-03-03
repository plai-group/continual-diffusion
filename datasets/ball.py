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


"""
shape_std=shape
def shape(A):
    if isinstance(A, ndarray):
        return shape_std(A)
    else:
        return A.shape()

size_std = size
def size(A):
    if isinstance(A, ndarray):
        return size_std(A)
    else:
        return A.size()

det = linalg.det

def new_speeds(m1, m2, v1, v2):
    new_v2 = (2*m1*v1 + v2*(m2-m1))/(m1+m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2


def norm(x): return sqrt((x**2).sum())
def sigmoid(x):        return 1./(1.+exp(-x))

SIZE=10
# size of bounding box: SIZE X SIZE.

def bounce_n(T=128, n=2, r=None, m=None):
    if r is None: r=array([1.2]*n)
    if m is None: m=array([1]*n)
    # r is to be rather small.
    X=zeros((T, n, 2), dtype='float')
    v = random.randn(n,2)
    v = v / norm(v)*.5
    good_config=False
    while not good_config:
        x = 2+ random.rand(n,2)*8
        good_config=True
        for i in range(n):
            for z in range(2):
                if x[i][z]-r[i]<0:      good_config=False
                if x[i][z]+r[i]>SIZE:     good_config=False

        # that's the main part.
        for i in range(n):
            for j in range(i):
                if norm(x[i]-x[j])<r[i]+r[j]:
                    good_config=False


    eps = .5
    for t in range(T):
        # for how long do we show small simulation

        for i in range(n):
            X[t,i]=x[i]

        for mu in range(int(1/eps)):

            for i in range(n):
                x[i]+=eps*v[i]

            for i in range(n):
                for z in range(2):
                    if x[i][z]-r[i]<0:  v[i][z]= abs(v[i][z]) # want positive
                    if x[i][z]+r[i]>SIZE: v[i][z]=-abs(v[i][z]) # want negative


            for i in range(n):
                for j in range(i):
                    if norm(x[i]-x[j])<r[i]+r[j]:
                        # the bouncing off part:
                        w    = x[i]-x[j]
                        w    = w / norm(w)

                        v_i  = dot(w.T_totalranspose(),v[i])
                        v_j  = dot(w.T_totalranspose(),v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)

                        v[i]+= w*(new_v_i - v_i)
                        v[j]+= w*(new_v_j - v_j)

    return X

def ar(x,y,z):
    return z/2+arange(x,y,z,dtype='float')

def matricize(X,res,r=None):

    T, n= shape(X)[0:2]
    if r is None: r=array([1.2]*n)

    A=zeros((T,res,res), dtype='float')

    [I, J]=meshgrid(ar(0,1,1./res)*SIZE, ar(0,1,1./res)*SIZE)

    for t in range(T):
        for i in range(n):
            A[t]+= exp(-(  ((I-X[t,i,0])**2+(J-X[t,i,1])**2)/(r[i]**2)  )**4    )

        A[t][A[t]>1]=1
    return A

def bounce_mat(res, n=2, T=128, r =None):
    if r is None: r=array([1.2]*n)
    x = bounce_n(T,n,r);
    A = matricize(x,res,r)
    return A

def bounce_vec(res, n=2, T=128, r =None, m =None):
    if r is None: r=array([1.2]*n)
    x = bounce_n(T,n,r,m);
    V = matricize(x,res,r)
    return V.reshape(T, res, res)
"""

FRICTION = False  # whether there is friction in the system
SIZE = 10


def norm(x): return sqrt((x**2).sum())
def sigmoid(x): return 1./(1.+exp(-x))


def new_speeds(m1, m2, v1, v2):
    new_v2 = (2*m1*v1 + v2*(m2-m1))/(m1+m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2

# size of bounding box: SIZE X SIZE.


def bounce_n(T=128, n=2, r=None, m=None):
    if r is None:
        r = array([4.0]*n)
    if m is None:
        m = array([1]*n)
    # r is to be rather small.
    X = zeros((T, n, 2), dtype='float')
    V = zeros((T, n, 2), dtype='float')
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

    return X, V


def ar(x, y, z):
    return z/2+arange(x, y, z, dtype='float')


def matricize(X, V, res, r=None):

    T, n = shape(X)[0:2]
    if r is None:
        r = array([4.0]*n)

    A = zeros((T, res, res, 3), dtype='float')

    [I, J] = meshgrid(ar(0, 1, 1./res)*SIZE, ar(0, 1, 1./res)*SIZE)

    for t in range(T):
        for i in range(n):
            A[t, :, :, 1] += exp(-(((I-X[t, i, 0])**2+(J-X[t, i, 1])**2)/(r[i]**2))**4)
            A[t, :, :, 0] += 1.0 * (V[t, i, 0] + .5) * exp(-(((I-X[t, i, 0])**2+(J-X[t, i, 1])**2)/(r[i]**2))**4)
            A[t, :, :, 2] += 1.0 * (V[t, i, 1] + .5) * exp(-(((I-X[t, i, 0])**2+(J-X[t, i, 1])**2)/(r[i]**2))**4)

        A[t, :, :, 0][A[t, :, :, 0] > 1] = 1
        A[t, :, :, 1][A[t, :, :, 1] > 1] = 1
        A[t, :, :, 2][A[t, :, :, 2] > 1] = 1
    return A


def bounce_mat(res, n=2, T=128, r=None):
    if r is None:
        r = array([1.2]*n)
    x = bounce_n(T, n, r)
    A = matricize(x, res, r)
    return A


def bounce_vec(res, n=2, T=128, r=None, m=None):
    if r is None:
        r = array([1.2]*n)
    x, v = bounce_n(T, n, r, m)  # x: <seq_len x num_balls x 2>
    return matricize(x, v, res, r)


def unsigmoid(x): return log(x) - log(1-x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="datasets/ball")
    parser.add_argument("--T_total", type=int, default=100000)
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--num_balls", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    train_path, test_path = f"{args.save_dir}/train", f"{args.save_dir}/test"
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    assert args.T_total % args.chunk_size == 0

    with open(f"{args.save_dir}/config.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    train_data = bounce_vec(args.resolution, args.num_balls, args.T_total)
    for n in range(args.T_total // args.chunk_size):
        with open(f"{train_path}/{n}.npy", 'wb') as f:
            save(f, train_data[args.chunk_size * n:args.chunk_size * (n + 1)])

    test_data = bounce_vec(args.resolution, args.num_balls, args.T_total)
    for n in range(args.T_total // args.chunk_size):
        with open(f"{test_path}/{n}.npy", 'wb') as f:
            save(f, test_data[args.chunk_size * n:args.chunk_size * (n + 1)])
