import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5.0,4.0)
plt.rcParams['image.interpolation']= 'nearest'
plt.rcParams['image.cmap']='gray'

#zero padding
#args
#X - (m, nh,nw,c)
#pad - numerical(1..9)
def zero_pad(X,pad):
    X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values = (0,0))

np.random.seed(1)
x = np.random.randn(4,3,3,2)
#4 images, 3by3 size, 2 channels
x_pad = zero_pad(x,2)

print("x.shape",x.shape)
print("x_pad.shape",x_pad.shape)
#plotting
fig, axarr = plt.subplots(1,2)
axarr[0].set_title('x')
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0,:,:,0])

#single step convolution
#args
#aslice- (f,f,nc)
#w-(f,f,nc)
#b-(1,1,1)
def conv_single_step(a_slice_prev, W,b):
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z = np.add(Z,b)
    return Z

np.random.seed(1)
a_slice_prev = np.random.randn(4,4,3)
#slice of dims 4by4 and 3 channel
W = np.random.randn(4,4,3)
b = np.random.randn(1,1,1)
Z = conv_single_step(a_slice_prev, W, b)
print("Z=",Z)


#cnn - forward pass
def conv_forward(A_prev, W, b, hparameters):
    (m,n_H_prev, n_W_prev,n_c_prev) = A_prev.shape
    (f,f,n_c_prev, n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']

    #now calc op dims
    n_H = int(((n_H_prev-f+2*pad)/stride)+1)
    n_W = int(((n_W_prev-f+2*pad)/stride)+1)
    Z = np.zeros((m,n_H,n_W,n_C))
    A_prev_pad = zero_pad(A_prev,pad)
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vs = stride*h
                    ve = vs+f
                    hs = stride*w
                    he = hs+f
                    a_slice_prev = a_prev_pad[vs:ve,hs:he,:]
                    Z[i,h,w,c] = conv_single_step(a_slice_prev,W[:,:,:,c],b[:,:,:,c])
    assert(Z.shape == (m,n_h,n_w,n_C))
    cache = (A_prev , W,b, hparameters)
    return Z, cache

np.random.seed(1)
A_prev = np.random.randn(10,4,43)
W = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)
hparameters = {"pad":2, "stride":2}
Z, cache_conv = conv_forward(A_prev, W, b, hparameter)
print(np.mean(Z))


#pooling layer
def pool_forward(A_prev, hparameters, mode="max"):
    (m,n_H_prev, n_W_prev,n_C_prev) = A_prev.shape
    f = hparameters['f']
    stride = hparameters['stride']
    n_H = int(1+(n_H_prev - f)/stride)
    n_W = int(1+(n_W_prev - f)/stride)
    n_C = n_C_prev
    A = np.zeros((m,n_H,n_W,n_C))
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vs = stride*h
                    ve = vs+f
                    hs = stride*w
                    he = hs+f
                    a_prev_slice = A_prev[i,vs:ve,hs:he,c]
                    if mode = 'max':
                        A[i,h,w,c] = np.max(a_prev_slice)
                    else:
                        A[i,h,w,c] = np.average(a_prev_slice)
    cache = (A_prev, hparameters)
    assert(A.shape == (m, n_H, n_W, n_C))
    return A, cache


np.random.seed(1)
A_prev = np.random.randn(2,4,4,3)
hparameters = {'stride':2,'f':3}
A,cache=pool_forward(A_prev, hparameters)
print(A)
A,cache = pool_forward(A_prev, hparameters, mode = "average")
print(A)
#back propagation should be taken care of estimators
