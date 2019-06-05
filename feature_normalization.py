
import numpy as np
import sklearn.preprocessing as sk
#from sklearn.preprocessing import MinMaxScaler
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
data = [[100,50, 2], [-100,50, 6], [0, 50,10], [1,50, 18]]
#data= [[100,50, 2]]
a=sk.minmax_scale(data, feature_range=(0, 1), axis=1, copy=True)
#print(a)
def scale_linear_bycolumn(rawpoints, high=1.0, low=0.0):
    mins = np.min(rawpoints, axis=1)
    maxs = np.max(rawpoints, axis=1)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)
k=scale_linear_bycolumn(data, high=1.0, low=0.0)
print(k)
'''
npmax=np.max(data,axis=0)
npmin=np.min(data,axis=0)
print(npmax,npmin)
rng = npmax - npmin
high=1
low=0
scaled_points = high - (((high - low) * (npmax - data)) / rng)
print(scaled_points)

'''

'''
scaler = MinMaxScaler()
print(scaler.fit(data))
MinMaxScaler(copy=True, feature_range=(0, 1))
print(scaler.data_max_)

print(scaler.transform(data))
print(scaler.transform([[2, 2]]))
'''

def scale_array(dat, out_range=(0, 1)):
    domain = [np.min(dat, axis=0), np.max(dat, axis=0)]

    def interp(x):
        return out_range[0] * (1.0 - x) + out_range[1] * x

    def uninterp(x):
        b = 0
        if (domain[1] - domain[0]) != 0:
            b = domain[1] - domain[0]
        else:
            b =  1.0 / domain[1]
        return (x - domain[0]) / b

    return interp(uninterp(dat))


#print(scale_array(np.array([[-6.28753302e-01 -2.69029002e-01 -1.79053937e+00  6.75913568e-01
#  -2.14641853e-01  6.99124561e-01  1.52792540e+00]], dtype=np.float)))
# Gives: [-1., 0., 1.]
#print(scale_array(np.array([-3, -2, -1], dtype=np.float)))
# Gives: [-1., 0., 1.]
