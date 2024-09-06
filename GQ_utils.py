# -*- coding: utf-8 -*-

"""
Created on August 2024
@author: Magued Al-Aghbary & Mohamed Sobh
"""

# Helper functions

import pandas as pd
from IPython.display import display
import numpy as np
from scipy.spatial import cKDTree as KDTree
import pyproj as proj
from pyproj import Transformer
from scipy import stats, interpolate

milli= 0.001



def table_grid_search(clf, all_columns=False, all_ranks=False, save=True):

    # Convert the cross validated results in a DataFrame ordered by `rank_test_score` and `mean_fit_time`.
    # As it is frequent to have more than one combination with the same max score,
    # the one with the least mean fit time SHALL appear first.
    cv_results = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score', 'mean_fit_time'])

    # Reorder
    columns = cv_results.columns.tolist()
    # rank_test_score first, mean_test_score second and std_test_score third
    columns = columns[-1:] + columns[-3:-1] + columns[:-3]
    cv_results = cv_results[columns]

    if save:
        cv_results.to_csv('--'.join(cv_results['params'][0].keys()) + '.csv', index=True, index_label='Id')

    # Unless all_columns are True, drop not wanted columns: params, std_* split*
    if not all_columns:
        cv_results.drop('params', axis='columns', inplace=True)
        cv_results.drop(list(cv_results.filter(regex='^std_.*')), axis='columns', inplace=True)
        cv_results.drop(list(cv_results.filter(regex='^split.*')), axis='columns', inplace=True)

    # Unless all_ranks are True, filter out those rows which have rank equal to one
    if not all_ranks:
        cv_results = cv_results[cv_results['rank_test_score'] == 1]
        cv_results.drop('rank_test_score', axis = 'columns', inplace = True)
        cv_results = cv_results.style.hide_index()

    display(cv_results)


# Haversine arc distance
def distance(lat1, lon1, lat2, lon2):
    '''
    Haversine formula returns distance between pairs of coordinates.
    coordinates as numpy arrays, lists or real
    The haversine formula determines the great-circle distance between
    two points on a sphere given their longitudes and latitudes
    '''
    p = 0.017453292519943295 # pi/180
    a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p)*np.cos(lat2*p) * (1-np.cos((lon2-lon1)*p)) / 2
    return 12742.0176 * np.arcsin(np.sqrt(a)) # returns in km



#function to make inverse distance weight interpolation

N = 10000
Ndim = 2
Nask = N  # N Nask 1e5: 24 sec 2d, 27 sec 3d on mac g4 ppc
Nnear = 8  # 8 2d, 11 3d => 5 % chance one-sided -- Wendel, mathoverflow.com
leafsize = 10
eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
p = 1  # weights ~ 1 / distance**p
cycle = .25
seed = 1

class Invdisttree:

    def __init__( self, X, z, leafsize=10, stat=0 ):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree( X, leafsize=leafsize )  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None;

    def __call__( self, q, nnear=6, eps=0, p=1, weights=None ):
            # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query( q, k=nnear, eps=eps )
        interpol = np.zeros( (len(self.distances),) + np.shape(self.z[0]) )
        jinterpol = 0
        for dist, ix in zip( self.distances, self.ix ):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot( w, self.z[ix] )
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1  else interpol[0]


# adapted from a script agrid https://github.com/TobbeTripitaka/agrid
# paper can be accssed from https://www.doi.org/10.5334/JORS.287


verbose = True
def _check_if_in(xx, yy, margin=2):
    '''Generate an array of the condition that coordinates
    are within the model or not.
    xx = list or array of x values
    yy = list or array of y values
    margin = extra cells added to mitigate extrapolation of
    interpolated values along the frame
    returns boolean array True for points within the frame.
    '''
    res = [xx.max()- xx.min(), yy.max(), yy.min() ]
    x_min = xx.min() - margin * res[0]
    x_max = xx.max() + margin * res[0]
    y_min = yy.min() - margin * res[1]
    y_max = yy.max() + margin * res[1]
    return (xx > x_min) & (xx < x_max) & (yy > y_min) & (yy < y_max)

def _set_meridian( x_array, center_at_0=True):
    '''
    Sloppy function to change longitude values from [0..360] to [-180..180]
    x_array :   Numpy array with longitude values (X)
    center_at_0 : Bool select direction of conversion.
    lon=(((lon + 180) % 360) - 180)
    '''
    if center_at_0:
        x_array[x_array > 180] = x_array[x_array > 180] - 360
    else:
        x_array[x_array < 0] = x_array[x_array < 0] + 360
    return x_array


def read_numpy(   i = 0,
                  j = 1,
                  k = 2,
                  data = None,
                  ds=None,
                  interpol='linear',
                  crs_from=None,
                  crs_to=None,
                  use_dask=None,
                  dask_chunks=None,
                  pad_around=False,
                  only_frame=True,
                  set_center=False,
                  verbose=False,
                  z_factor=1,
                  **kwargs):
        '''Read numpy array and interpolate to grid.

        Keyword arguments:
        x,y,z numpy arrays of same size, eg, A[0,:], A[1,:], A[2,:]
        Returns numpy array


        kwargs to interpolation
        '''


        if data is not None:
            x = data[:,i]
            y = data[:,j]
            z = data[:,k]

        assert(np.shape(x)==np.shape(y)==np.shape(z)), 'x, y, and z must have the same shape.'





        if crs_from is None:
            crs_from = crs_from

        if crs_to is None:
            crs_to = crs_to


        if verbose:
            print('Shape:', np.shape(x))

        if z_factor is not 1:
            z *= z_factor


        # Set longitude, case from 0 to -360 insetad of -180 to 180
        if set_center:
            x = _set_meridian(x)

        transformer = Transformer.from_crs(crs_from, crs_to)
        xv, yv =   transformer.transform(x, y)

        #xv, yv = proj.transform(proj.Proj(crs_src),
        #                        proj.Proj(crs), x, y)


        n = z.size
        zi = np.reshape(z, (n))
        xi = np.reshape(xv, (n))
        yi = np.reshape(yv, (n))

        # Check and interpolate only elements in the frame
        if only_frame:
            is_in = _check_if_in(xi, yi)
            xi = xi[is_in]
            yi = yi[is_in]
            zi = zi[is_in]

        ny = len(ds.Y)
        nx = len(ds.X)
        nn = (ny, nx)

        if interpol == "IDW":
            X = np.array([xv, yv]).T
            coords = np.stack([  ds.coords['XV'].values.flatten() ,
                               ds.coords['YV'].values.flatten() ], axis=1)
            invdisttree = Invdisttree( X,
                                      z.reshape(-1,1), leafsize=leafsize, stat=1 )
            arr = invdisttree( coords,
                              nnear=Nnear, eps=eps, p=p )
            arr = arr.ravel().reshape(nn)


        else:
            arr = interpolate.griddata((xi, yi),
                                       zi,
                                       (ds.coords['XV'],
                                        ds.coords['YV']),
                                       method=interpol,
                                       **kwargs)

        if pad_around:
            for i in range(ep_max)[::-1]:
                arr[:, i][np.isnan(arr[:, i])] = arr[
                    :, i + 1][np.isnan(arr[:, i])]
                arr[:, -i][np.isnan(arr[:, -i])] = arr[:, -
                                                       i - 1][np.isnan(arr[:, -i])]
                arr[i, :][np.isnan(arr[i, :])] = arr[
                    i + 1, :][np.isnan(arr[i, :])]
                arr[-i, :][np.isnan(arr[-i, :])] = arr[-i -
                                                       1, :][np.isnan(arr[-i, :])]


        if use_dask:
            if dask_chunks is None:
                dask_chunks = (nx // chunk_n,) * arr.ndim
            arr = da.from_array(arr, chunks=dask_chunks)

        return arr



# object of pyproj and inistilize it with WGS84 model
geod = proj.Geod(ellps='WGS84' )

km = 1000
# this function get the distance in km and avoid uncessary unpaking
def geodinv(lon1, lat1, lon2, lat2):
    _, _, ds_ = geod.inv(lon1, lat1, lon2, lat2)
    return ds_/km


# Define function to correct magnetic values
def mag_log(data, C=400, clip_min=-1, clip_max=1):
    return np.clip(np.sign(data)*np.log(1+np.abs(data)/C), clip_min, clip_max)
