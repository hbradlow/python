"""
Register point clouds to each other
"""

from __future__ import division
from jds_utils import math_utils
import numpy as np
import scipy.spatial.distance as ssd
from numpy import cos, sin, pi
import scipy.optimize as opt
from copy import deepcopy
from lfd import warping
from brett2.ros_utils import Marker
from jds_utils import conversions
import matplotlib
import matplotlib.pyplot as plt
import tps
from svds import svds

def tps_kernel(dist, dim):
    if dim == 1:
        return dist**3
    elif dim == 2:
        K = dist**2 * np.log(dist)
        np.putmask(K, dist==0, 0)
        return K
    elif dim == 3:
        return -dist
    else:
        raise NotImplementedError

class Transformation(object):
    n_params = 0
    def fit(self, x_nd, y_nd):
        raise NotImplementedError
    def transform_points(self, x_nd):
        raise NotImplementedError
    def transform_frames(self, x_nd, rot_nkk, orthogonalize=True):
        raise NotImplementedError

    def jacobian(self, x_d, epsilon=0.0001):
        x0 = np.asfarray(x_d)
        f0 = self.transform_points(x0)
        jac = np.zeros(len(x0), len(f0))
        dx = np.zeros(len(x0))
        for i in range(len(x0)):
            dx[i] = epsilon
            jac[i] = (self.transform_points(x0+dx) - f0) / epsilon
            dx[i] = 0.
        return jac.transpose()

    def approx_deriv(self, x_d, dx_d, dist=None):
        x_1d, dx_1d = np.asarray([x_d]), np.asarray([dx_d])
        if dist is None: dist = np.linalg.norm(dx_d)
        return (self.transform_points(x_1d)[0] - self.transform_points(x_1d + dx_1d)[0])/float(dist)

class ThinPlateSpline(Transformation):
    def __init__(self, d):
        self.n = 0
        self.d = d
        self.x_na = np.zeros((0,d))
        self.lin_ag = np.eye(d)
        self.trans_g = np.zeros(d)
        self.w_ng = np.zeros((0,d))
        
    @staticmethod
    def identity(d):
        return ThinPlateSpline(d)
        
    def fit(self, x_na, y_ng, bend_coef=.1, rot_coef = 1e-5, wt_n=None, verbose=True):
        """
        x_nd: source cloud
        y_nd: target cloud
        smoothing: penalize non-affine part
        angular_spring: penalize rotation
        wt_n: weight the points        
        """
        self.n, self.d = n,d = x_na.shape
        self.lin_ag, self.trans_g, self.w_ng = tps.tps_fit2(x_na, y_ng, bend_coef, rot_coef, wt_n)
        self.x_na = x_na
        
    def transform_points(self, x_md):
        return tps.tps_eval(x_md, self.lin_ag, self.trans_g, self.w_ng, self.x_na)
    
    def transform_frames(self, x_ma, rot_mda, orthogonalize="cross"):
        """
        orthogonalize: none, svd, qr
        """
        m,d = x_ma.shape
        
        grad_mga = tps.tps_grad(x_ma, self.lin_ag, self.trans_g, self.w_ng, self.x_na)
        newrot_mdg = (rot_mda[:,:,None,:] * grad_mga[:,None,:,:]).sum(axis=3)
        # mdg               md_a                  m_ga
        
        newpt_mg = tps.tps_eval(x_ma, self.lin_ag, self.trans_g, self.w_ng, self.x_na)


        if orthogonalize == "qr": 
            newrot_mdg =  orthogonalize3_qr(newrot_mdg)
        elif orthogonalize == "svd":
            newrot_mdg = orthogonalize3_svd(newrot_mdg)
        elif orthogonalize == "cross":
            newrot_mdg = orthogonalize3_cross(newrot_mdg)
        elif orthogonalize == "none":
            pass
        else: raise Exception("unknown orthogonalization method %s"%orthogonalize)
        return newpt_mg, newrot_mdg

class CompositeTransformation(Transformation):
    def __init__(self, init_fn):
      self.fns = [init_fn]

    def compose_with(self, f):
      self.fns.append(f)

    def get_last_fn(self):
      return self.fns[-1]

    def fit(self, x_nd, y_nd):
      assert False

    def transform_points(self, x_md):
      for f in self.fns:
        x_md = f.transform_points(x_md)
      return x_md

    def transform_frames(self, x_md, rot_mdd, orthogonalize=True):
      for f in self.fns:
        x_md, rot_mdd = f.transform_frames(x_md, rot_mdd, orthogonalize)
      return x_md, rot_mdd

class ThinPlateSplineFixedRot(ThinPlateSpline):
    """same as ThinPlateSpline except during fitting, affine part is a fixed rotation around z axis"""
    
    def __init__(self, rot):
        #raise NotImplementedError
        ThinPlateSpline.__init__(self)
        assert rot.ndim == 2 and rot.shape[0] == rot.shape[1]
        self.n = 0
        self.d = rot.shape[0]
        self.x_nd = np.zeros((0,self.d))
        self.w_nd = np.zeros((0,self.d))
        self.a_Dd = np.eye(self.d+1,self.d)
        self.a_Dd[:self.d, :] = rot
    
    def fit(self, x_nd, y_nd, smoothing=.1, wt_n=None, verbose=True):
        """
        x_nd: source cloud
        y_nd: target cloud
        smoothing: penalize non-affine part
        angular_spring: penalize rotation
        wt_n: weight the points        
        """
        self.n, self.d = n,d = x_nd.shape
        
        dists_tri = ssd.pdist(x_nd)
        K_nn = ssd.squareform(tps_kernel(dists_tri, d))
        
        
        
        if wt_n is None:
            wt_n = np.ones(n)

        reg_nn = smoothing * np.diag(1/(wt_n+1e-6))
        A = np.r_[
            np.c_[K_nn + reg_nn,       np.ones((n,1))],
            np.c_[np.ones((1,n)),      0]]
        b = np.r_[y_nd - np.dot(x_nd, self.a_Dd[:d,:]), np.zeros((1,d))]

        
        coeffs = np.linalg.lstsq(A, b)[0]
        self.w_nd = coeffs[:n,:]
        self.a_Dd[-1] = coeffs[n,:]
        rotation_cost = 0
        
        self.x_nd = x_nd

        residual_cost = (wt_n[:,None] * ((y_nd - self.transform_points(x_nd))**2).sum(axis=1)).sum()
        curvature_cost = smoothing * np.trace(np.dot(self.w_nd.T, np.dot(K_nn, self.w_nd)))
        self.cost = residual_cost + curvature_cost + rotation_cost
        self.residual_cost, self.curvature_cost, self.rotation_cost = residual_cost, curvature_cost, rotation_cost
        if verbose:
            print "cost = residual + curvature + rotation"
            print " %.3g = %.3g + %.3g + %.3g"%(self.cost, residual_cost, curvature_cost, rotation_cost)
            print "affine transform:\n", self.a_Dd    

def plot_warped_grid_2d(f, mins, maxes, grid_res=None, flipax = True):
    import matplotlib.pyplot as plt
    import matplotlib
    xmin, ymin = mins
    xmax, ymax = maxes
    ncoarse = 10
    nfine = 30
    
    if grid_res is None:
        xcoarse = np.linspace(xmin, xmax, ncoarse)
        ycoarse = np.linspace(ymin, ymax, ncoarse)
    else:
        xcoarse = np.arange(xmin, xmax, grid_res)
        ycoarse = np.arange(ymin, ymax, grid_res)
    xfine = np.linspace(xmin, xmax, nfine)
    yfine = np.linspace(ymin, ymax, nfine)
    
    lines = []
    
    sgn = -1 if flipax else 1
    
    for x in xcoarse:
        xy = np.zeros((nfine, 2))
        xy[:,0] = x
        xy[:,1] = yfine
        lines.append(f(xy)[:,::sgn])

    for y in ycoarse:
        xy = np.zeros((nfine, 2))
        xy[:,0] = xfine
        xy[:,1] = y
        lines.append(f(xy)[:,::sgn])        
    
    lc = matplotlib.collections.LineCollection(lines,colors='gray',lw=2)
    ax = plt.gca()
    ax.add_collection(lc)
    plt.draw()
    
def plot_correspondence(x_nd, y_nd):
    lines = np.array(zip(x_nd, y_nd))
    lc = matplotlib.collections.LineCollection(lines)
    ax = plt.gca()
    ax.add_collection(lc)
    plt.draw()

def plot_correspondence_3d(x_nd, y_nd):
    # if center:
    #     c = np.mean(x_nd, axis=0)
    #     x_nd, y_nd = x_nd - c[None,:], y_nd - c[None,:]
    lines = np.array(zip(x_nd, y_nd))
    from mpl_toolkits import mplot3d
    from mpl_toolkits.mplot3d import art3d
    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    ax.add_collection3d(art3d.Line3DCollection(lines))
    ax.scatter(x_nd[:,0], x_nd[:,1], x_nd[:,2], c='g')
    ax.scatter(y_nd[:,0], y_nd[:,1], y_nd[:,2], c='r')
    plt.draw()

def loglinspace(a,b,n):
    "n numbers between a to b (inclusive) with constant ratio between consecutive numbers"
    return np.exp(np.linspace(np.log(a),np.log(b),n))    

class Globals:
    handles = []
    rviz = None
    @staticmethod
    def setup():
        if Globals.rviz is None:
            from brett2.ros_utils import RvizWrapper
            Globals.rviz = RvizWrapper.create()
            import time
            time.sleep(.2)
    
def tps_rpm(x_nd, y_md, n_iter = 5, reg_init = .1, reg_final = .001, rad_init = .2, rad_final = .001, plotting = False, verbose=True, f_init = None, return_full = False, ns_prefix='tpsrpm', p=.2, interactive=False):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    reg_init/reg_final: regularization on curvature
    rad_init/rad_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    """
    n,d = x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)
    f = ThinPlateSpline.identity(d)
    #f.trans_g = y_md.mean(axis=0) - x_nd.mean(axis=0)
    
    for i in xrange(n_iter):
        if f.d==2 and i%plotting==0:
            import matplotlib.pyplot as plt
            plt.clf()
        if i==0 and f_init is not None:
            xwarped_nd = f_init(x_nd)
            print xwarped_nd.max(axis=0)
        else:
            xwarped_nd = f.transform_points(x_nd)

        if plotting and i%plotting == 0 and interactive:
            raw_input('press enter to continue')

        # targ_nd = find_targets(x_nd, y_md, corr_opts = dict(r = rads[i], p = .1))
        corr_nm = calc_correspondence_matrix(xwarped_nd, y_md, r=rads[i], p=p)        

        wt_n = corr_nm.sum(axis=1)
        
        #goodn = wt_n > .1
        goodn = wt_n > -.1

        targ_Nd = np.dot(corr_nm[goodn, :]/wt_n[goodn][:,None], y_md)

        assert (corr_nm[goodn,:].sum(axis=1) == wt_n[goodn]).all()

        #print "warning: changed angular spring!"        
        f.fit(x_nd[goodn], targ_Nd, bend_coef = regs[i], wt_n = wt_n[goodn], rot_coef = 10*regs[i], verbose=verbose)

        if plotting and i%plotting==0:
            plot_orig_and_warped_clouds(f.transform_points, x_nd, y_md, ns_prefix=ns_prefix)
            targ_pose_array = conversions.array_to_pose_array(targ_Nd, 'base_footprint')
            Globals.handles.append(Globals.rviz.draw_curve(targ_pose_array,rgba=(1,1,0,1),type=Marker.CUBE_LIST, ns=ns_prefix+'_targ_Nd'))

    if plotting:
        plot_correspondence_3d(x_nd[goodn], targ_Nd)

    if return_full:
        info = {}
        info["corr_nm"] = corr_nm
        info["goodn"] = goodn
        info["x_Nd"] = x_nd[goodn,:]
        info["targ_Nd"] = targ_Nd
        info["wt_N"] = wt_n[goodn]
        return f, info
    else:
        return f

def tps_rpm_zrot(x_nd, y_md, n_iter = 5, reg_init = .1, reg_final = .001, rad_init = .2, rad_final = .001, plotting = False, verbose=True):
    """
    Do tps_rpm algorithm for each z angle rotation
    Then don't reestimate affine part in tps optimization
    """
    n,d = x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)
    zrots = np.linspace(-np.pi/3, pi/3, 7)

    displacement = np.median(y_md,axis=0) - np.median(x_nd, axis=0) 
    
    costs = []
    
    if plotting:
        plotter = FuncPlotter()
        
    zrot2func = {}
    def fit0(zrot):
        f = ThinPlateSplineFixedRot(np.array([[cos(zrot), sin(zrot), 0], [-sin(zrot), cos(zrot), 0], [0, 0, 1]]))        
        f.a_Dd[3, :3] = displacement
        
        for i in xrange(n_iter):
    
            if f.d==2 and i%plotting==0: 
                import matplotlib.pyplot as plt            
                plt.clf()
    
            xwarped_nd = f.transform_points(x_nd)
            corr_nm = calc_correspondence_matrix(xwarped_nd, y_md, r=rads[i], p=.2)
            
            wt_n = corr_nm.sum(axis=1)
            targ_nd = np.dot(corr_nm/wt_n[:,None], y_md)
            f.fit(x_nd, targ_nd, regs[i], wt_n = wt_n, verbose=verbose)
    
            if plotting and i%plotting==0:
                plot_orig_and_warped_clouds(f.transform_points, x_nd, y_md)

        print "zrot: %.3e, cost: %.3e,  meancorr: %.3e"%(zrot, f.cost, corr_nm.mean())
        cost = abs(zrot)/2 + f.cost
        if plotting: plotter.addpt(zrot, cost)
        zrot2func[zrot] = f
        return cost
    
    
    costs = [fit0(zrot) for zrot in zrots]
    i_best = np.argmin(costs)

    import scipy.optimize as so
    zspacing = zrots[1] - zrots[0]  
    print (zrots[i_best] - zspacing, zrots[i_best], zrots[i_best] + zspacing)
    zrot_best = so.golden(fit0, brack = (zrots[i_best] - zspacing, zrots[i_best], zrots[i_best] + zspacing), tol = .15)
    f_best = zrot2func[zrot_best]
    return f_best

def fit_affine_by_tpsrpm(x_nd, y_md):
  # use tps-rpm to get correspondences, then fit an affine transformation by least squares
  f = tps_rpm(x_nd, y_md, reg_init=.1,reg_final=.1,n_iter=100,verbose=False, plotting=False)
  partners = f.corr.argmax(axis=1)

  M = np.zeros((3*len(x_nd), 12))
  for i, pt in enumerate(x_nd):
    M[3*i,:3] = M[3*i+1,3:6] = M[3*i+2,6:9] = pt
    M[3*i,9] = M[3*i+1,10] = M[3*i+2,11] = 1
  q = y_md[partners].reshape((-1, 1))
  print 'M', M, 'q', q
  S, c, _, _ = np.linalg.lstsq(M, q)
  print 'residues', c
  print 'S', S, S[0,:], S[1,:], S[2,:]
  return S, S[0:9].reshape(3,3), (S[9:].T)[0]

class FuncPlotter(object):
    def __init__(self, fignum = 1):
        self.fignum = fignum
        plt.figure(fignum)
        plt.clf()
        self.xs = []
        self.ys = []
    def addpt(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        self.plotme()
    def plotme(self):
        plt.figure(self.fignum)
        plt.clf()
        plt.plot(self.xs, self.ys,'x')
        plt.draw()
 
def plot_orig_and_warped_clouds(f, x_nd, y_md, res=.1, d=3, ns_prefix='tpsrpm'):
    if d==2:
        import matplotlib.pyplot as plt
        plt.plot(x_nd[:,1], x_nd[:,0],'r.')
        plt.plot(y_md[:,1], y_md[:,0], 'b.')
    pred = f(x_nd)
    if d==2:
        plt.plot(pred[:,1], pred[:,0], 'g.')
    if d == 2:
        plot_warped_grid_2d(f, x_nd.min(axis=0), x_nd.max(axis=0))
        plt.ginput()
    elif d == 3:
        
        Globals.setup()

        mins = x_nd.min(axis=0)
        maxes = x_nd.max(axis=0)
        mins -= np.array([.1, .1, .01])
        maxes += np.array([.1, .1, .01])
        Globals.handles = warping.draw_grid(Globals.rviz, f, mins, maxes, 'base_footprint', xres=res, yres=res, zres=-1, ns=ns_prefix+'_grid')
        orig_pose_array = conversions.array_to_pose_array(x_nd, "base_footprint")
        target_pose_array = conversions.array_to_pose_array(y_md, "base_footprint")
        warped_pose_array = conversions.array_to_pose_array(f(x_nd), 'base_footprint')
        Globals.handles.append(Globals.rviz.draw_curve(orig_pose_array,rgba=(1,0,0,.4),type=Marker.CUBE_LIST, ns=ns_prefix+'_demo_cloud'))
        Globals.handles.append(Globals.rviz.draw_curve(target_pose_array,rgba=(0,0,1,.4),type=Marker.CUBE_LIST, ns=ns_prefix+'_target_cloud'))
        Globals.handles.append(Globals.rviz.draw_curve(warped_pose_array,rgba=(0,1,0,.4),type=Marker.CUBE_LIST, ns=ns_prefix+'_warped_cloud'))

        
def find_targets(x_md, y_nd, corr_opts):
    """finds correspondence matrix, and then for each point in source cloud,
    find the weighted average of its "partners" in the target cloud"""

    corr_mn = calc_correspondence_matrix(x_md, y_nd, **corr_opts)
    # corr_mn = M.match(x_md, y_nd)
    # corr_mn = corr_mn / corr_mn.sum(axis=1)[:,None]
    return np.dot(corr_mn, y_nd)        
    
def calc_correspondence_matrix(x_nd, y_md, r, p, n_iter=20):
    "sinkhorn procedure. see tps-rpm paper"    
    n = x_nd.shape[0]
    m = y_md.shape[0]
    dist_nm = ssd.cdist(x_nd, y_md,'euclidean')
    prob_nm = np.exp(-dist_nm / r)


    A = np.empty((n+1,m+1))
    A[:n, :m] = prob_nm
    A[n,:] = p
    A[:,m] = p

    before = A.copy()


    for _ in xrange(n_iter):
        A /= A.sum(axis=0)[None,:]  # cols sum to n/m
        A /= A.sum(axis=1)[:,None]
        
    #print "row sums:", A.sum(axis=1)
    #print "col sums/ratio:", A.sum(axis=0)/(n/m)

    #print "after/before max", (A/before).max()

    A = np.sqrt(A*before)

    return A[:n, :m]

def nan2zero(x):
    np.putmask(x, np.isnan(x), 0)
    return x

def orthogonalize3_cross(mats_n33):
    "turns each matrix into a rotation"

    x_n3 = mats_n33[:,:,0]
    y_n3 = mats_n33[:,:,1]
    # z_n3 = mats_n33[:,:,2]
    
    xnew_n3 = math_utils.normr(x_n3)
    znew_n3 = math_utils.normr(np.cross(xnew_n3, y_n3))
    ynew_n3 = math_utils.normr(np.cross(znew_n3, xnew_n3))
    
    return np.concatenate([xnew_n3[:,:,None], ynew_n3[:,:,None], znew_n3[:,:,None]],2)
    

def orthogonalize3_svd(x_k33):
    u_k33, s_k3, v_k33 = svds(x_k33)
    return (u_k33[:,:,:,None] * v_k33[:,None,:,:]).sum(axis=3)
def orthogonalize3_qr(x_k33):
    raise NotImplementedError

    
    
def fit_score(src, targ, dist_param):
    "how good of a partial match is src to targ"
    sqdists = ssd.cdist(src, targ,'sqeuclidean')
    return -np.exp(-sqdists/dist_param**2).sum()


class Rigid2d(Transformation):
    n_params = 3
    tx = 0
    ty = 0
    angle = 0

    def set_params(self, params):
        self.tx, self.ty, self.angle = params        
    def get_params(self):
        return np.r_[self.tx, self.ty, self.angle]

    def fit(self, x_n3, y_n3):        

        trans_init = (y_n3.mean(axis=0) - x_n3.mean(axis=0))[:2]
        rot_inits = [0]

        self_copy = deepcopy(self)
        def f(params):
            self_copy.set_params(params)
            xmapped_n3 = self_copy.transform_points(x_n3)
            return fit_score(xmapped_n3, y_n3,.5)


        vals_params = []
        for rot_init in rot_inits:
            opt_params, opt_val, _, _, _ = opt.fmin_cg(f, np.r_[trans_init, rot_init],full_output=True)
            vals_params.append((opt_val, opt_params))


        best_val, best_params = min(vals_params, key = lambda x:x[0])
        print "best_params:", best_params
        self.set_params(best_params)
        self.objective = best_val

    def transform_points(self, x_n3):

        a = self.angle
        rot_mat = np.array([[cos(a), sin(a), 0],
                            [-sin(a), cos(a), 0],
                            [0,     0,      1]])

        return np.dot(x_n3, rot_mat.T) + np.r_[self.tx, self.ty, 0][None,:]
        
        
    def transform_frames(self, x_n3, rot_n33, orthogonalize=True):
        newx_n3 = self.transform_points(x_n3)
        
        a = self.angle
        j = np.array([[cos(a), sin(a), 0],
                      [-sin(a), cos(a), 0],
                      [0,     0,      1]])        
        
        newrot_n33 = np.array([np.dot(j, rot) for rot in rot_n33])
        return newx_n3, newrot_n33


class Rigid3d(Transformation):
    n_params = 6
    tx = 0
    ty = 0
    tz = 0
    alpha = 0
    beta = 0
    gamma = 0

    def set_params(self, params):
        self.tx, self.ty, self.tz, self.alpha, self.beta, self.gamma = params
    def get_params(self):
        return np.r_[self.tx, self.ty, self.tz, self.alpha, self.beta, self.gamma]

    def fit(self, x_n3, y_n3, prev_params=None):

        if prev_params is None:
            trans_init = y_n3.mean(axis=0) - x_n3.mean(axis=0)
            rot_inits = [(0, 0, 0)]
        else:
            trans_init = prev_params[:3]
            rot_inits = [prev_params[3:]]

        self_copy = deepcopy(self)
        def f(params):
            self_copy.set_params(params)
            xmapped_n3 = self_copy.transform_points(x_n3)
            return fit_score(xmapped_n3, y_n3,.5)


        vals_params = []
        for rot_init in rot_inits:
            opt_params, opt_val, _, _, _ = opt.fmin_cg(f, np.r_[trans_init, rot_init],full_output=True)
            vals_params.append((opt_val, opt_params))


        best_val, best_params = min(vals_params, key = lambda x:x[0])
        print "best_params:", best_params
        self.set_params(best_params)
        self.objective = best_val

    def rot_mat(self):
        a, b, c = self.alpha, self.beta, self.gamma
        return np.array([[cos(a)*cos(b), cos(a)*sin(b)*sin(c)-sin(a)*cos(c), cos(a)*sin(b)*cos(c)+sin(a)*sin(c)],
                         [sin(a)*cos(b), sin(a)*sin(b)*sin(c)+cos(a)*cos(c), sin(a)*sin(b)*cos(c)-cos(a)*sin(c)],
                         [-sin(b), cos(b)*sin(c), cos(b)*cos(c)]])

    def transform_points(self, x_n3):
        return np.dot(x_n3, self.rot_mat().T) + np.r_[self.tx, self.ty, self.tz][None,:]

    def transform_frames(self, x_n3, rot_n33, orthogonalize=True):
        newx_n3 = self.transform_points(x_n3)
        j = self.rot_mat()
        newrot_n33 = np.array([np.dot(j, rot) for rot in rot_n33])
        return newx_n3, newrot_n33


class Translation2d(Transformation):
    n_params = 2
    tx = 0
    ty = 0

    def set_params(self, params):
        self.tx, self.ty = params        
    def get_params(self, params):
        return np.r_[self.tx, self.ty]

    def fit(self, x_n3, y_n3):        

        trans_init = (y_n3.mean(axis=0) - x_n3.mean(axis=0))[:2]

        self_copy = deepcopy(self)
        def f(params):
            self_copy.set_params(params)
            xmapped_n3 = self_copy.transform_points(x_n3)
            return fit_score(xmapped_n3, y_n3,.5)


        # vals_params = []
        # for rot_init in rot_inits:
        #     opt_params, opt_val, _, _, _ = opt.fmin_cg(f, np.r_[trans_init],full_output=True)
        #     vals_params.append((opt_val, opt_params))
        # 
        # 
        # best_val, best_params = min(vals_params, key = lambda x:x[0])
        
        best_params, best_val, _,_,_ = opt.fmin_cg(f, np.r_[trans_init], full_output=True)
        
        print "best_params:", best_params
        self.set_params(best_params)
        self.objective = best_val

    def transform_points(self, x_n3):

        return x_n3 + np.r_[self.tx, self.ty, 0][None,:]


    def transform_frames(self, x_n3, rot_n33, orthogonalize=True):
        newx_n3 = self.transform_points(x_n3)

        return newx_n3, rot_n33

class Translation3d(Transformation):
    n_params = 3
    tx = 0
    ty = 0
    tz = 0

    def set_params(self, params):
        self.tx, self.ty, self.tz = params
    def get_params(self):
        return np.r_[self.tx, self.ty, self.tz]

    def fit(self, x_n3, y_n3):        

        trans_init = (y_n3.mean(axis=0) - x_n3.mean(axis=0))[:3]

        self_copy = deepcopy(self)
        def f(params):
            self_copy.set_params(params)
            xmapped_n3 = self_copy.transform_points(x_n3)
            return fit_score(xmapped_n3, y_n3,.5)


        # vals_params = []
        # for rot_init in rot_inits:
        #     opt_params, opt_val, _, _, _ = opt.fmin_cg(f, np.r_[trans_init],full_output=True)
        #     vals_params.append((opt_val, opt_params))
        # 
        # 
        # best_val, best_params = min(vals_params, key = lambda x:x[0])
        
        best_params, best_val, _,_,_ = opt.fmin_cg(f, np.r_[trans_init], full_output=True)
        
        print "best_params:", best_params
        self.set_params(best_params)
        self.objective = best_val

    def transform_points(self, x_n3):

        return x_n3 + np.r_[self.tx, self.ty, self.tz][None,:]


    def transform_frames(self, x_n3, rot_n33, orthogonalize=True):
        newx_n3 = self.transform_points(x_n3)

        return newx_n3, rot_n33
