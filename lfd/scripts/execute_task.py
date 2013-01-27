#!/usr/bin/env python

"""
Perform deformable object manipulation task, where data is stored in some h5 file
Currently works for tying an overhand knot or folding up a laid-out towel
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task")
parser.add_argument("--human_select_demo", action="store_true")
parser.add_argument("--prompt_before_motion", action="store_true")
parser.add_argument("--count_steps",action="store_true")
parser.add_argument("--hard_table",action="store_true")
parser.add_argument("--test",action="store_true")
parser.add_argument("--use_tracking", action="store_true")
parser.add_argument("--reg_final", type=float, default=.025)
parser.add_argument("--use_rigid", action="store_true")
parser.add_argument("--cloud_topic", type=str, default="/preprocessor/points")
parser.add_argument("--delay_before_look", type=float, default=-1)
parser.add_argument("--use_nr", action="store_true", help="use nonrigidy tps")
parser.add_argument("--log_name", type=str, default="/tmp")
parser.add_argument("--use_base", action="store_true")
args = parser.parse_args()

import roslib; roslib.load_manifest("smach_ros")
import smach
import lfd
from lfd import registration, trajectory_library, warping, recognition, lfd_traj
from kinematics import kinbodies
from jds_utils.yes_or_no import yes_or_no
import sensor_msgs.msg
import geometry_msgs.msg as gm
import trajectory_msgs.msg as tm
import rospy
import os
import os.path as osp
from brett2 import ros_utils, PR2
from brett2.ros_utils import Marker
import numpy as np
from jds_utils import conversions
from jds_image_proc.clouds import voxel_downsample
import jds_utils.math_utils as mu
try:
    from jds_image_proc.alpha_shapes import get_concave_hull
except Exception:
    pass
try:
    from bulletsim_msgs.msg import TrackedObject
except Exception:
    pass
import h5py
from collections import defaultdict
import yaml
import time, datetime
import pickle
import atexit

########## INITIALIZATION ##########
class ExecLog(object):
    def __init__(self):
        self.name = str(np.random.randint(9999999999))
        self.events = []

        dirname = osp.join(osp.dirname(lfd.__file__), 'data', 'logs', args.log_name)
        if not osp.exists(dirname):
            rospy.loginfo('Creating log directory %s', dirname)
            os.makedirs(dirname)
        self.filename = osp.join(
            dirname,
            'execute_task_%d_%s_%s.log' % (os.getpid(), datetime.datetime.now().isoformat(), self.name)
        )
        rospy.loginfo('Will write event log to %s', self.filename)

    def log(self, state, msg, data):
        pickle.dumps(data) # just make sure that the data is pickle-able
        self.events.append({
            'time': time.time(),
            'state': state,
            'msg': msg,
            'data': data,
        })

    def dump(self):
        if len(self.events) == 0:
            return
        with open(self.filename, 'w') as f:
            pickle.dump(self.events, f, protocol=2)
        rospy.loginfo('Wrote %d events to %s', len(self.events), self.filename)
        self.events = []

ELOG = ExecLog()
atexit.register(ELOG.dump)

ELOG.log('global', 'args', args)

data_dir = osp.join(osp.dirname(lfd.__file__), "data")
with open(osp.join(data_dir, "knot_demos.yaml"),"r") as fh:
    task_info = yaml.load(fh)

DS_LENGTH = .02
DS_METHOD = "voxel"
if args.task.startswith("fold"):
    DS_METHOD="hull"
#else:
    #DS_METHOD = "voxel"

H5FILE = osp.join(data_dir, task_info[args.task]["db_file"])
demos_file = h5py.File(H5FILE,"r")
rospy.loginfo("loading demos into memory")
demos = warping.group_to_dict(demos_file)

if args.test:
    lfd_traj.ALWAYS_FAKE_SUCCESS = True


########## UTILITIES ##########
def draw_table():
    aabb = Globals.pr2.robot.GetEnv().GetKinBody("table").GetLinks()[0].ComputeAABB()
    ps =gm.PoseStamped()
    ps.header.frame_id = "base_footprint"
    ps.pose.position = gm.Point(*aabb.pos())
    ps.pose.orientation = gm.Quaternion(0,0,0,1)
    Globals.handles.append(Globals.rviz.draw_marker(ps, type=Marker.CUBE, scale = aabb.extents()*2, id = 24019,rgba = (1,0,0,.25)))

def load_table():
    table_bounds = map(float, rospy.get_param("table_bounds").split())
    kinbodies.create_box_from_bounds(Globals.pr2.env,table_bounds, name="table")

def increment_pose(arm, translation):
    cur_pose = arm.get_pose_matrix("base_footprint", "r_gripper_tool_frame")
    new_pose = cur_pose.copy()
    new_pose[:3,3] += translation
    arm.goto_pose_matrix(new_pose, "base_footprint", "r_gripper_tool_frame")

def select_from_list(list):
    strlist = [str(item) for item in list]
    while True:
        print "choose from the following options:"
        print " ".join("(%s)"%item for item in strlist)
        resp = raw_input("?) ")
        if resp not in strlist:
            print "invalid response. try again."
        else:
            return list[strlist.index(resp)]

def clipinplace(x,lo,hi):
    np.clip(x,lo,hi,out=x)

def downsample(xyz):
    if DS_METHOD == "voxel":
        xyz_ds, ds_inds = voxel_downsample(xyz, DS_LENGTH, return_inds = True)
    elif DS_METHOD == "hull":
        xyz = np.squeeze(xyz)
        _, inds = get_concave_hull(xyz[:,0:2],.05)
        xyz_ds = xyz[inds]
        ds_inds = [[i] for i in inds]
    return xyz_ds, ds_inds

def alternate(arr1, arr2):
    assert arr1.shape == arr2.shape
    out = np.zeros((2*arr1.shape[0], arr1.shape[1]),arr1.dtype)
    out[0::2] = arr1
    out[1::2] = arr2
    return out

def calc_seg_cost(seg_name, xyz_new_ds, dists_new):
    candidate_demo = demos[seg_name]
    xyz_demo_ds = np.squeeze(candidate_demo["cloud_xyz_ds"])
    dists_demo = candidate_demo["geodesic_dists"]
  #  cost = recognition.calc_match_score(xyz_new_ds, xyz_demo_ds, dists0 = dists_new, dists1 = dists_demo)
    cost = recognition.calc_match_score(xyz_demo_ds, xyz_new_ds, dists0 = dists_demo, dists1 = dists_new)
    print "seg_name: %s. cost: %s"%(seg_name, cost)
    return cost, seg_name

class Globals:
    pr2 = None
    rviz = None
    handles = []
    isinstance(pr2, PR2.PR2)
    isinstance(rviz, ros_utils.RvizWrapper)
    if args.count_steps: stage = 0

    def __init__(self): raise

    @staticmethod
    def setup():
        if Globals.pr2 is None:
            Globals.pr2 = PR2.PR2.create(rave_only=args.test)
            if not args.test: load_table()
        if Globals.rviz is None: Globals.rviz = ros_utils.RvizWrapper.create()
        Globals.table_height = rospy.get_param("table_height")


########## STATES ##########

class LookAtObject(smach.State):
    def __init__(self):
        smach.State.__init__(self,
            outcomes = ["success", "failure"],
            input_keys = [],
            output_keys = ["points"] # object points
        )


    def execute(self,userdata):
        """
        - move head to the right place
        - get a point cloud
        returns: success, failure
        """
        Globals.handles = []
        draw_table()

        Globals.pr2.rgrip.set_angle(.08)
        Globals.pr2.lgrip.set_angle(.08)
        Globals.pr2.join_all()
        if not args.use_tracking:
            Globals.pr2.larm.goto_posture('side')
            Globals.pr2.rarm.goto_posture('side')
        else:
            try: increment_pose(Globals.pr2.rarm, [0,0,.02])
            except PR2.IKFail: print "couldn't raise right arm"
            try: increment_pose(Globals.pr2.larm, [0,0,.02])
            except PR2.IKFail: print "couldn't raise left arm"
        Globals.pr2.rgrip.set_angle(.08)
        Globals.pr2.lgrip.set_angle(.08)
        Globals.pr2.join_all()

        if args.delay_before_look > 0:
            rospy.loginfo('sleeping for %f secs before looking', args.delay_before_look)
            rospy.sleep(args.delay_before_look)

        if args.test:
            xyz = np.squeeze(np.asarray(demos[select_from_list(demos.keys())]["cloud_xyz"]))
        elif args.use_tracking:
            msg = rospy.wait_for_message("/tracker/object", TrackedObject)
            xyz = [(pt.x, pt.y, pt.z) for pt in msg.rope.nodes]
        else:
            msg = rospy.wait_for_message(args.cloud_topic, sensor_msgs.msg.PointCloud2)
            xyz, rgb = ros_utils.pc2xyzrgb(msg)
            xyz = xyz.reshape(-1,3)
            xyz = ros_utils.transform_points(xyz, Globals.pr2.tf_listener, "base_footprint", msg.header.frame_id)

        ELOG.log('LookAtObject', 'xyz', xyz)
        userdata.points = xyz

        return "success"

class SelectTrajectory(smach.State):
    f = None
    def __init__(self):
        smach.State.__init__(self,
            outcomes = ["done", "not_done","failure", "move_base"],
            input_keys = ["points"],
            output_keys = ["trajectory", "base_offset"])

        rospy.loginfo("preprocessing demo point clouds...")
        for (_,demo) in demos.items():
            demo["cloud_xyz_ds"], ds_inds = downsample(demo["cloud_xyz"])
            demo["cloud_xyz"] = np.squeeze(demo["cloud_xyz"])
            demo["geodesic_dists"] = recognition.calc_geodesic_distances_downsampled_old(demo["cloud_xyz"], demo["cloud_xyz_ds"], ds_inds)

        if args.count_steps:
            self.count2segnames = defaultdict(list)
            for (name, demo) in demos.items():
                self.count2segnames[int(demo["seg_index"])].append(name)

        rospy.loginfo("done")

    def execute(self,userdata):
        """
        - lookup closest trajectory from database
        - if it's a terminal state, we're done
        - warp it based on the current rope
        returns: done, not_done, failure
        """
        xyz_new = np.squeeze(np.asarray(userdata.points))
        #if args.obj == "cloth": xyz_new = voxel_downsample(xyz_new, .025)

        xyz_new_ds, ds_inds = downsample(xyz_new)
        dists_new = recognition.calc_geodesic_distances_downsampled_old(xyz_new,xyz_new_ds, ds_inds)
        ELOG.log('SelectTrajectory', 'xyz_new', xyz_new)
        ELOG.log('SelectTrajectory', 'xyz_new_ds', xyz_new_ds)
        ELOG.log('SelectTrajectory', 'dists_new', dists_new)

        if args.count_steps: candidate_demo_names = self.count2segnames[Globals.stage]
        else: candidate_demo_names = demos.keys()

        from joblib import parallel

        costs_names = parallel.Parallel(n_jobs=-2)(parallel.delayed(calc_seg_cost)(seg_name, xyz_new_ds, dists_new) for seg_name in candidate_demo_names)
        #costs_names = [calc_seg_cost(seg_name, xyz_new_ds, dists_new) for seg_name in candidate_demo_names]
        #costs_names = [calc_seg_cost(seg_name) for seg_name in candidate_demo_names]

        ELOG.log('SelectTrajectory', 'costs_names', costs_names)
        _, best_name = min(costs_names)

        if args.human_select_demo:
            print 'Calculated best demo:', best_name
            best_name = None
            while best_name not in demos:
                print 'Select demo from', demos.keys()
                best_name = raw_input('> ')

        ELOG.log('SelectTrajectory', 'best_name', best_name)
        best_demo = demos[best_name]
        if best_demo["done"]:
            rospy.loginfo("best demo was a 'done' state")
            return "done"

        best_demo = demos[best_name]
        rospy.loginfo("best segment name: %s", best_name)
        xyz_demo_ds = best_demo["cloud_xyz_ds"]
        ELOG.log('SelectTrajectory', 'xyz_demo_ds', xyz_demo_ds)

        if args.test: n_iter = 21
        else: n_iter = 101
        if args.use_rigid:
            self.f = registration.Translation2d()
            self.f.fit(xyz_demo_ds, xyz_new_ds)
            ELOG.log('SelectTrajectory', 'f', self.f)
        else:
            self.f, info = registration.tps_rpm(xyz_demo_ds, xyz_new_ds, plotting = 20, reg_init=1,reg_final=.01,n_iter=n_iter,verbose=False, return_full=True)#, interactive=True)
            ELOG.log('SelectTrajectory', 'f', self.f)
            ELOG.log('SelectTrajectory', 'f_info', info)
            if args.use_nr:
                rospy.loginfo('Using nonrigidity costs')
                from lfd import tps
                import scipy.spatial.distance as ssd
                pts_grip = []
                for lr in "lr":
                  if best_demo["arms_used"] in ["b", lr]:
                    pts_grip.extend(best_demo["%s_gripper_tool_frame"%lr]["position"])
                pts_grip = np.array(pts_grip)
                dist_to_rope = ssd.cdist(pts_grip, xyz_demo_ds).min(axis=1)
                pts_grip_near_rope = pts_grip[dist_to_rope < .04,:]
                pts_rigid = voxel_downsample(pts_grip_near_rope, .01)
                self.f.lin_ag, self.f.trans_g, self.f.w_ng, self.f.x_na = tps.tps_nr_fit_enhanced(info["x_Nd"], info["targ_Nd"], 0.01, pts_rigid, 0.001, method="newton", plotting=5)
            # print 'correspondences', self.f.corr_nm


        #################### Generate new trajectory ##################

        #### Plot original and warped point clouds #######
        # orig_pose_array = conversions.array_to_pose_array(np.squeeze(best_demo["cloud_xyz_ds"]), "base_footprint")
        # warped_pose_array = conversions.array_to_pose_array(self.f.transform_points(np.squeeze(best_demo["cloud_xyz_ds"])), "base_footprint")
        # Globals.handles.append(Globals.rviz.draw_curve(orig_pose_array,rgba=(1,0,0,1),id=19024,type=Marker.CUBE_LIST))
        # Globals.handles.append(Globals.rviz.draw_curve(warped_pose_array,rgba=(0,1,0,1),id=2983,type=Marker.CUBE_LIST))

        #### Plot grid ########
        mins = np.squeeze(best_demo["cloud_xyz"]).min(axis=0)
        maxes = np.squeeze(best_demo["cloud_xyz"]).max(axis=0)
        mins[2] -= .1
        maxes[2] += .1
        grid_handle = warping.draw_grid(Globals.rviz, self.f.transform_points, mins, maxes, 'base_footprint')
        Globals.handles.append(grid_handle)

        #### Actually generate the trajectory ###########
        warped_demo = warping.transform_demo_with_fingertips(self.f, best_demo)
        # if yes_or_no('dump warped demo?'):
        #     import pickle
        #     fname = '/tmp/warped_demo_' + str(np.random.randint(9999999999)) + '.pkl'
        #     with open(fname, 'w') as f:
        #         pickle.dump(warped_demo, f)
        #     print 'saved to', fname
        ELOG.log('SelectTrajectory', 'warped_demo', warped_demo)

        def make_traj(warped_demo, inds=None, xyz_offset=0):
            traj = {}
            total_feas_inds = 0
            all_feas = True
            for lr in "lr":
                leftright = {"l":"left","r":"right"}[lr]
                if best_demo["arms_used"] in [lr, "b"]:
                    if args.hard_table:
                        clipinplace(warped_demo["l_gripper_tool_frame"]["position"][:,2],Globals.table_height+.032,np.inf)
                        clipinplace(warped_demo["r_gripper_tool_frame"]["position"][:,2],Globals.table_height+.032,np.inf)
                    pos = warped_demo["%s_gripper_tool_frame"%lr]["position"]
                    ori = warped_demo["%s_gripper_tool_frame"%lr]["orientation"]
                    if inds is not None:
                        pos, ori = pos[inds], ori[inds]
                    arm_traj, feas_inds = lfd_traj.make_joint_traj_by_graph_search(
                        pos + xyz_offset,
                        ori,
                        Globals.pr2.robot.GetManipulator("%sarm"%leftright),
                        "%s_gripper_tool_frame"%lr,
                        check_collisions=True)
                    traj["%s_arm"%lr] = arm_traj
                    traj["%s_arm_feas_inds"%lr] = feas_inds
                    total_feas_inds += len(feas_inds)
                    all_feas = all_feas and len(feas_inds) == len(arm_traj)
                    rospy.loginfo("%s arm: %i of %i points feasible", leftright, len(feas_inds), len(arm_traj))
            return traj, total_feas_inds, all_feas

        # Check if we need to move the base for reachability
        base_offset = np.array([0, 0, 0])
        if args.use_base:
            # First figure out how much we need to move the base to maximize feasible points
            OFFSET = 0.1
            XYZ_OFFSETS = np.array([[0., 0., 0.], [-OFFSET, 0, 0], [OFFSET, 0, 0], [0, -OFFSET, 0], [0, OFFSET, 0]])

            # demo_len = 0
            # for lr in "lr":
            #     if "%s_gripper_tool_frame"%lr in warped_demo:
            #         demo_len = len(warped_demo["%s_gripper_tool_frame"%lr]["position"])
            #         break
            #inds_to_check = np.arange(0, demo_len, demo_len/40) # only check a few of the trajectory points
            inds_to_check = lfd_traj.where_near_rope(best_demo, xyz_demo_ds)
            print 'checking inds', inds_to_check

            need_to_move_base = False
            best_feas_inds, best_xyz_offset = -1, None
            for xyz_offset in XYZ_OFFSETS:
                _, n_feas_inds, all_feas = make_traj(warped_demo, inds=inds_to_check, xyz_offset=xyz_offset)
                rospy.loginfo('Cloud offset %s has feas inds %d', str(xyz_offset), n_feas_inds)
                if n_feas_inds > best_feas_inds:
                    best_feas_inds, best_xyz_offset = n_feas_inds, xyz_offset
                if all_feas: break
            if np.linalg.norm(best_xyz_offset) > 0.01:
                need_to_move_base = True
            base_offset = -best_xyz_offset
            rospy.loginfo('Best base offset: %s, with %d feas inds', str(base_offset), best_feas_inds)
            raw_input('continue?')

            # Move the base
            if need_to_move_base:
                userdata.base_offset = base_offset
                return 'move_base'

        Globals.pr2.update_rave()

        # calculate joint trajectory using IK
        trajectory = make_traj(warped_demo)[0]
        # fill in gripper/grab stuff
        for lr in "lr":
            leftright = {"l":"left","r":"right"}[lr]
            if best_demo["arms_used"] in [lr, "b"]:
                print trajectory["%s_arm_feas_inds"%lr]
                if len(trajectory["%s_arm_feas_inds"%lr]) == 0: return "failure"
                trajectory["%s_grab"%lr] = best_demo["%s_gripper_joint"%lr] < .07
                trajectory["%s_gripper"%lr] = warped_demo["%s_gripper_joint"%lr]
                trajectory["%s_gripper"%lr][trajectory["%s_grab"%lr]] = 0
        # for lr in "lr":
        #     leftright = {"l":"left","r":"right"}[lr]
        #     if best_demo["arms_used"] in [lr, "b"]:
        #         if args.hard_table:
        #             clipinplace(warped_demo["l_gripper_tool_frame"]["position"][:,2],Globals.table_height+.032,np.inf)
        #             clipinplace(warped_demo["r_gripper_tool_frame"]["position"][:,2],Globals.table_height+.032,np.inf)
        #         arm_traj, feas_inds = lfd_traj.make_joint_traj_by_graph_search(
        #             warped_demo["%s_gripper_tool_frame"%lr]["position"],
        #             warped_demo["%s_gripper_tool_frame"%lr]["orientation"],
        #             Globals.pr2.robot.GetManipulator("%sarm"%leftright),
        #             "%s_gripper_tool_frame"%lr,
        #             check_collisions=True
        #         )
        #         if len(feas_inds) == 0: return "failure"
        #         trajectory["%s_arm"%lr] = arm_traj
        #         trajectory["%s_grab"%lr] = best_demo["%s_gripper_joint"%lr] < .07
        #         trajectory["%s_gripper"%lr] = warped_demo["%s_gripper_joint"%lr]
        #         trajectory["%s_gripper"%lr][trajectory["%s_grab"%lr]] = 0
        # smooth any discontinuities in the arm traj
        for lr in "lr":
            leftright = {"l":"left","r":"right"}[lr]
            if best_demo["arms_used"] in [lr, "b"]:
                trajectory["%s_arm"%lr], discont_times, n_steps = lfd_traj.smooth_disconts(
                    trajectory["%s_arm"%lr],
                    Globals.pr2.env,
                    Globals.pr2.robot.GetManipulator("%sarm"%leftright),
                    "%s_gripper_tool_frame"%lr
                )
                # after smoothing the arm traj, we need to fill in all other trajectories (in both arms)
                other_lr = 'r' if lr == 'l' else 'l'
                if best_demo["arms_used"] in [other_lr, "b"]:
                    trajectory["%s_arm"%other_lr] = lfd_traj.fill_stationary(trajectory["%s_arm"%other_lr], discont_times, n_steps)
                for tmp_lr in 'lr':
                    if best_demo["arms_used"] in [tmp_lr, "b"]:
                        trajectory["%s_grab"%tmp_lr] = lfd_traj.fill_stationary(trajectory["%s_grab"%tmp_lr], discont_times, n_steps)
                        trajectory["%s_gripper"%tmp_lr] = lfd_traj.fill_stationary(trajectory["%s_gripper"%tmp_lr], discont_times, n_steps)
                        trajectory["%s_gripper"%tmp_lr][trajectory["%s_grab"%tmp_lr]] = 0

        # plotting
        for lr in "lr":
            leftright = {"l":"left","r":"right"}[lr]
            if best_demo["arms_used"] in [lr, "b"]:
                # plot warped trajectory
                Globals.handles.append(Globals.rviz.draw_curve(
                  conversions.array_to_pose_array(
                    alternate(warped_demo["%s_gripper_l_finger_tip_link"%lr]["position"], warped_demo["%s_gripper_r_finger_tip_link"%lr]["position"]),
                    "base_footprint"
                  ),
                  width=.001, rgba = (1,0,1,.4), type=Marker.LINE_LIST,
                  ns='warped_finger_traj'
                ))
                # plot original trajectory
                Globals.handles.append(Globals.rviz.draw_curve(
                  conversions.array_to_pose_array(
                    alternate(best_demo["%s_gripper_l_finger_tip_link"%lr]["position"], best_demo["%s_gripper_r_finger_tip_link"%lr]["position"]),
                    "base_footprint"
                  ),
                  width=.001, rgba = (0,1,1,.4), type=Marker.LINE_LIST,
                  ns='demo_finger_traj'
                ))

        ELOG.log('SelectTrajectory', 'trajectory', trajectory)
        userdata.trajectory = trajectory

        if args.prompt_before_motion:
            consent = yes_or_no("trajectory ok?")
        else:
            consent = True

        if consent: return "not_done"
        else: return "failure"


class MoveBase(smach.State):
    def __init__(self):
        smach.State.__init__(self,
            outcomes = ["success"],
            input_keys = ["base_offset"],
            output_keys = [])

    def execute(self, userdata):
        Globals.pr2.update_rave()
        base_offset = userdata.base_offset

        STEPS = 10; TIME = 5.
        xyas = mu.interp2d(np.linspace(0, 1, STEPS), [0, 1], [[0, 0, 0], base_offset])
        rospy.loginfo('Following base trajectory %s', str(xyas))
        ts = np.linspace(0, TIME, STEPS)

        pub = rospy.Publisher("base_traj_controller/command", tm.JointTrajectory)
        #xyacur = np.array(Globals.pr2.base.get_pose("odom_combined"))
        jt = tm.JointTrajectory()
        jt.header.frame_id = "base_footprint"
        for i in xrange(len(xyas)):
            jtp = tm.JointTrajectoryPoint()
            jtp.time_from_start = rospy.Duration(ts[i])
            jtp.positions = xyas[i]#+xyacur
            jt.points.append(jtp)
        pub.publish(jt)

        rospy.sleep(TIME*1.5)
        ELOG.log('MoveBase', 'base_offset', base_offset)
        return 'success'


class ExecuteTrajectory(smach.State):
    def __init__(self):
        """
        - first show trajectory in rviz and see if it's ok
        - then execute it
        returns: success, failure
        """
        smach.State.__init__(self,
            outcomes = ["success", "failure"],
            input_keys = ["trajectory"],
            output_keys = [])


    def execute(self, userdata):
        #if not args.test: draw_table()
        Globals.pr2.update_rave()
        # if yes_or_no('about to execute trajectory. save?'):
        #     import pickle
        #     fname = '/tmp/trajectory_' + str(np.random.randint(9999999999)) + '.pkl'
        #     with open(fname, 'w') as f:
        #         pickle.dump(userdata.trajectory, f)
        #     print 'saved to', fname
        success = lfd_traj.follow_trajectory_with_grabs(Globals.pr2, userdata.trajectory)
        ELOG.log('ExecuteTrajectory', 'success', success)
        #raw_input('done executing segment. press enter to continue')
        if success:
            if args.count_steps: Globals.stage += 1
            return "success"
        else: return "failure"

def make_tie_knot_sm():
    sm = smach.StateMachine(outcomes = ["success", "failure"])
    with sm:
        smach.StateMachine.add("look_at_object", LookAtObject(), transitions = {"success":"select_traj", "failure":"failure"})
        smach.StateMachine.add("select_traj", SelectTrajectory(), transitions = {"done":"success","not_done":"execute_traj", "failure":"failure", "move_base":"move_base"})
        smach.StateMachine.add("move_base", MoveBase(), transitions = {"success":"look_at_object"})
        smach.StateMachine.add("execute_traj", ExecuteTrajectory(), transitions = {"success":"look_at_object","failure":"look_at_object"})
    return sm


if __name__ == "__main__":
    Globals.handles = []
    if rospy.get_name() == '/unnamed':
        rospy.init_node("tie_knot", disable_signals=True)
    Globals.setup()
    #Globals.pr2.torso.go_up()
    #Globals.pr2.head.set_pan_tilt(0, HEAD_TILT)
    if args.use_tracking:
        Globals.pr2.larm.goto_posture('side')
        Globals.pr2.rarm.goto_posture('side')
    Globals.pr2.join_all()
    tie_knot_sm = make_tie_knot_sm()
    tie_knot_sm.execute()
