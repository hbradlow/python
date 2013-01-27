import curve_perturbation as cpert

import rospy
from brett2 import ros_utils, PR2
from brett2.ros_utils import Marker
import numpy as np
import os.path as osp
import h5py
import warping
import yaml
import lfd
import rope_initialization as ri
from jds_utils import conversions

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task")
parser.add_argument("--seg", default=None)
parser.add_argument('--from_log', type=str, default='')
parser.add_argument('--s', action='store', type=float, default=0.001, help='variance of randomness to introduce to the b-spline control points')
parser.add_argument('--n', action='store', type=int, default=1, help='num samples to draw')
parser.add_argument('--const_radius', type=bool, default=True, help='don\'t use gaussian around each control point with variance s (just use a random angle, with constant radius sqrt(s)')
args = parser.parse_args()

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

class Globals:
    pr2 = None
    rviz = None
    handles = []
    isinstance(pr2, PR2.PR2)
    isinstance(rviz, ros_utils.RvizWrapper)

    def __init__(self): raise

    @staticmethod
    def setup():
        if Globals.pr2 is None:
            Globals.pr2 = PR2.PR2.create()
            #execute_task.load_table()
        if Globals.rviz is None: Globals.rviz = ros_utils.RvizWrapper.create()
        #Globals.table_height = rospy.get_param("table_height")


def read_demos():
    data_dir = osp.join(osp.dirname(lfd.__file__), "data")
    with open(osp.join(data_dir, "knot_demos.yaml"),"r") as fh:
        task_info = yaml.load(fh)
    H5FILE = osp.join(data_dir, task_info[args.task]["db_file"])
    demos_file = h5py.File(H5FILE,"r")
    rospy.loginfo("loading demos into memory")
    return warping.group_to_dict(demos_file)

def read_cloud(demos):
    seg = args.seg or select_from_list(demos.keys())
    return np.squeeze(np.asarray(demos[seg]["cloud_xyz"]))

def draw_rope(rope, width, rgba, ns):
    Globals.handles.append(Globals.rviz.draw_curve(
        conversions.array_to_pose_array(rope, "base_footprint"),
        width=width, rgba=rgba, type=Marker.LINE_STRIP, ns=ns
    ))

def draw_cloud(cloud, width, rgba, ns):
    Globals.handles.append(Globals.rviz.draw_curve(
        conversions.array_to_pose_array(np.squeeze(cloud), "base_footprint"),
        rgba=rgba, type=Marker.CUBE_LIST, ns=ns
    ))

def main():
    Globals.handles = []
    if rospy.get_name() == '/unnamed':
        rospy.init_node("publish_pert_rope", disable_signals=True)
    Globals.setup()

    rospy.sleep(1)

    if args.from_log:
        import logtool
        cloud_xyz = logtool.get_first_seen_cloud(logtool.read_log(args.from_log))
    else:
        cloud_xyz = read_cloud(read_demos())
    rope = ri.find_path_through_point_cloud(cloud_xyz)
    prope = cpert.perturb_curve(rope, args.s, args.const_radius)

    draw_cloud(cloud_xyz, width=0.01, rgba=(1, 0, 1, .5), ns='publish_pert_rope_cloud_orig')
    draw_rope(rope, width=0.01, rgba=(1, 1, 0, 1), ns='publish_pert_rope_orig')
    draw_rope(prope, width=0.01, rgba=(0, 1, 1, 1), ns='publish_pert_rope_perturbed')

    rospy.loginfo('Publishing...')
    while not rospy.is_shutdown():
        for h in Globals.handles:
            h.pub.publish(h.marker)
        rospy.sleep(1)

if __name__ == '__main__':
    main()
