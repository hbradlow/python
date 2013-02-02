#!/usr/bin/env python

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from jds_utils import conversions
from brett2.ros_utils import Marker

def read_logs(logfiles):
    logs = []
    for logfile in logfiles:
        assert os.path.exists(logfile)
        with open(logfile, 'r') as f:
            logs.append(pickle.load(f))
    return logs

def read_log(logfile):
    return read_logs([logfile])[0]

def get_cloud(log_data, i=-1):
    cloud_xyz = None
    if i == -1:
        for entry in log_data:
            if entry['state'] == 'LookAtObject' and entry['msg'] == 'xyz':
                cloud_xyz = entry['data']
                break
        if cloud_xyz is None:
            raise RuntimeError('cannot find LookAtObject xyz entry')
    else:
        entry = log_data[i]
        #assert entry['state'] == 'LookAtObject' and entry['msg'] == 'xyz'
        cloud_xyz = entry['data']
    return cloud_xyz

def alternate(arr1, arr2):
    assert arr1.shape == arr2.shape
    out = np.zeros((2*arr1.shape[0], arr1.shape[1]),arr1.dtype)
    out[0::2] = arr1
    out[1::2] = arr2
    return out

# def view_cloud(logs):
#     for log_data in logs:
#         cloud_xyz = get_first_seen_cloud(log_data)
#         plt.plot(cloud_xyz[:,0], cloud_xyz[:,1], 'b.')
#     plt.axis('equal')
#     plt.show()

def calc_rope_dist(rope, prope):
    min_len = min(len(rope), len(prope))
    rope, prope = rope[:min_len], prope[:min_len]
    rope = rope - np.mean(rope, axis=0)
    prope = prope - np.mean(prope, axis=0)
    return np.mean(np.sqrt(((rope - prope)**2).sum(axis=1)))

def view_cloud(args, logs):
    import rope_initialization as ri
    if args.calc_dist_vs != -1:
        base_rope = ri.find_path_through_point_cloud_simple(get_cloud(logs[args.calc_dist_vs]))
    for i, log_data in enumerate(logs):
        offset_x = i*args.offset_x if args.offset_x > 0 else 0
        cloud_xyz = get_cloud(log_data, args.i)
        plt.plot(cloud_xyz[:,0] + offset_x, cloud_xyz[:,1], 'b.', alpha=.5)
        if args.fit_rope:
            rope = ri.find_path_through_point_cloud(cloud_xyz)
            plt.plot(rope[:,0] + offset_x, rope[:,1], 'g.-', alpha=.3)

        if i != args.calc_dist_vs and args.calc_dist_vs != -1:
            print 'Dist from', i, 'to', args.calc_dist_vs, '=', min(calc_rope_dist(rope, base_rope), calc_rope_dist(rope, base_rope[::-1]))

    plt.axis('equal')
    plt.show()

def view_warp(args, logs):
    import registration
    assert len(logs) == 1
    log_data = logs[0]
    xyz_demo_ds = get_cloud(log_data, i=args.demo_cloud_idx)
    xyz_new_ds = get_cloud(log_data, i=args.new_cloud_idx)
    f, info = registration.tps_rpm(xyz_demo_ds, xyz_new_ds, plotting=args.use_rviz, reg_init=1, reg_final=.01, n_iter=101, verbose=False, return_full=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    registration.plot_orig_and_warped_clouds(f.transform_points, xyz_demo_ds, xyz_new_ds, res=.05, force_pyplot=True, ax=ax)

    handles = []
    if args.taskname is not None and args.best_seg_name is not None and args.warped_demo_idx != -1:
        from demo_loading import load_demos
        demos = load_demos(args.taskname, 'knot_demos.yaml')
        best_demo = demos[args.best_seg_name]
        warped_demo = log_data[args.warped_demo_idx]['data']
        for lr in "lr":
            if best_demo["arms_used"] in [lr, "b"]:
                # plot original trajectory
                #['r_gripper_joint', 'r_gripper_l_finger_tip_link', 'r_gripper_tool_frame', 'l_gripper_r_finger_tip_link', 'l_gripper_l_finger_tip_link', 'r_gripper_r_finger_tip_link', 'l_gripper_joint', 'l_gripper_tool_frame']
                #orig_traj_pos = best_demo['%s_gripper_r_finger_tip_link'%lr]['position']
                #orig_traj_pos = 0.5 * (best_demo['%s_gripper_l_finger_tip_link'%lr]['position'] + best_demo['%s_gripper_r_finger_tip_link'%lr]['position'])

                orig_traj_pos = best_demo['%s_gripper_tool_frame'%lr]['position']
                ax.plot(orig_traj_pos[:,0], orig_traj_pos[:,1], 'c-', linewidth=2, label='demo traj')
                #ax.plot(orig_traj_pos[0,0], orig_traj_pos[0,1], 'mo', markersize=10)
                # plot warped trajectory
                warped_traj_pos = warped_demo['%s_gripper_tool_frame'%lr]['position']
                ax.plot(warped_traj_pos[:,0], warped_traj_pos[:,1], 'g-', linewidth=2, label='warped traj')

        # rviz plotting
        if args.use_rviz:
            from brett2 import ros_utils
            rviz = ros_utils.RvizWrapper.create()
            for lr in "lr":
                if best_demo["arms_used"] in [lr, "b"]:
                    # plot warped trajectory
                    handles.append(rviz.draw_curve(
                      conversions.array_to_pose_array(
                        #alternate(warped_demo["%s_gripper_l_finger_tip_link"%lr]["position"], warped_demo["%s_gripper_r_finger_tip_link"%lr]["position"]),
                        warped_demo['%s_gripper_tool_frame'%lr]['position'],
                        "base_footprint"
                      ),
                      width=.003, rgba = (0,1,1,1), type=Marker.LINE_STRIP,
                      ns='warped_finger_traj'
                    ))
                    # plot original trajectory
                    handles.append(rviz.draw_curve(
                      conversions.array_to_pose_array(
                        best_demo['%s_gripper_tool_frame'%lr]['position'],
                        #alternate(best_demo["%s_gripper_l_finger_tip_link"%lr]["position"], best_demo["%s_gripper_r_finger_tip_link"%lr]["position"]),
                        "base_footprint"
                      ),
                      width=.003, rgba = (1,0,1,1), type=Marker.LINE_STRIP,
                      ns='demo_finger_traj'
                    ))

    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles, labels)
    #plt.axis('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()
    if args.out is not None:
        fig.savefig(args.out, bbox_inches='tight')

def extract_data(args, logs):
    assert len(logs) == 1
    log_data = logs[0]
    if args.out == '':
        print log_data[args.i]['data']
    else:
        with open(args.out, 'w') as f:
            pickle.dump(log_data[args.i]['data'], f)
        print 'Wrote to', args.out

def print_summary(args, logs):
    for i, log_data in enumerate(logs):
        print '==========', 'Event summary for log', i, '=========='
        for j, x in enumerate(log_data):
            print j, x['state'], x['msg'], x['data'] if x['msg'] == 'best_name' else ''
        print


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, action='append', help='log file (specify multiple times for multiple log files)')
    parser.add_argument('--use_rviz', action='store_true')
    subparsers = parser.add_subparsers()

    parser_view_cloud = subparsers.add_parser('view_cloud')
    parser_view_cloud.add_argument('--offset_x', type=float, default=-1)
    parser_view_cloud.add_argument('--calc_dist_vs', type=int, default=-1)
    parser_view_cloud.add_argument('-i', type=int, default=-1, help='index of entry in log')
    parser_view_cloud.add_argument('--fit_rope', action='store_true', help='fit a rope')
    parser_view_cloud.set_defaults(func=view_cloud)

    parser_extract_data = subparsers.add_parser('extract_data')
    parser_extract_data.add_argument('-i', type=int, required=True)
    parser_extract_data.add_argument('--out', type=str, default='')
    parser_extract_data.set_defaults(func=extract_data)

    parser_print_summary = subparsers.add_parser('print_summary')
    parser_print_summary.set_defaults(func=print_summary)

    parser_view_warp = subparsers.add_parser('view_warp')
    parser_view_warp.add_argument('--demo_cloud_idx', type=int, default=-1)
    parser_view_warp.add_argument('--new_cloud_idx', type=int, default=-1)
    parser_view_warp.add_argument('--taskname', type=str, default=None)
    parser_view_warp.add_argument('--best_seg_name', type=str, default=None)
    parser_view_warp.add_argument('--warped_demo_idx', type=int, default=-1)
    parser_view_warp.add_argument('--out', type=str, default=None, help='output file, if desired')
    parser_view_warp.set_defaults(func=view_warp)

    args = parser.parse_args()
    logs = read_logs(args.log)
    if args.use_rviz:
        import rospy
        if rospy.get_name() == '/unnamed':
            rospy.init_node("tie_knot", disable_signals=True)
    args.func(args, logs)


if __name__ == '__main__':
    main()
