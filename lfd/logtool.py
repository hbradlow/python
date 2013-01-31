#!/usr/bin/env python

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

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
        assert entry['state'] == 'LookAtObject' and entry['msg'] == 'xyz'
        cloud_xyz = entry['data']
    return cloud_xyz

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

def view_rope(args, logs):
    import rope_initialization as ri
    if args.calc_dist_vs != -1:
        base_rope = ri.find_path_through_point_cloud_simple(get_cloud(logs[args.calc_dist_vs]))
    for i, log_data in enumerate(logs):
        offset_x = i*args.offset_x if args.offset_x > 0 else 0
        cloud_xyz = get_cloud(log_data, args.i)
        plt.plot(cloud_xyz[:,0] + offset_x, cloud_xyz[:,1], 'b.', alpha=.5)
        if not args.cloud_only:
            rope = ri.find_path_through_point_cloud(cloud_xyz)
            plt.plot(rope[:,0] + offset_x, rope[:,1], 'g.-', alpha=.3)

        if i != args.calc_dist_vs and args.calc_dist_vs != -1:
            print 'Dist from', i, 'to', args.calc_dist_vs, '=', min(calc_rope_dist(rope, base_rope), calc_rope_dist(rope, base_rope[::-1]))

    plt.axis('equal')
    plt.show()

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
    subparsers = parser.add_subparsers()

    parser_view_rope = subparsers.add_parser('view_rope')
    parser_view_rope.add_argument('--offset_x', type=float, default=-1)
    parser_view_rope.add_argument('--calc_dist_vs', type=int, default=-1)
    parser_view_rope.add_argument('-i', type=int, default=-1, help='index of entry in log')
    parser_view_rope.add_argument('--cloud_only', action='store_true', help='don\'t try to fit rope')
    parser_view_rope.set_defaults(func=view_rope)

    parser_extract_data = subparsers.add_parser('extract_data')
    parser_extract_data.add_argument('-i', type=int, required=True)
    parser_extract_data.add_argument('--out', type=str, default='')
    parser_extract_data.set_defaults(func=extract_data)

    parser_print_summary = subparsers.add_parser('print_summary')
    parser_print_summary.set_defaults(func=print_summary)

    args = parser.parse_args()
    logs = read_logs(args.log)
    args.func(args, logs)


if __name__ == '__main__':
    main()
