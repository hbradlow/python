#!/usr/bin/env python

import os
import pickle
import matplotlib.pyplot as plt

def read_logs(logfiles):
    logs = []
    for logfile in logfiles:
        assert os.path.exists(logfile)
        with open(logfile, 'r') as f:
            logs.append(pickle.load(f))
    return logs

def read_log(logfile):
    return read_logs([logfile])[0]

def get_first_seen_cloud(log_data):
    cloud_xyz = None
    for entry in log_data:
        if entry['state'] == 'LookAtObject' and entry['msg'] == 'xyz':
            cloud_xyz = entry['data']
            break
    if cloud_xyz is None:
        raise RuntimeError('cannot find LookAtObject xyz entry')
    return cloud_xyz

# def view_cloud(logs):
#     for log_data in logs:
#         cloud_xyz = get_first_seen_cloud(log_data)
#         plt.plot(cloud_xyz[:,0], cloud_xyz[:,1], 'b.')
#     plt.axis('equal')
#     plt.show()

def view_rope(args, logs):
    import rope_initialization as ri
    for i, log_data in enumerate(logs):
        offset_x = i*args.offset_x if args.offset_x > 0 else 0
        cloud_xyz = get_first_seen_cloud(log_data)
        rope = ri.find_path_through_point_cloud(cloud_xyz)
        plt.plot(cloud_xyz[:,0] + offset_x, cloud_xyz[:,1], 'bo', alpha=.1)
        plt.plot(rope[:,0] + offset_x, rope[:,1], 'g.-')
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
