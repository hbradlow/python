#!/usr/bin/env python

import numpy as np

import lfd
import os
from lfd import recognition, warping, registration
import os.path as osp
import cloud_io

PARALLEL = False

def draw_comparison(left_cloud, right_cloud, show_shape_context=False):
  from traits.api import HasTraits, Instance, Button, on_trait_change
  from traitsui.api import View, Item, HSplit, Group
  from mayavi import mlab
  from mayavi.core.ui.api import MlabSceneModel, SceneEditor

  class ComparisonDialog(HasTraits):
    scene1 = Instance(MlabSceneModel, ())
    scene2 = Instance(MlabSceneModel, ())
    view = View(
      HSplit(
        Group(
          Item(
            'scene1',
            editor=SceneEditor(),
            height=400,
            width=400,
            show_label=False,
          ),
          show_labels=False,
        ),
        Group(
          Item(
            'scene2',
            editor=SceneEditor(),
            height=400,
            width=400,
            show_label=False
          ),
          show_labels=False,
        ),
      ),
      resizable=True,
    )

    def draw_shape_context(self, scene, xyz_matched):
      assert 'shape_context_costs' in xyz_matched
      costs, matched_ds = xyz_matched['shape_context_costs'].copy(), xyz_matched['cloud_xyz_ds']
      costs -= min(costs)
      #costs /= max(costs) + 1
      costs /= 0.5
      print costs, costs.shape
      print matched_ds.shape
      x, y, z = matched_ds.transpose()
      mlab.points3d(x, y, z, costs, mode='sphere', figure=scene.mayavi_scene)

    @on_trait_change('scene1.activated')
    def redraw_scene1(self):
      self.redraw_scene(self.scene1, left_cloud)

    @on_trait_change('scene2.activated')
    def redraw_scene2(self):
      self.redraw_scene(self.scene2, right_cloud['cloud_xyz'])
      #self.redraw_scene(self.scene2, left_cloud, clear=False)
      if show_shape_context:
        self.draw_shape_context(self.scene2, right_cloud)

    def redraw_scene(self, scene, input_xyz, clear=True):
      if clear: mlab.clf(figure=scene.mayavi_scene)
      if input_xyz.shape[1] == 2:
        x, y = input_xyz.transpose()
        z = np.zeros(x.shape)
      elif input_xyz.shape[1] == 3:
        x, y, z = input_xyz.transpose()
      else:
        raise NotImplementedError
      mlab.points3d(x, y, z, mode='point', figure=scene.mayavi_scene)

  m = ComparisonDialog()
  m.configure_traits()

def select_from_list(lst, prompt='Choose from the following options: '):
    strlist = [str(item) for item in lst]
    while True:
      print prompt + " ".join("(%s)"%item for item in strlist)
      resp = raw_input("> ")
      if resp not in strlist:
        print "invalid response. try again."
      else:
        return lst[strlist.index(resp)]            

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='overhand_knot', help='name of dataset')
  parser.add_argument('--method', choices=['geodesic_dist', 'shape_context', 'geodesic_dist+shape_context'], default='geodesic_dist', help='matching algorithm')
  parser.add_argument('input', type=str)
  parser.add_argument('--show_shape_context', action='store_true')
  parser.add_argument('--save_input', type=str, default='')
  args = parser.parse_args()

  dataset = recognition.DataSet.LoadFromTaskDemos(args.dataset)

  # read input to match
  input_xyz = cloud_io.read_from_rsrc(args.input)
  if args.save_input:
    cloud_io.write_cloud(args.save_input, input_xyz)

  # create the matcher
  matcher = None
  if args.method == 'geodesic_dist':
    matcher = recognition.GeodesicDistMatcher(dataset)
  elif args.method == 'shape_context':
    matcher = recognition.ShapeContextMatcher(dataset)
  elif args.method == 'geodesic_dist+shape_context':
    matcher = recognition.CombinedNNMatcher(dataset, [recognition.GeodesicDistMatcher, recognition.ShapeContextMatcher], [1, 0.1])
  else:
    raise NotImplementedError

  # do the matching
  best_match_name, best_match_cost = matcher.match(input_xyz)
  print 'Best match name:', best_match_name, 'with cost:', best_match_cost
  draw_comparison(left_cloud=input_xyz, right_cloud=dataset[best_match_name], show_shape_context=args.show_shape_context)

if __name__ == '__main__':
  main()
