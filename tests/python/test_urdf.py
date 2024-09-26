#  Copyright (c) 2024 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

import pyPlan
from pyPlan import urdf
import os


def test_parser():
    path = os.path.join(os.path.dirname(__file__),
                        '../../../example-robot-data/robots/panda_description/urdf/panda.urdf')
    model = urdf.parseURDFFile(path)
    for name, joint in model.joints.items():
        print(f"joint name: {name}")
    for name, link in model.links.items():
        print(f"link name: {name}")
        if link.visual:
            geometry = link.visual.geometry
            if isinstance(geometry, urdf.Mesh):
                print(f"mesh file: {geometry.filename}")