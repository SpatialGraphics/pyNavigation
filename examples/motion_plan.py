#  Copyright (c) 2024 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

import time
import os
import mujoco
import mujoco.viewer
from pyPlan import ompl

if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__),
                        '../../mujoco_menagerie/franka_emika_panda/scene.xml')

    spec = mujoco.MjSpec()
    spec.from_file(path)
    body = spec.worldbody.add_body()
    body.name = "block"
    body.pos = [0, .6, .4]
    body.quat = [.5, -.2, 1, 0]
    geom = body.add_geom()
    geom.rgba = [1, 0, 0, 1]
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [.15, .15, .15]

    model = spec.compile()
    data = mujoco.MjData(model)

    state_space = ompl.CompoundStateSpace()
    for i in range(model.nu):
        if model.actuator_ctrllimited[i]:
            subspace = ompl.RealVectorStateSpace(1)
            subspace.setBounds(model.actuator_ctrlrange[i][0], model.actuator_ctrlrange[i][1])
            state_space.addSubspace(subspace, 1)

    space_information = ompl.SpaceInformation(state_space)
    simple_setup = ompl.geometry.SimpleSetup(space_information)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.7
        viewer.cam.elevation = -15
        viewer.cam.azimuth = -130
        viewer.cam.lookat = (0, 0, .3)

        with viewer.lock():
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        #     viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE

        while viewer.is_running():
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
