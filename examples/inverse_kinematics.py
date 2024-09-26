#  Copyright (c) 2024 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

import os
import time

import mujoco
import mujoco.viewer
import numpy as np
from mujoco import minimize

if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__),
                        '../../mujoco_menagerie/franka_emika_panda/scene.xml')

    spec = mujoco.MjSpec()
    spec.from_file(path)
    body = spec.worldbody.add_body()
    body.childclass = 'target'
    body.name = "target"
    body.mocap = True
    body.pos = [0, .6, .4]
    body.quat = [.5, -.2, 1, 0]
    target = body.add_site()
    target.name = "target"
    target.group = 2

    geom = body.add_geom()
    geom.fromto = [.07, 0, 0, .15, 0, 0]
    geom.rgba = [1, 0, 0, 1]
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size[0] = .015
    geom = body.add_geom()
    geom.fromto = [0, .07, 0, 0, .15, 0]
    geom.rgba = [0, 1, 0, 1]
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size[0] = .015
    geom = body.add_geom()
    geom.fromto = [0, 0, .07, 0, 0, .15]
    geom.rgba = [0, 0, 1, 1]
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size[0] = .015

    hand = spec.find_body('hand')
    effector = hand.add_site()
    effector.name = "effector"
    effector.pos = [0, 0, 0.08]

    model = spec.compile()
    key = model.key('home')
    key.qpos = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0, 0]
    key.ctrl = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0]

    data = mujoco.MjData(model)
    # Reset the state to the "home" keyframe.
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    # Bounds at the joint limits.
    bounds = [model.jnt_range[:, 0], model.jnt_range[:, 1]]
    x0 = model.key('home').qpos


    def pose(time):
        pos = (0.4 * np.sin(time),
               0.4 * np.cos(time),
               0.4 + 0.2 * np.sin(3 * time))
        quat = np.array((1.0, np.sin(2 * time), np.sin(time), 0))
        quat /= np.linalg.norm(quat)
        return pos, quat


    def ik(x, pos=None, quat=None, radius=0.04, reg=1e-3, reg_target=None):
        """Residual for inverse kinematics.

        Args:
          x: joint angles.
          pos: target position for the end effector.
          quat: target orientation for the end effector.
          radius: scaling of the 3D cross.

        Returns:
          The residual of the Inverse Kinematics task.
        """

        # Move the mocap body to the target
        id = model.body('target').mocapid
        data.mocap_pos[id] = model.body('target').pos if pos is None else pos
        data.mocap_quat[id] = model.body('target').quat if quat is None else quat

        # Set qpos, compute forward kinematics.
        res = []
        for i in range(x.shape[1]):
            data.qpos = x[:, i]
            mujoco.mj_kinematics(model, data)

            # Position residual.
            res_pos = data.site('effector').xpos - data.site('target').xpos

            # Effector quat, use mju_mat2quat.
            effector_quat = np.empty(4)
            mujoco.mju_mat2Quat(effector_quat, data.site('effector').xmat)

            # Target quat, exploit the fact that the site is aligned with the body.
            target_quat = data.body('target').xquat

            # Orientation residual: quaternion difference.
            res_quat = np.empty(3)
            mujoco.mju_subQuat(res_quat, target_quat, effector_quat)
            res_quat *= radius

            # Regularization residual.
            reg_target = model.key('home').qpos if reg_target is None else reg_target
            res_reg = reg * (x[:, i] - reg_target)

            res_i = np.hstack((res_pos, res_quat, res_reg))
            res.append(np.atleast_2d(res_i).T)

        return np.hstack(res)


    def ik_jac(x, res, pos=None, quat=None, radius=.04, reg=1e-3):
        """Analytic Jacobian of inverse kinematics residual

        Args:
          x: joint angles.
          pos: target position for the end effector.
          quat: target orientation for the end effector.
          radius: scaling of the 3D cross.

        Returns:
          The Jacobian of the Inverse Kinematics task.
        """
        # least_squares() passes the value of the residual at x which is sometimes
        # useful, but we don't need it here.
        del res

        # Call mj_kinematics and mj_comPos (required for Jacobians).
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        # Get end-effector site Jacobian.
        jac_pos = np.empty((3, model.nv))
        jac_quat = np.empty((3, model.nv))
        mujoco.mj_jacSite(model, data, jac_pos, jac_quat, data.site('effector').id)

        # Get Deffector, the 3x3 mju_subquat Jacobian
        effector_quat = np.empty(4)
        mujoco.mju_mat2Quat(effector_quat, data.site('effector').xmat)
        target_quat = data.body('target').xquat
        Deffector = np.empty((3, 3))
        mujoco.mjd_subQuat(target_quat, effector_quat, None, Deffector)

        # Rotate into target frame, multiply by subQuat Jacobian, scale by radius.
        target_mat = data.site('target').xmat.reshape(3, 3)
        mat = radius * Deffector.T @ target_mat.T
        jac_quat = mat @ jac_quat

        # Regularization Jacobian.
        jac_reg = reg * np.eye(model.nv)

        return np.vstack((jac_pos, jac_quat, jac_reg))


    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.7
        viewer.cam.elevation = -15
        viewer.cam.azimuth = -130
        viewer.cam.lookat = (0, 0, .3)

        with viewer.lock():
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        #     viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE

        x = x0
        while viewer.is_running():
            step_start = time.time()
            pos, quat = pose(step_start)
            x_prev = x.copy()

            # Define IK problem
            ik_target = lambda x: ik(x, pos=pos, quat=quat, reg_target=x_prev, reg=.1)
            jac_target = lambda x, r: ik_jac(x, r, pos=pos, quat=quat)

            x, _ = minimize.least_squares(x, ik_target, bounds, jacobian=jac_target, verbose=0)

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
