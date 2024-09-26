#  Copyright (c) 2024 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

import os

import mujoco
import mujoco.viewer
from pyPlan import urdf

if __name__ == '__main__':
    directory = os.path.join(os.path.dirname(__file__), '../../')
    path = os.path.join(os.path.dirname(__file__),
                        '../../example-robot-data/robots/ur_description/urdf/ur3_robot.urdf')
    urdf_model = urdf.parseURDFFile(path)

    path = os.path.join(os.path.dirname(__file__),
                        'assets/empty_scene.xml')
    spec = mujoco.MjSpec()
    spec.from_file(path)


    def visit_link(link: urdf.Link | None, mujoco_body: mujoco.MjsBody):
        inertial = link.inertial
        if inertial:
            mujoco_body.mass = inertial.mass if inertial.mass > 0 else 1
            mujoco_body.inertia = [1, 1, 1]

        if link.parent_joint:
            urdf_joint = link.parent_joint
            parent_pose = urdf_joint.parent_to_joint_origin_transform
            if parent_pose:
                mujoco_body.pos = [parent_pose.position.x, parent_pose.position.y,
                                   parent_pose.position.z]
                mujoco_body.quat = [parent_pose.rotation.w, parent_pose.rotation.x,
                                    parent_pose.rotation.y, parent_pose.rotation.z]

            joint = mujoco_body.add_joint()
            if urdf_joint.JointType == urdf.Joint.JointType.REVOLUTE:
                joint.type = mujoco.mjtJoint.mjJNT_HINGE
            elif urdf_joint.JointType == urdf.Joint.JointType.PRISMATIC:
                joint.type = mujoco.mjtJoint.mjJNT_SLIDE
            else:
                joint.type = mujoco.mjtJoint.mjJNT_SLIDE

        for collision in link.collision_array:
            geom = mujoco_body.add_geom()
            geom.pos = [collision.origin.position.x, collision.origin.position.y,
                        collision.origin.position.z]
            geom.quat = [collision.origin.rotation.w, collision.origin.rotation.x,
                         collision.origin.rotation.y, collision.origin.rotation.z]

            geometry = collision.geometry
            if isinstance(geometry, urdf.Box):
                geom.type = mujoco.mjtGeom.mjGEOM_BOX
                geom.size = [geometry.dim.x, geometry.dim.y, geometry.dim.z]
            elif isinstance(geometry, urdf.Sphere):
                geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
                geom.size = [geometry.radius, 0, 0]
            elif isinstance(geometry, urdf.Mesh):
                geom.type = mujoco.mjtGeom.mjGEOM_MESH
                geom.meshname = geometry.filename

                mesh = spec.add_mesh()
                mesh.name = geometry.filename
                filename = geometry.filename[len("package://"):]
                mesh_path = os.path.join(directory, filename)
                mesh.file = mesh_path
                mesh.content_type = "model/stl"

                print(mesh_path)

        for child_link in link.child_links:
            mj_child_body = mujoco_body.add_body()
            visit_link(child_link, mj_child_body)


    root = urdf_model.root_link
    body = spec.worldbody.add_body()

    visit_link(root, body)

    model = spec.compile()
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
