import sys
import os
sys.path.append("../utils")
sys.path.append("./")
import pdb
import BVH
import time
import numpy as np
import bpy
import mathutils
import pdb

#scale factor for bone length
global_scale = 10

class BVH_file:
    def __init__(self, file_path, start=0):
        self.anim, self.names, self.frametime = BVH.load(file_path, start=start)

        #permute (x, y, z) to (z, x, y)
        tmp = self.anim.offsets.copy()
        self.anim.offsets[..., 0] = tmp[..., 2]
        self.anim.offsets[..., 1] = tmp[..., 0]
        self.anim.offsets[..., 2] = tmp[..., 1]

        tmp = self.anim.positions.copy()
        self.anim.positions[..., 0] = tmp[..., 2]
        self.anim.positions[..., 1] = tmp[..., 0]
        self.anim.positions[..., 2] = tmp[..., 1]

        tmp = self.anim.rotations.qs.copy()
        self.anim.rotations.qs[..., 1] = tmp[..., 3]
        self.anim.rotations.qs[..., 2] = tmp[..., 1]
        self.anim.rotations.qs[..., 3] = tmp[..., 2]

        self.joint_num = self.anim.rotations.shape[1]
        self.frame_num = self.anim.rotations.shape[0]

        self.normalize()

    @property
    def topology(self):
        return self.anim.parents

    @property
    def offsets(self):
        return self.anim.offsets

    # Normalize bone length by height and translate the (x, y) mean to (0, 0)
    def normalize(self):
        height = self.get_height() / global_scale
        self.anim.offsets /= height
        self.anim.positions /= height
        mean_position = np.mean(self.anim.positions[:, 0, :], axis=0)
        self.anim.positions[:, 0, 0] -= mean_position[0]
        self.anim.positions[:, 0, 1] -= mean_position[1]


    def get_height(self):
        low = high = 0

        def dfs(i, pos):
            nonlocal low
            nonlocal high
            low = min(low, pos[-1])
            high = max(high, pos[-1])

            for j in range(self.joint_num):
                if self.topology[j] == i:
                    dfs(j, pos + self.offsets[j])

        dfs(0, np.array([0, 0, 0]))

        return high - low


def add_bone(offset, parent_obj, name):
    center = parent_obj.location + offset / 2
    length = offset.dot(offset) ** 0.5
    r1 = 0.032 * global_scale
    r2 = 0.0082 * global_scale

    if name.startswith("LeftHandIndex") or name.startswith("RightHandIndex"):
        length = length * 4
    # if name.startswith("LeftFingerBase") or name.startswith("LeftFingerBase"):
    #     r1 /= 4
    #     r2 /= 4

    base = mathutils.Vector((0., 0., 1.))
    target = offset.normalized()
    axis = base.cross(target)
    theta = np.math.acos(base.dot(target))
    rot = mathutils.Quaternion(axis, theta)
    

    bpy.ops.mesh.primitive_cone_add(vertices=8, radius1=r1, radius2=r2, depth=length, enter_editmode=False, location=center)
    new_bone = bpy.context.object
    new_bone.name = name
    new_bone.rotation_mode = 'QUATERNION'
    new_bone.rotation_quaternion = rot

    set_parent(parent_obj, new_bone)

    return new_bone

def add_joint(location, parent_obj, name):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=0.001, enter_editmode=False, location=location)
    new_joint = bpy.context.object
    if parent_obj is not None:
        set_parent(parent_obj, new_joint)
        pass
    new_joint.name = name

    return new_joint

def build_t_pose(file: BVH_file, joint, parent_obj, all_obj, index=0):
    if joint != 0:
        offset = mathutils.Vector(file.offsets[joint])
        new_bone = add_bone(offset, parent_obj, file.names[joint] + '_bone_{}'.format(index))
        new_joint = add_joint(parent_obj.location + offset, new_bone, file.names[joint] + '_end_{}'.format(index))
        all_obj.append(new_bone)
        all_obj.append(new_joint)
    else:
        new_joint = add_joint(mathutils.Vector((0., 0., 0.)), None, file.names[joint] + '_end_{}'.format(index))
        all_obj.append(new_joint)

    for i in range(len(file.topology)):
        if file.topology[i] == joint:
            build_t_pose(file, i, new_joint, all_obj, index=index)


def set_parent(parent, child):
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()
    '''
        See https://blender.stackexchange.com/questions/9200/how-to-make-object-a-a-parent-of-object-b-via-blenders-python-api
    '''


def set_animation(file, joints):
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = file.anim.rotations.shape[0] - 1

    bpy.context.scene.render.fps = 1 / file.frametime

    bpy.ops.object.select_all(action='DESELECT')

    print('Set fps to', bpy.context.scene.render.fps)
    print(file.frame_num, 'frames in total')
    print(file.joint_num, 'joint number in total')

    for frame in range(0, file.frame_num):
        joints[0].location = file.anim.positions[frame, 0, :]
        joints[0].keyframe_insert(data_path='location', frame=frame)
        if frame % 100 == 99:
            print('[{}/{}] done.'.format(frame+1, file.frame_num))
        for j in range(file.joint_num):
            joints[j].rotation_mode = 'QUATERNION'
            joints[j].rotation_quaternion = mathutils.Quaternion(file.anim.rotations.qs[frame, j, :])
            joints[j].keyframe_insert(data_path='rotation_quaternion', frame=frame)

    bpy.context.scene.frame_current = 0


def load_bvh(file_name, index=0, start=0):
    print('Loading BVH file......')
    file = BVH_file(file_name, start=start)
    print('Loading BVH file done.')

    print('Building T-Pose......')
    all_obj = []
    build_t_pose(file, 0, None, all_obj, index=index)
    print('Building T-Pose done.')

    print('Loading keyframes......')

    #pairing object order and file.animation's order
    all_joints = []
    for j in range(file.joint_num):
        name = file.names[j]
        for obj in all_obj:
            if obj.name == name + '_end' + '_{}'.format(index):
                all_joints.append(obj)
                break
    set_animation(file, all_joints)
    print('Loading keyframes done.')

    bpy.ops.object.select_all(action='DESELECT')
    for obj in all_obj:
        obj.select_set(True)
    bpy.ops.object.move_to_collection(collection_index=0, is_new=True, new_collection_name="Character_{}".format(index))
    bpy.ops.object.select_all(action='DESELECT')
    print('Load bvh all done!')

    return all_obj


if __name__ == '__main__':
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    file_name = './neutral_01_000.bvh'

    starting_frame_array = [0, 30, 60]

    for index in range(len(starting_frame_array)):
        load_bvh(file_name, index=index, start=starting_frame_array[index])
        time.sleep(3)
