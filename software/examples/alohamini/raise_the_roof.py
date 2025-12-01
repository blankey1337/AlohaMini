import argparse
import time
import math
import xml.etree.ElementTree as ET
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R
import os

def parse_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    links = {}
    joints = []

    for link in root.findall('link'):
        name = link.get('name')
        visual = link.find('visual')
        geometry_info = None
        color_info = None
        origin_info = None
        
        if visual is not None:
            geo = visual.find('geometry')
            if geo is not None:
                box = geo.find('box')
                cylinder = geo.find('cylinder')
                if box is not None:
                    geometry_info = {'type': 'box', 'size': [float(x) for x in box.get('size').split()]}
                elif cylinder is not None:
                    geometry_info = {'type': 'cylinder', 'length': float(cylinder.get('length')), 'radius': float(cylinder.get('radius'))}
            
            mat = visual.find('material')
            if mat is not None:
                color_info = mat.get('name') # Simplified
            
            orig = visual.find('origin')
            if orig is not None:
                xyz = [float(x) for x in orig.get('xyz', '0 0 0').split()]
                rpy = [float(x) for x in orig.get('rpy', '0 0 0').split()]
                origin_info = {'xyz': xyz, 'rpy': rpy}
            else:
                origin_info = {'xyz': [0,0,0], 'rpy': [0,0,0]}

        links[name] = {
            'visual': geometry_info,
            'color': color_info,
            'visual_origin': origin_info
        }

    for joint in root.findall('joint'):
        name = joint.get('name')
        type_ = joint.get('type')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        
        origin = joint.find('origin')
        xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()] if origin is not None else [0,0,0]
        rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()] if origin is not None else [0,0,0]
        
        axis_elem = joint.find('axis')
        axis = [float(x) for x in axis_elem.get('xyz', '1 0 0').split()] if axis_elem is not None else [1,0,0]

        limit = joint.find('limit')
        limits = None
        if limit is not None:
            limits = (float(limit.get('lower', -3.14)), float(limit.get('upper', 3.14)))

        joints.append({
            'name': name,
            'type': type_,
            'parent': parent,
            'child': child,
            'xyz': xyz,
            'rpy': rpy,
            'axis': axis,
            'limits': limits
        })

    return links, joints

def get_transform(xyz, rpy):
    rot = R.from_euler('xyz', rpy).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = xyz
    return T

def visualize(urdf_path):
    print("Starting visualize...")
    rr.init("raise_the_roof", spawn=True)
    print("Rerun initialized.")
    
    links, joints = parse_urdf(urdf_path)
    
    # Build tree structure
    joint_map = {}
    for j in joints:
        p = j['parent']
        if p not in joint_map:
            joint_map[p] = []
        joint_map[p].append(j)

    t = 0.0
    
    while True:
        # "Raise the roof" motion
        # Cycle time: 1 second up, 1 second down
        cycle = math.sin(t * 8) # Speed up a bit more
        
        # Lift Axis: Oscillate between 0.1 and 0.5
        # Range of motion: 0.4
        # Center: 0.3
        lift_height = 0.3 + 0.2 * cycle
        
        # Static Hands Up Pose ("Goal Post" / "Surrender" pose)
        # Arms out to the sides, forearms pointing up.
        
        # Shoulder Pan: Out to sides (+/- 90 degrees)
        left_pan = 1.57
        right_pan = -1.57
        
        # Shoulder Lift: Horizontal (0.0)
        shoulder_lift_angle = 0.0
        
        # Elbow: Bent 90 degrees to point forearms up
        elbow_angle = 1.57 
        
        joint_values = {}
        for j in joints:
            name = j['name']
            
            # Prismatic Lift Axis
            if name == 'lift_axis':
                joint_values[name] = lift_height
                
            elif 'shoulder_lift' in name:
                joint_values[name] = shoulder_lift_angle
            
            elif 'elbow_flex' in name:
                joint_values[name] = elbow_angle
            
            elif 'shoulder_pan' in name:
                 if 'left' in name:
                     joint_values[name] = left_pan
                 else:
                     joint_values[name] = right_pan

            elif 'wrist' in name:
                joint_values[name] = 0.0
            
            elif j['type'] == 'prismatic' and name != 'lift_axis':
                joint_values[name] = 0.0
            elif j['type'] not in ['revolute', 'continuous', 'prismatic']:
                joint_values[name] = 0.0
            elif name not in joint_values:
                 joint_values[name] = 0.0

        # FK
        stack = [('base_link', np.eye(4))]
        
        while stack:
            link_name, T_parent = stack.pop()
            
            # Log link visual
            link_data = links.get(link_name)
            if link_data and link_data['visual']:
                v_orig = link_data['visual_origin']
                T_visual_offset = get_transform(v_orig['xyz'], v_orig['rpy'])
                T_visual = T_parent @ T_visual_offset
                
                rr.set_time_seconds("sim_time", t)
                
                trans = T_visual[:3, 3]
                rot = R.from_matrix(T_visual[:3, :3]).as_quat() # xyzw
                
                entity_path = f"robot/{link_name}"
                
                geo = link_data['visual']
                if geo['type'] == 'box':
                    rr.log(entity_path, rr.Boxes3D(half_sizes=[s/2 for s in geo['size']], centers=[0,0,0]), rr.Transform3D(translation=trans, rotation=rr.Quaternion(xyzw=rot)))
                elif geo['type'] == 'cylinder':
                    rr.log(entity_path, rr.Boxes3D(half_sizes=[geo['radius'], geo['radius'], geo['length']/2], centers=[0,0,0]), rr.Transform3D(translation=trans, rotation=rr.Quaternion(xyzw=rot)))

            # Children
            children_joints = joint_map.get(link_name, [])
            for j in children_joints:
                T_static = get_transform(j['xyz'], j['rpy'])
                
                T_joint = np.eye(4)
                if j['type'] in ['revolute', 'continuous']:
                    angle = joint_values.get(j['name'], 0)
                    axis = np.array(j['axis'])
                    rot_j = R.from_rotvec(axis * angle).as_matrix()
                    T_joint[:3, :3] = rot_j
                elif j['type'] == 'prismatic':
                    dist = joint_values.get(j['name'], 0)
                    axis = np.array(j['axis'])
                    T_joint[:3, 3] = axis * dist
                
                T_child = T_parent @ T_static @ T_joint
                stack.append((j['child'], T_child))

        t += 0.05
        time.sleep(0.05)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path relative to this script
    default_urdf = os.path.join(os.path.dirname(__file__), "../../src/lerobot/robots/alohamini/alohamini.urdf")
    parser.add_argument("--urdf", default=default_urdf)
    args = parser.parse_args()
    
    visualize(args.urdf)
