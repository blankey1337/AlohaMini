import argparse
import time
import math
import xml.etree.ElementTree as ET
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R

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
    rr.init("urdf_visualizer", spawn=True)
    
    links, joints = parse_urdf(urdf_path)
    
    # Build tree structure
    # joint_map: parent_link -> list of joints where this link is parent
    joint_map = {}
    for j in joints:
        p = j['parent']
        if p not in joint_map:
            joint_map[p] = []
        joint_map[p].append(j)

    # State
    t = 0.0
    
    while True:
        # Update joint values (sine waves)
        joint_values = {}
        for j in joints:
            if j['type'] in ['revolute', 'continuous']:
                val = math.sin(t + sum(map(ord, j['name']))) # Random phase
                joint_values[j['name']] = val
            elif j['type'] == 'prismatic':
                val = 0.3 + 0.3 * math.sin(t)
                joint_values[j['name']] = val
            else:
                joint_values[j['name']] = 0.0

        # FK
        # Stack: (link_name, accumulated_transform)
        stack = [('base_link', np.eye(4))]
        
        while stack:
            link_name, T_parent = stack.pop()
            
            # Log link visual
            link_data = links.get(link_name)
            if link_data and link_data['visual']:
                # Visual origin offset
                v_orig = link_data['visual_origin']
                T_visual_offset = get_transform(v_orig['xyz'], v_orig['rpy'])
                T_visual = T_parent @ T_visual_offset
                
                rr.set_time_seconds("sim_time", t)
                
                # Decompose for rerun
                trans = T_visual[:3, 3]
                rot = R.from_matrix(T_visual[:3, :3]).as_quat() # xyzw
                # Rearrange to xyzw for scipy -> xyzw for rerun (check convention, rerun uses xyzw)
                
                entity_path = f"robot/{link_name}"
                
                geo = link_data['visual']
                if geo['type'] == 'box':
                    rr.log(entity_path, rr.Boxes3D(half_sizes=[s/2 for s in geo['size']], centers=[0,0,0]), rr.Transform3D(translation=trans, rotation=rr.Quaternion(xyzw=rot)))
                elif geo['type'] == 'cylinder':
                    # Rerun cylinder is along Z?
                    # URDF cylinder is along Z (usually)
                    # We might need to check visualization, but simple logging:
                    # Rerun doesn't have Cylinder3D primitive in older versions, checking...
                    # It does have standard primitives now? Or we use Mesh3D / Points.
                    # As fallback, log a box approximating cylinder or use Points if needed. 
                    # Use Box for now to be safe and simple.
                    rr.log(entity_path, rr.Boxes3D(half_sizes=[geo['radius'], geo['radius'], geo['length']/2], centers=[0,0,0]), rr.Transform3D(translation=trans, rotation=rr.Quaternion(xyzw=rot)))

            # Children
            children_joints = joint_map.get(link_name, [])
            for j in children_joints:
                # Static transform
                T_static = get_transform(j['xyz'], j['rpy'])
                
                # Joint transform
                T_joint = np.eye(4)
                if j['type'] in ['revolute', 'continuous']:
                    angle = joint_values.get(j['name'], 0)
                    axis = np.array(j['axis'])
                    # Axis-angle rotation
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
    parser.add_argument("--urdf", default="../../src/lerobot/robots/alohamini/alohamini.urdf")
    args = parser.parse_args()
    
    visualize(args.urdf)
