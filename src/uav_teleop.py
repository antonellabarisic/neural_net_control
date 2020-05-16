#!/usr/bin/env python

from __future__ import print_function

import rospy

from geometry_msgs.msg import Pose
import sys
import select
import termios
import tty
import math

move_bindings = {
        'w':(1, 0),
        'e':(1, -1),
        'd':(0, -1),
        'c':(-1, -1),
        's':(-1, 0),
        'y':(-1, 1),
        'a':(0, 1),
        'q':(1, 1)     
}

increase_vel_bindings = {
        'i':0.01,
        'k':-0.01
}

increase_turn_bindings = {
        'j':0.01,
        'l':-0.01
}

T = 1/30.0

def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def vels(speed,turn):
    return "currently:\tspeed %s\tturn %s " % (speed,turn)

if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)

    pub = rospy.Publisher('/uav/pose_ref', Pose, queue_size = 1)
    rospy.init_node('uav_teleop')
    
    vector_x = 1.0
    vector_y = 0.0
    angle = 0.5*math.pi
    theta = 0
    s = 0
    
    lin_vel = 1.5
    angle_vel = 0.1
    
    pose = Pose()
    pose.position.x = 250.0;
    pose.position.y = -114.0;
    pose.position.z = 15.1;
    pose.orientation.x = 0.0;
    pose.orientation.y = 0.0;
    pose.orientation.z = math.sin(angle/2);
    pose.orientation.w = math.cos(angle/2);
    
    #starting rotation to get the correct vector. Rotate -45 degrees
    rotated_x = math.cos(0.25*math.pi + angle)*vector_x - math.sin(0.25*math.pi + angle)*vector_y
    vector_y = math.sin(0.25*math.pi + angle)*vector_x + math.cos(0.25*math.pi + angle)*vector_y
    vector_x = rotated_x
    
    try:
        while not rospy.is_shutdown():
            key = getKey()
                
            if key in increase_vel_bindings.keys():
                lin_vel += increase_vel_bindings[key]
                print(vels(lin_vel, angle_vel))
                
            elif key in increase_turn_bindings.keys():
                angle_vel += increase_turn_bindings[key]
                print(vels(lin_vel, angle_vel))
                
            elif key in move_bindings.keys():
                # orientation
                rot_theta = T * angle_vel * move_bindings[key][1]               
                theta += rot_theta
                
                pose.orientation.z = math.sin((angle + theta)/2.0)
                pose.orientation.w = math.cos((angle + theta)/2.0)
                
                # rotate movement vector
                rotated_x = math.cos(rot_theta)*vector_x - math.sin(rot_theta)*vector_y
                vector_y = math.sin(rot_theta)*vector_x + math.cos(rot_theta)*vector_y
                vector_x = rotated_x
                
                # movement
                pose.position.x += T * lin_vel * move_bindings[key][0] * vector_x
                pose.position.y += T * lin_vel * move_bindings[key][0] * vector_y
            
            elif key == 'u':
                pose.position.z += T * lin_vel
            
            elif key == 'h':
                pose.position.z -= T * lin_vel              
                
            elif (key == '\x03'):
                break
            
            pub.publish(pose)

    except Exception as e:
        print(e)

    finally:
        pose = Pose()
        pub.publish(pose)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
