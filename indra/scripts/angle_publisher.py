#!/usr/bin/env python3

import rospy
from indra.msg import robot_input
from copy import deepcopy

default = [90, 0, 0, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 0, 0, 90, 90]

try:
    pub = rospy.Publisher("input", robot_input, queue_size=10)
    rospy.init_node('angle_publisher', anonymous=False)
    l = [90, 0, 0, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 0, 0, 90, 90]
    rate = rospy.Rate(100)
    pub.publish(l)
    while (1):
        temp = input().split()
        if ('r' in temp):
            l = deepcopy(default)
        else:
            for i in temp:
                number, angle = [int(j) for j in i.split(",")]
                l[number] = angle
        l[6] = min(270 - l[5] - l[4], 180)
        l[9] = min(270 - l[10] - l[11], 180)
        l[7] = l[3]
        l[8] = l[12]
        pub.publish(l)
except rospy.ROSInterruptException:
    pass