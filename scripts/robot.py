#!/usr/bin/env python

############################################################
# CSE 190 PA 3
# Name : Albert Richie
# Email: arichie@ucsd.edu
# PID  : A98407956
# Filename: robot.py
############################################################

import rospy
from qlearning import qlearning
from std_msgs.msg import Bool

class robot():
    
    def __init__(self):

        rospy.init_node("robot")
        rospy.sleep(1)
        
        # Topics published
        self.completion_publish = rospy.Publisher(
            "/map_node/sim_complete",
            Bool,
            queue_size = 10
            )

        qlearning()
        
        self.publish_complete()
        
############################################################
    # Class functions to publish message outbound
    def publish_complete(self):

        rospy.sleep(1)
        self.completion_publish.publish(True)
        rospy.sleep(1)
        rospy.signal_shutdown("Shutting Down...")

if __name__ == "__main__":
    try:
        rt = robot()
    except rospy.ROSInterruptException:
        pass
