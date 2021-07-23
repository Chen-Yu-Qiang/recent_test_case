#!/usr/bin/env python
from geometry_msgs.msg import Twist

class meanFilter:
    def __init__(self,N):
        self.n=N
        self.data=list()
        # [old...new]
    def update(self,obj):
        if len(self.data)<self.n:
            self.data=self.data+[obj]
        else:
            self.data=self.data[1:self.n+1]+[obj]
        return self.calMean()
    
    def calMean(self):
        if len(self.data)==0:
            return 0
        if str(type(self.data[0]))=="<type 'float'>" or str(type(self.data[0]))=="<type 'int'>":
            s=0.0            
            for i in range(len(self.data)):
                s=s+self.data[i]
            return s/len(self.data)
        elif str(type(self.data[0]))=="<class 'geometry_msgs.msg._Twist.Twist'>":
            s=Twist()
            
            for i in range(len(self.data)):
                s.linear.x = s.linear.x + float(self.data[i].linear.x)/len(self.data) 
                s.linear.y = s.linear.y + float(self.data[i].linear.y)/len(self.data) 
                s.linear.z = s.linear.z + float(self.data[i].linear.z)/len(self.data) 
                s.angular.z = s.angular.z + float(self.data[i].angular.z)/len(self.data) 
            s.angular.x = self.data[-1].angular.x
            
            return s      

if __name__ == '__main__':
    f=meanFilter(3)
    for i in [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0]:
        print(i,f.update(i))

    f2=meanFilter(3)

    d1=Twist()
    d1.linear.x = 0
    d1.linear.y = 0
    d1.linear.z = 0
    d1.angular.z = 0

    d2=Twist()
    d2.linear.x = 1
    d2.linear.y = 0
    d2.linear.z = 0
    d2.angular.z = 5

    d3=Twist()
    d3.linear.x = 0
    d3.linear.y = 1
    d3.linear.z = 0
    d3.angular.z = 6

    d4=Twist()
    d4.linear.x = 0
    d4.linear.y = 1
    d4.linear.z = 3
    d4.angular.z = 6

    d5=Twist()
    d5.linear.x = 0
    d5.linear.y = 1
    d5.linear.z = 3
    d5.angular.z = 6

    d6=Twist()
    d6.linear.x = 0
    d6.linear.y = 1
    d6.linear.z = 3
    d6.angular.z = 6
    for i in [d1,d2,d3,d4,d5,d6]:
        print(i)
        print("++++++")
        print(f2.update(i))
        print("====================")

