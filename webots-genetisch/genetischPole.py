"""genetischPole controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, PositionSensor, Motor
from robot_model import PoleRobot
import time

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
#get all available devices for the robot
for i in range(robot.getNumberOfDevices()):
    print(str(robot.getDeviceByIndex(i)) + ": " + robot.getDeviceByIndex(i).getName())

#get access to pole position sensor
polePos:PositionSensor = robot.getDevice('polePosSensor')
polePos.enable(timestep)

#get access to motor
motor1: Motor = robot.getDevice('wheel1')
motor2: Motor = robot.getDevice('wheel1')
motor3: Motor = robot.getDevice('wheel1')
motor4: Motor = robot.getDevice('wheel1')

#set wheels to velocity control
motor1.setPosition(float('inf'))
motor2.setPosition(float('inf'))
motor3.setPosition(float('inf'))
motor4.setPosition(float('inf'))


def fitness_function(model):
    start_time = time.time()

    while robot.step(timestep) != -1:
        passed_time = start_time-time.time()/60 # --> minutes
        weighted_factor = passed_time if passed_time > 1 else 1# factor to be multiplied to poleposition (reward if pole is steady for longer time)

        poleVal = polePos.getValue()
        fitness = poleVal*weighted_factor

        with torch.no_grad():
            pass
        # Read the sensors:
        # Enter here functions to read sensor data, like:
        #  val = ds.getValue()

        # Process sensor data here.

        # Enter here functions to send actuator commands, like:
        #  motor.setPosition(10.0)
        pass

# Enter here exit cleanup code.
