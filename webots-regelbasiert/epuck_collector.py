"""epuck_collector controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot,Receiver, Motor, DistanceSensor
import msgpack
import numpy as np
import logging as log
import cv2
# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = 128#int(robot.getBasicTimeStep())

camera = robot.getDevice('camera')
camera.enable(timestep)

#motorsachen
motorLeft:Motor = robot.getDevice('left wheel motor')
motorRight:Motor = robot.getDevice('right wheel motor')
motorLeft.setPosition(float('inf')) #this sets the motor to velocity control instead of position control
motorRight.setPosition(float('inf'))
motorLeft.setVelocity(0)
motorRight.setVelocity(0)
maxVelocity = motorLeft.getMaxVelocity()

motorLeftSensor:Motor = robot.getDevice('left wheel sensor')
motorLeftSensor.enable(timestep)
motorRightSensor:Motor = robot.getDevice('right wheel sensor')
motorRightSensor.enable(timestep)

for i in range(robot.getNumberOfDevices()):
    print(str(robot.getDeviceByIndex(i)) + ": " + robot.getDeviceByIndex(i).getName())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getMotor('motorname')
#  ds = robot.getDistanceSensor('dsname')
#  ds.enable(timestep)

rec:Receiver = robot.getDevice("receiver")
rec.enable(timestep)
log.basicConfig(level=log.INFO, format='%(asctime)s %(filename)s %(levelname)s: %(message)s')
#counter mit 1 initialisieren
counter = 1

#simulation mit drehbewegung initialisieren
motorRight.setVelocity(0)
motorLeft.setVelocity(2)

position_list = []

while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    left_pos = motorLeftSensor.getValue()
    right_pos = motorRightSensor.getValue()
    
    #Kamerabild auslesen
    img = camera.getImageArray()
    #Bild drehen und anpassen, damit es mit den Richtungen passt
    img = cv2.rotate(np.array(img), cv2.cv2.ROTATE_90_CLOCKWISE)
    img = cv2.flip(img, 1)
    #kamera höhe und breite auslesen, um später fensterkoordinaten anpassen zu können
    x,y = camera.getWidth(), camera.getHeight()
    
    # convert image to grayscale, um Circledetection anwenden zu können
    cimg = cv2.cvtColor(np.float32(np.array(img)), cv2.COLOR_BGR2GRAY)
    # apply a blur using the median filter
    cimg = cv2.medianBlur(cimg, 5)
    
    # finds the circles in the grayscale image using the Hough transform
    try:
        cimg = np.uint8(np.array(cimg))#umwandeln des grauen bildes in richtiges format
        circles = cv2.HoughCircles(image=cimg, method=cv2.HOUGH_GRADIENT, dp = 1.25,minDist=1, 
                            param1=135, param2=19.5, maxRadius=40)#cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,20,
                                    #param1=50,param2=30,minRadius=0,maxRadius=0)
        
        if isinstance(circles, np.ndarray):#kreise wurden gefunden
            circles = np.uint16(np.around(circles))#datentyp noch anpassen
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),1)
                # draw the center of the circle
                cv2.circle(img,(i[0],i[1]),2,(0,0,255),1)

            #da kamera nicht funktioniert, hier visuelle ausgabe wenn kreise erkannt werden
            test = cv2.resize(np.float32(np.array(img)/255), (520,390))
            cv2.imshow('Vergroesserte Kameraeingabe', test)
            cv2.waitKey(500)#cv2.waitKey(1)
            cv2.destroyAllWindows()

            counter = 5#counter hochsetzen, dass die beschleunigung in die entsprechende richtung häufiger durchgeführt wird als 1mal
            x_,y_,radius = circles[0][0]#ersten kreis aus liste auswählen

            #werte auslesen in denen sich der kreis im bild befindet
            xmin_, xmax_ = int(x_-radius), int(x_+radius)
            ymin_, ymax_ = int(y_-radius), int(y_+radius)

            #koordinatenwerte anpassen, falls sie aus dem Bild rausfallen
            if ymin_ < 0:
                ymin_ = 0
            if xmin_ < 0:
                xmin_ = 0
            if xmax_ >= x:
                xmax_ = x-1#index
            if ymax_ >= y:
                ymax_ = y-1#index

            #bild basierend auf den kreiskoordinaten ausschneiden
            img_cropped = img[xmin_:xmax_]
            #print('ausgeschnittenes image:', img_cropped)
            #print(img_cropped.shape)

            if x_ < x/2 - 7:#kugel ist mehr links
                
                for k in range(img_cropped.shape[0]):
                    line = img_cropped[k]
                    mean_color = np.mean(line, axis = 0)
                if mean_color[1] > mean_color[0] and mean_color[1] > mean_color[2]:#bedingung für grüne kugeln
                    direction = direction#bei bekannter richtung bleiben
                else:
                    direction = 'left'#richtung zwischenspeichern in der gesucht wird
                    #geschwindigkeiten so setzen, dass in die richtige richtung gedreht wird
                    motorRight.setVelocity(maxVelocity)
                    motorLeft.setVelocity(-2)
            elif x_ > x/2 + 7:#kugel ist mehr rechts
                
                for k in range(img_cropped.shape[0]):
                    line = img_cropped[k]
                    #print('line shape', line.shape)
                    mean_color = np.mean(line, axis = 0)
                if mean_color[1] > mean_color[0] and mean_color[1] > mean_color[2]:#bedingung für grüne kugeln
                    direction = direction
                else:
                    direction = 'right'
                    for k in range(3):#für rechts wird mehr "überzeugung" benötigt --> 3 in for schleife
                        motorLeft.setVelocity(maxVelocity)
                        motorRight.setVelocity(-2)
                
            elif x_ == x//2:
                
                for k in range(img_cropped.shape[0]):
                    line = img_cropped[k]
                    mean_color = np.mean(line, axis = 0)
                if mean_color[1] > mean_color[0] and mean_color[1] > mean_color[2]:#bedingung für grüne kugeln
                    direction = direction#kugel ist grün --> bekannte richtung beibehalten
                else:
                    direction = 'forward'
                    motorRight.setVelocity(maxVelocity)
                    motorLeft.setVelocity(maxVelocity)
            else:
                if direction == 'right':
                    motorRight.setVelocity(0)
                    motorLeft.setVelocity(0)

                    motorRight.setVelocity(0)
                    motorLeft.setVelocity(2)
                elif direction == 'left':
                    motorRight.setVelocity(0)
                    motorLeft.setVelocity(0)

                    motorRight.setVelocity(2)
                    motorLeft.setVelocity(0)
                else:
                    motorRight.setVelocity(0)
                    motorLeft.setVelocity(0)

                    motorRight.setVelocity(maxVelocity)
                    motorLeft.setVelocity(maxVelocity)
            #auf verdacht mal nach vorne fahren, um die kugel näher zu haben, es wurde ja schon in
            #die richtige richtung gedreht
            for i in range(counter):
                motorRight.setVelocity(maxVelocity)
                motorLeft.setVelocity(maxVelocity)
        else:
            counter = 1
            #abfragen basierend darauf, wo zuletzt eine kugel gesehen wurde, oder welche drehrichtung eingestellt ist nachdem eine
            # eingestellt wurde. Für den fall dass in dem aktuellen durchlauf keine kugeln detektiert werden, obwohl sie noch zu sehen sein müssten 
            
            if direction == 'right':
                motorRight.setVelocity(0)
                motorLeft.setVelocity(0)

                motorRight.setVelocity(0)
                motorLeft.setVelocity(2)
            elif direction == 'left':
                motorRight.setVelocity(0)
                motorLeft.setVelocity(0)

                motorRight.setVelocity(2)
                motorLeft.setVelocity(0)
            else:
                motorRight.setVelocity(0)
                motorLeft.setVelocity(0)

                motorRight.setVelocity(maxVelocity)
                motorLeft.setVelocity(maxVelocity)
            
            
    except Exception as e:
        #print('circles nicht geklappt')      
        #print(e)              
        pass
    

    while rec.getQueueLength() > 0:
        msg_dat = rec.getData()
        print('msg data', msg_dat)
        rec.nextPacket()
        msg = msgpack.unpackb(msg_dat)
        print('msg', msg)
        log.info(msg)

        counter = 1
        motorRight.setVelocity(-1)
        motorLeft.setVelocity(2)
        direction = 'left'
        
      

# Enter here exit cleanup code.
