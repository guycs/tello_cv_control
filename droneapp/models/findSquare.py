import threading

import cv2 as cv
import config
import math
import logging
import numpy as np
import statistics

logger = logging.getLogger(__name__)

class Tracker:

    def __init__ (self, debug):
        self.debug = debug
        self.processingEnded = True
        self.processingSucceeded = False
        self.failInARow = 0
        self.successInARow = 0
        self.movingAvarageLength = 4
        self.processingTime = 0
        self.targetValid = False
        self.startTime = cv.getTickCount() / cv.getTickFrequency()
        self.targetStatus = 'initial'
        self.target = {
            'ex': 'start',
            'roi_start': (0,0),
            'roi_end':(config.FRAME_SIZE_X  - 1, config.FRAME_SIZE_Y - 1),
            'status': 'initial',
            'timeStamp': self.startTime
        }
        self.predictedTarget = {
            'roi_start': (0,0),
            'roi_end':(config.FRAME_SIZE_X - 1, config.FRAME_SIZE_Y - 1),
            'status':'initial'
        }
        self.smoothedTarget = {
            'roi_start': (0, 0),
            'roi_end': (config.FRAME_SIZE_X - 1, config.FRAME_SIZE_Y - 1),
            'xList': [],
            'yList': [],
            'sideAngleList': [],
            'upAngleList': [],
            'timeStampList':[],
            'rangeList':[]
        }

        self.sideAngleGraph=self.createGraphCanvas(640,512).copy()
        self.xGraph = self.createGraphCanvas(640, 512).copy()




    def get_target(self, frame):
        if (self.processingEnded):
            smoothedTarget = self.smoothTarget(self.target, self.movingAvarageLength, self.processingSucceeded)

            if (self.processingSucceeded):
                self.failInARow = 0
                self.successInARow += 1
                if (self.successInARow > 4):
                    self.targetValid = True
                    for key in smoothedTarget: self.predictedTarget[key] =  smoothedTarget[key]
                self.processingSucceeded = False
            else:
                cv.putText(frame, self.target['ex'], (50, 20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                self.failInARow += 1
                self.successInARow = 0
                if (self.failInARow > 90 and self.targetValid):
                    self.targetValid = False
                    logger.info('switching valid to false')

            tracker_thread = threading.Thread(
                target=self.findSquare,
                args=(frame.copy(), config.LOWER_BLUE, config.UPPER_BLUE,))
            # logger.info('starting findSquare method')
            self.processingEnded = False
            self.processingTime = 0
            tracker_thread.start()

        if (self.targetValid):
            contourColor = (150, 0, 0) if self.predictedTarget['status'] == 'updated' else (0, 150, 150)
            cv.drawContours(frame, [self.predictedTarget['contour']], -1, contourColor, 5)
            centerColor = (0, 150, 0) if self.predictedTarget['status'] == 'updated' else (150, 0, 0)
            #logger.info('status:' + self.targetStatus + ', center: ' + str(self.predictedTarget['center']) + "color: " + str(centerColor))
            cv.circle(frame, self.predictedTarget['center'], 5, centerColor, -1)
            cv.putText(frame, str(int(self.predictedTarget['range'])), (config.FRAME_SIZE_X//2 - 20,config.FRAME_SIZE_Y - 50), cv.FONT_HERSHEY_COMPLEX, 1, 0, 2)

        if self.predictedTarget['status'] != 'initial':  self.updatePrediction(self.predictedTarget)
        logger.info(self.predictedTarget)

            # plot graph
            # ------------

            #self.plotGraphPoint(self.sideAngleGraph, 'sideAngle', graphSpreadFactorX =10, graphSpreadFactorY=500, shiftYStart=-400)
            #self.plotGraphPointFromArray(self.xGraph, 'center',0, graphSpreadFactorX=10, graphSpreadFactorY=0.3,shiftYStart=0)


        cv.putText(frame, str(self.successInARow), (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, 0, 2)
        cv.putText(frame, str(self.failInARow), (125, 100), cv.FONT_HERSHEY_COMPLEX, 1, 0, 2)
        cv.putText(frame, str(self.processingTime), (200, 100), cv.FONT_HERSHEY_COMPLEX, 1, 0, 2)
        cv.putText(frame, str(len(self.smoothedTarget['timeStampList'])), (275, 100), cv.FONT_HERSHEY_COMPLEX, 1, 0, 2)
        cv.rectangle(frame, self.target['roi_start'], self.target['roi_end'], (100, 100, 100), 2)


        self.processingTime += 1
        return frame

    def createGraphCanvas (self, sizeX, sizeY):
        graph = np.zeros([sizeY, sizeX, 3], dtype=np.uint8)
        graph.fill(255)
        cv.putText(graph, 'raw', (50, 20), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv.putText(graph, 'smoothed', (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv.putText(graph, 'predicted', (50, 180), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        return graph


    def plotGraphPoint(self, graph, patameter, graphSpreadFactorX, graphSpreadFactorY, shiftYStart):
        currentTime = (cv.getTickCount() / cv.getTickFrequency() - self.startTime)
        graphTime = int(currentTime * graphSpreadFactorX) % graph.shape[1]
        shiftY = shiftYStart - 200 * (graphTime //  graph.shape[1])

        cv.circle(graph, (graphTime,
                  int(self.target[patameter] * graphSpreadFactorY + shiftY)), 5, (255, 0, 0),-1)

        cv.circle(graph, (graphTime,
                  int(self.smoothedTarget[patameter] * graphSpreadFactorY + shiftY)), 3,(0, 255, 0), -1)

        cv.circle(graph, (graphTime,
                  int(self.predictedTarget[patameter] * graphSpreadFactorY + shiftY)), 1,(0, 0, 255), -1)

        cv.imshow(patameter, graph)
        cv.waitKey(1)

    def plotGraphPointFromArray (self, graph, patameter, index, graphSpreadFactorX, graphSpreadFactorY, shiftYStart):
        currentTime = (cv.getTickCount() / cv.getTickFrequency() - self.startTime)
        graphTime = int(currentTime * graphSpreadFactorX) %  graph.shape[1]
        shiftY = shiftYStart - 200 * (graphTime //  graph.shape[1])

        cv.circle(graph, (graphTime,
                int(self.target[patameter][index] * graphSpreadFactorY + shiftY)), 5,(255, 0, 0), -1)

        cv.circle(graph, (graphTime,
                int(self.smoothedTarget[patameter][index]  * graphSpreadFactorY + shiftY)), 3,(0, 255, 0), -1)

        cv.circle(graph, (graphTime,
                int(self.predictedTarget[patameter][index]  * graphSpreadFactorY + shiftY)), 1,(0, 0, 255), -1)

        cv.imshow(patameter, graph)
        cv.waitKey(1)


    def findSquare (self, originalFrame, lowerColorTreshold, upperColorTreshold, ):
        img = originalFrame.copy()
        if (self.debug):
            cv.imshow('process', img)
            print('img')
            cv.waitKey(1)
            while True:
                if cv.waitKey(1) & 0xFF == ord('x'):
                    break

        if (not self.targetValid):  #(not self.targetValid):
            roi = img
            self.target['roi_start'] = (0, 0)
            self.target['roi_end'] = (config.FRAME_SIZE_X-1, config.FRAME_SIZE_Y-1)
        else:
            margin = int(config.FRAME_SIZE_Y / 10 * (self.failInARow+1))
            xMin = min(self.target['downLeftCor'][0], self.target['upLeftCor'][0]) - margin
            if (xMin < 0): xMin =0

            xMax = max(self.target['downRightCor'][0], self.target['upRightCor'][0]) + margin
            if (xMax > config.FRAME_SIZE_X-1): xMax = config.FRAME_SIZE_X-1

            yMin = min(self.target['upRightCor'][1], self.target['upLeftCor'][1]) - margin
            if (yMin < 0): yMin = 0

            yMax = max(self.target['downRightCor'][1] , self.target['downLeftCor'][1])+ margin
            if (yMax > config.FRAME_SIZE_Y-1): yMax = config.FRAME_SIZE_Y-1

            roi_mask = np.zeros(img.shape[:2], dtype="uint8")
            cv.rectangle(roi_mask, (xMin, yMin), (xMax, yMax), 255, -1)
            roi = cv.bitwise_and(img, img, mask=roi_mask)
            self.target['roi_start'] = (xMin,yMin)
            self.target['roi_end'] = (xMax,yMax)
            if (self.debug):
                cv.imshow('process', roi)
                print('roi')
                cv.waitKey(1)
                while True:
                    if cv.waitKey(1) & 0xFF == ord('x'):
                        break

        # Convert BGR to HSV
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        if (self.debug):
            cv.imshow('process', hsv)
            print('hsv')
            cv.waitKey(1)
            while True:
                if cv.waitKey(1) & 0xFF == ord('x'):
                    break

        # Threshold the HSV image to get only blue colors
        mask = cv.inRange(hsv, lowerColorTreshold, upperColorTreshold)
        if (self.debug):
            cv.imshow('process', mask)
            print('mask')
            cv.waitKey(1)
            while True:
                if cv.waitKey(1) & 0xFF == ord('x'):
                    break


        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        timeStamp = cv.getTickCount()/cv.getTickFrequency()
        if (self.debug):
            print('found', len (contours), 'contours')
            cv.drawContours(img, contours, -1, (0, 0, 255), 5)
            cv.imshow('process', img)
            #cv.drawContours(mask, contours, -1, (255, 0, 255), 5)
            #cv.imshow('process', mask)
            print('draw contours')
            cv.waitKey(1)
            while True:
                if cv.waitKey(1) & 0xFF == ord('x'):
                    break
        if (self.debug): print('contours:')

        lastTarget = self.target.copy()
        lastTimeStamp = lastTarget['timeStamp']
        duration = timeStamp - lastTimeStamp
        highestExLevel = 0
        exMessage = ''
        for contour in contours:
            aContour = self.analyze_contour(contour, lastTarget, duration).copy()
            if (self.debug): print(aContour)
            if (not aContour['valid']):
                if (aContour['exLevel'] > highestExLevel):
                    exMessage   = aContour['ex']
                    highestExLevel = aContour['exLevel']
            else:
                for key in aContour: self.target[key] = aContour[key]

                self.target['timeStamp'] = timeStamp
                self.target['ex'] = ''
                self.target['exLevel'] = -1

                self.processingSucceeded = True
                if (self.debug):
                    cv.drawContours(img, [self.target['contour']], -1, (0, 255, 0), 5)
                    cv.circle(img, self.target['center'], 5, (255, 0, 255), -1)
                    for i in range(4):
                        print('i:', i)
                        print('self.targetContour corner:', (self.target['contour'][i][0][0]))
                        cx, cy = self.target['contour'][i][0]
                        cv.putText(img, str(i), (cx, cy), cv.FONT_HERSHEY_COMPLEX, 1, 0, 2)
                    cv.imshow('process', img)
                    print('shaw result')
                    cv.waitKey(1)
                    while True:
                        if cv.waitKey(1) & 0xFF == ord('x'):
                            break
                break

        if not self.processingSucceeded:
            self.target['ex'] = exMessage
            self.target['exLevel'] = highestExLevel

        if (self.debug) : cv.destroyWindow('process')
        cv.waitKey(1)
        self.processingEnded = True
        return (self.target)

#----------------------------------------------------------------------#



    def analyze_contour (self, contour, lastTarget, duration):

        aContour ={'ex': 'no contours', 'exLevel': 0}
        try:
            epsilon = 0.1 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
        except:
            aContour['valid'] = False
            aContour['ex'] = 'failed in contour smoothing'
            aContour['exLevel'] += 1
            return aContour.copy()

        if (len(approx) != 4):
            aContour['valid'] = False
            aContour['ex'] = 'no square contours'
            aContour['exLevel'] += 1
            return aContour.copy()

        aContour['area'] = cv.contourArea(approx)
        if (aContour['area'] < 500):
            aContour['ex'] = 'contours too small'
            aContour['exLevel'] += 1
            aContour['valid'] = False
            return aContour.copy()


        aContour ['contour'] = self.arrangeContour(approx)
        c = aContour ['contour']
        (x1,y1) = c[0][0]
        (x2,y2) = c[1][0]
        (x3,y3) = c[2][0]
        (x4,y4) = c[3][0]

        #   1   3
        #   2   4

        aContour['upLeftCor'] = (x1,y1)
        aContour['downLeftCor'] = (x2, y2)
        aContour['upRightCor'] = (x3, y3)
        aContour['downRightCor'] = (x4, y4)
        xCenter = int((x1+x4)/2)
        yCenter = int((y1+y4)/2)
        aContour['center'] = (xCenter, yCenter)

        if self.cropped(aContour):
            aContour['valid'] = False
            aContour['ex'] = 'cropped'
            aContour['exLevel'] += 1
            return aContour.copy()

        if lastTarget['status'] == 'initial' or not lastTarget['valid']:
            aContour['xSpeed'] = 0
        else:
            aContour['xSpeed'] = (xCenter - lastTarget['center'][0]) / duration

       # aContour['xSpeed'] =(0 if lastTarget['status'] == 'initial' or not lastTarget['valid'] else
       #                        (xCenter - lastTarget['center'][0]) / duration)

        aContour['ySpeed'] =(0 if lastTarget['status'] == 'initial' or not lastTarget['valid'] else
                              (yCenter - lastTarget['center'][1]) / duration)

        if y2 == y1 or y4 == y3 or x3 == x1 or x4 == x2:
            aContour['valid'] = False
            aContour['ex'] = 'not a square'
            aContour['exLevel'] += 1
            return aContour.copy()

        leftSlope = (x2-x1)/(y2-y1)
        rightSlope = (x4-x3)/(y4-y3)
        upperSlope = (y3-y1)/(x3-x1)
        lowerSlope = (y4-y2)/(x4-x2)

      #  slopeLimit = 0.3
      #  if (leftSlope>slopeLimit or rightSlope > slopeLimit or upperSlope >slopeLimit or lowerSlope > slopeLimit):
      #      aContour['valid'] = False
      #      aContour['ex'] = 'slope value outside range'
      #      return aContour.copy()

        aContour['verticalSlopesDiff'] = abs(leftSlope - rightSlope)
        if (aContour['verticalSlopesDiff']  > 0.3 ):
            aContour['valid'] = False
            aContour['ex'] = 'vertical slope diff outside range'
            aContour['exLevel'] += 1
            return aContour.copy()

        aContour['horizontalSlopesDiff'] = abs(upperSlope - lowerSlope)
        if (aContour['horizontalSlopesDiff'] > 0.3 ):
            aContour['valid'] = False
            aContour['ex'] = 'horizontal slope diff outside range'
            aContour['exLevel'] += 1
            return aContour.copy()

        aContour['upperLength'] = int(math.sqrt((y3-y1)**2 + (x3-x1)**2))
        aContour['lowerLength'] = int(math.sqrt ((y4-y2)**2 + (x4-x2)**2))
        aContour['leftLength'] = int(math.sqrt((y1 - y2) ** 2 + (x2 - x1) ** 2))
        aContour['rightLength'] = int(math.sqrt((y3 - y4) ** 2 + (x3 - x4) ** 2))

        aContour['upAngle'] = aContour['upperLength'] / aContour['lowerLength']
        aContour['upAngleSpeed'] = (0 if lastTarget['status'] == 'initial' or not lastTarget['valid'] else
                                      (aContour['upAngle'] - lastTarget['upAngle']) / duration)

        aContour['sideAngle'] = aContour['rightLength'] / aContour['leftLength']
        aContour['sideAngleSpeed'] = (0 if lastTarget['status'] == 'initial' or not lastTarget['valid'] else
                                      (aContour['sideAngle'] - lastTarget['sideAngle']) / duration)


        if ( aContour['upAngle'] < 0.8 or aContour['upAngle']  > 1.2):
            aContour['valid'] = False
            aContour['ex'] = 'horizontal sides ratio out of range'
            aContour['exLevel'] += 1
            return aContour.copy()


        #aContour['area'] = (aContour['upperLength'] + aContour['lowerLength'])**2 / 4
        if (self.targetValid):
            areaShpill = 0.2 * (self.failInARow + 1)
            if (self.target['area'] / aContour['area'] < (1 - areaShpill) or self.target['area'] / aContour['area'] > (1 + areaShpill)):
                aContour['valid'] = False
                aContour['ex'] = 'area change is out of range: ' + str(areaShpill)
                aContour['exLevel'] += 1
                return aContour.copy()

        aContour['range'] = (math.sqrt(config.FRAME_SIZE_AREA / aContour['area']))*10
        aContour['rangeSpeed'] = (0 if lastTarget['status'] == 'initial' or not lastTarget['valid'] else
                               (aContour['range'] - lastTarget['range']) / duration)

   #     M = cv.moments(approx)
    #    if M['m00'] != 0.0:
     #       x = int(M['m10'] / M['m00'])
      #      y = int(M['m01'] / M['m00'])
       #     centerCordinates = (x, y)
        #    aContour['center'] = centerCordinates
        #else:
        #    aContour['valid'] = False
        #    aContour['ex'] = 'contour moment failed '
        #    aContour['exLevel'] += 1
        #    return aContour.copy()

        aContour['valid'] = True
        return aContour.copy()


    def arrangeContour (self, sContour):
        sContour = self.sortCorners(sContour, 0, 0, 4)
        sContour = self.sortCorners(sContour, 1, 0, 2)
        sContour = self.sortCorners(sContour, 1, 2, 4)
        return sContour.copy()

    def sortCorners(self, sContour, col, start,end):
        for i in range (start, end-1):
            min = sContour[i][0][col]
            for j in range (i+1,end):
                if(sContour[j][0][col] < min):
                    temp = sContour[i].copy()
                    sContour[i] = sContour[j].copy()
                    sContour[j] = temp.copy()
        return sContour.copy()

    def cropped(self, target):
        xMin = min(target['downLeftCor'][0], target['upLeftCor'][0])
        if (xMin <= 0): return True

        xMax = max(target['downRightCor'][0], target['upRightCor'][0])
        if (xMax >= config.FRAME_SIZE_X - 1): return True

        yMin = min(target['upRightCor'][1], target['upLeftCor'][1])
        if (yMin <= 0): return True

        yMax = max(target['downRightCor'][1], target['downLeftCor'][1])
        if (yMax >= config.FRAME_SIZE_Y - 1): return True

        return False


    def updatePrediction (self, _target):
        target = _target.copy()
        self.predictedTarget['status'] = 'predicted'

        timeStamp = cv.getTickCount()/cv.getTickFrequency()
        duration = (timeStamp - target['timeStamp'])
        (x, y) = target['center']
        self.predictedTarget['center'] = (
        int(x + duration * target['xSpeed']), int(y + duration * target['ySpeed']))
        self.predictedTarget['sideAngle'] = (target['sideAngle'] + duration * target['sideAngleSpeed'])
        self.predictedTarget['upAngle'] = (target['upAngle'] + duration * target['upAngleSpeed'])
        self.predictedTarget['timeStamp'] = timeStamp
        self.predictedTarget['range'] = (target['range'] + duration * target['rangeSpeed'])

        return self.predictedTarget


    def smoothTarget(self, target, requiredLength, processingSucceeded):

        if processingSucceeded:
            self.smoothedTarget['timeStampList'].append(target['timeStamp'])
            self.smoothedTarget['xList'].append(target['center'][0])
            self.smoothedTarget['yList'].append(target['center'][1])
            self.smoothedTarget['sideAngleList'].append(target['sideAngle'])
            self.smoothedTarget['upAngleList'].append(target['upAngle'])
            self.smoothedTarget['rangeList'].append(target['range'])

        length = len(self.smoothedTarget['timeStampList'])
        if  (processingSucceeded and length > requiredLength) or (not processingSucceeded and length > 0):
            self.smoothedTarget['timeStampList'].pop(0)
            self.smoothedTarget['xList'].pop(0)
            self.smoothedTarget['yList'].pop(0)
            self.smoothedTarget['sideAngleList'].pop(0)
            self.smoothedTarget['upAngleList'].pop(0)
            self.smoothedTarget['rangeList'].pop(0)
            length -=1

        if (length < 2):
            self.smoothedTarget['status'] = 'initial'
            return target

        duration = self.smoothedTarget['timeStampList'][-1] - self.smoothedTarget['timeStampList'][0]

        smoothedTargetX = statistics.mean(self.smoothedTarget['xList'])
        self.smoothedTarget['xSpeed'] = (self.smoothedTarget['xList'][-1] - self.smoothedTarget['xList'][0])/duration

        smoothedTargetY = statistics.mean(self.smoothedTarget['yList'])
        self.smoothedTarget['ySpeed'] = (self.smoothedTarget['yList'][-1] - self.smoothedTarget['yList'][0]) / duration

        self.smoothedTarget['center'] = (int(smoothedTargetX), int(smoothedTargetY))

        self.smoothedTarget['sideAngle'] = statistics.mean(self.smoothedTarget['sideAngleList'])
        self.smoothedTarget['sideAngleSpeed'] = (self.smoothedTarget['sideAngleList'][-1] - self.smoothedTarget['sideAngleList'][0]) / duration

        self.smoothedTarget['upAngle'] = statistics.mean(self.smoothedTarget['upAngleList'])
        self.smoothedTarget['upAngleSpeed'] = (self.smoothedTarget['upAngleList'][-1] - self.smoothedTarget['upAngleList'][0]) / duration

        self.smoothedTarget['range'] = statistics.mean(self.smoothedTarget['rangeList'])
        self.smoothedTarget['rangeSpeed'] = (self.smoothedTarget['rangeList'][-1] -
                                                   self.smoothedTarget['rangeList'][0]) / duration

        self.smoothedTarget['contour'] = target['contour']
        self.smoothedTarget['timeStamp'] = self.smoothedTarget['timeStampList'][-1]
        self.smoothedTarget['status'] = 'updated'
        return self.smoothedTarget


