import cv2
import colorsys
import numpy as np
class AnnotateFrame:
    def __init__(self,window_identifier,colorCount,frame):
        self.enforceMaxPoint = False
        self.pointLimit=1E18
        self.colorsList = self.generateDistinctBrightColors(colorCount)
        self.window_identifier = window_identifier
        self.original_frame = frame.copy()
        self._frameData =[] # list of list of pts [ [(),(),(),()...], ... ]
        self.newDataset()
        self._currentIndex = 1E18 # ensures the pointer is adjusted to the last element
        self.maxPointCheck=-1
        self._draggedPointIndex = None
        self.zeroFrame = np.zeros(frame.shape,dtype=frame.dtype)
    
    @property
    def frameData(self): # obj.frameData -> 
        return self._frameData

    @frameData.setter
    def frameData(self,frameData): # stores framedata and sets the last dataset of frameData as the current dataset and update the frame
        # if frame data is empty, create a new dataset
        self._frameData = frameData
        if not len(self._frameData):
            self.newDataset(check=False)
        elif self._currentIndex >= len(self._frameData):
            self.current_dataset=self._frameData[-1]
            self._currentIndex = len(self._frameData)-1
        frame = self.original_frame.copy()
        self.updateFrameFromFrameData(frame,self._frameData)

    def updateFrameFromFrameData(self,frame, frameData):
        # frameData -> list of list of coordinates
        for i in range(len(frameData)):
            if (i==self._currentIndex):
                self.draw_line_from_list(frameData[i],frame,self.colorsList[i],3,dragIndex=self._draggedPointIndex,showNumbers=True)
                frame = self.getTransparentPolyfill(frame,frameData[i],self.colorsList[i])
            else:
                self.draw_line_from_list(frameData[i],frame,self.desaturateColour(self.colorsList[i]),2)
        
        cv2.imshow(self.window_identifier, frame)

    def setPointLimit(self,limit:int,enforce:bool):
        # for loading of data, call this after setting enforce
        self.pointLimit = limit
        self.enforceMaxPoint = enforce
    
    def undoShape(self): # bind to some other key to use 
        if len(self._frameData):
            self._frameData.pop(self._currentIndex)
            self.shiftColor(self._currentIndex)
            
            self._currentIndex = 1E18 # reset index to new last element
            self.frameData = self._frameData

    def mouseCallback(self,event,x,y,flags,param):
        # param is the Optional variable for other stuff if needed
        if event == cv2.EVENT_LBUTTONDOWN:
            self._draggedPointIndex = self.checkForPointPress((x,y))
            if self._draggedPointIndex!=None:
                return
            else:
                if len(self.current_dataset) >= self.pointLimit:
                    self.newDataset()
                self.current_dataset.append((x,y))
                self._draggedPointIndex = len(self.current_dataset)-1
                self.frameData = self._frameData

        if self._draggedPointIndex!=None:
            if event==cv2.EVENT_MOUSEMOVE:
                self.current_dataset[self._draggedPointIndex] = (x,y)
            elif event==cv2.EVENT_LBUTTONUP:
                self.current_dataset[self._draggedPointIndex] = (x,y)
                self._draggedPointIndex = None

            self.frameData = self._frameData
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            clickedIndex = self.checkForPointPress((x,y))
            if clickedIndex ==None:
                clickedIndex=-1
            if len(self.current_dataset) ==1:
                self._frameData.pop(self._currentIndex)
                self.shiftColor(self._currentIndex)
                self._currentIndex = 1E18 # resets index to new last element
            elif len(self.current_dataset):
                self.current_dataset.pop(clickedIndex)
            elif len(self._frameData):
                self._frameData.pop()
                if len(self._frameData):
                    self.current_dataset = self._frameData[-1]
                    self.current_dataset.pop()
            self.frameData = self._frameData

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags>0:
                self._currentIndex+=1 if self._currentIndex<len(self._frameData)-1 else 0
            else:
                self._currentIndex-=1 if self._currentIndex>0 else 0
            self.current_dataset = self._frameData[self._currentIndex]
            self.frameData = self._frameData

        elif event == cv2.EVENT_MBUTTONUP:
            self.newDataset()

    def shiftColor(self,index):
        #call this function to shift the color at self._currentIndex to the end of the colors array
        # mainly used to maintain color consistency when deleting shapes from the middle of the data
        sortedList = self.colorsList[index:]
        shifted_color = self.colorsList[index]
        sortedList = sorted(sortedList,key=lambda x:(x==shifted_color).all())
        self.colorsList = np.array(list(self.colorsList[0:index]) + sortedList)
        
                
    def changeLineColor(self):
        self.colorsList[self._currentIndex],self.colorsList[-1] = self.colorsList[-1], self.colorsList[self._currentIndex]
        self.shiftColor(len(self._frameData))
        self.frameData = self._frameData

    def newDataset(self,check = True): # returns a reference to the new dataset for the frame 
        # for loading of data, call this before setting enforce
        if check and self.enforceMaxPoint:
            if len(self.current_dataset) == self.pointLimit:
                newDataset = []
                self._frameData.append(newDataset)
                self.current_dataset = newDataset
                self._currentIndex = len(self._frameData)-1
            return    
            
        newDataset = []
        self._frameData.append(newDataset)
        self.current_dataset = newDataset
        self._currentIndex = len(self._frameData)-1

    def checkForPointPress(self,xy):
        for i,pt in enumerate(self.current_dataset):
            if self.checkPtInCircle(xy,pt,10):
                return i
        else:
            return None
    
    def getTransparentPolyfill(self,frame,ptList,color,alpha=0.2):        
        if len(ptList) < 3: # unable to form a 2d shape
            return frame
        frame1=self.zeroFrame.copy()
        cv2.fillConvexPoly(frame1,np.array(ptList,'int32'),color*alpha)
        return cv2.add(frame,frame1)

    @staticmethod
    def draw_line_from_list(ptList,frame,color,thickness,dragIndex=None,showNumbers=False):
        font = cv2.FONT_HERSHEY_SIMPLEX
        if len(ptList):
            for i in range(len(ptList)):
                if (showNumbers):
                    cv2.putText(frame,str(i),ptList[i],font,1.2,color,thickness)
                if (i==dragIndex):
                    cv2.circle(frame,ptList[i],thickness+16,color,-1)
                else:
                    cv2.circle(frame,ptList[i],thickness+4,color,-1)
                cv2.line(frame,ptList[i-1],ptList[i],color,thickness) # 0-1 = -1 last point to first point

    @staticmethod
    def desaturateColour(rgb): # rgb is a nparray
        
        (hue,lumin,sat) =  colorsys.rgb_to_hls(*(rgb/255))
        sat /= 4
        return np.array(colorsys.hls_to_rgb(hue,lumin,sat))*255

    @staticmethod
    def generateDistinctBrightColors(n):
        #python format (hue,lumin,sat)
        #generate n distinct points in the volume of the hls cylinder
        def maximizeDiff(arr):
            arr = sorted(arr)
            i = 0
            j = len(arr)-1
            new_arr = []
            while j-i>2:
                new_arr.append(arr[i])
                new_arr.append(arr[j])
                i+=1
                j-=1
            if j-1 == 2:
                new_arr.append(arr[i+1])
            return new_arr
        hue = maximizeDiff(np.linspace(0,1,n))
        sat = maximizeDiff(np.linspace(0.8,1,n))
        lum = maximizeDiff(np.linspace(0.4,0.6,n))
        return np.array([colorsys.hls_to_rgb(*x) for x in zip(hue,lum,sat)])*255

    @staticmethod
    def checkPtInCircle(point,center,radius):
        return (point[1] - center[1])**2 + (point[0]-center[0])**2 <= radius*radius