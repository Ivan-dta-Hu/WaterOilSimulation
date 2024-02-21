import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
# exit()
# Define constants:
height = 101                         # lattice dimensions
width = 101
four9ths = 4.0/9.0                  # abbreviations for lattice-Boltzmann weight factors
one9th   = 1.0/9.0
one36th  = 1.0/36.0
 # flow direction: center, E, N, W, S, NE, NW, SW, SE
w=np.array([four9ths,one9th,one9th,one9th,one9th,one36th,one36th,one36th,one36th])

# Initialize barriers:
barrier = np.zeros((height,width), bool)                 # True wherever there's a barrier
for y in range(height):
    for x in range(width):
        if (y-int(height/2))**2+(x-int(height/2))**2 >= int(height/2)**2:
            barrier[y,x] = True
# bounce back
barrierN = ~np.roll(barrier,  1, axis=0) & barrier
vacancyN = np.roll(barrierN, -1 , axis=0)
barrierS = ~np.roll(barrier, -1, axis=0) & barrier
vacancyS = np.roll(barrierS, 1 , axis=0)
barrierE = ~np.roll(barrier,  1, axis=1) & barrier
vacancyE = np.roll(barrierE, -1 , axis=1)
barrierW = ~np.roll(barrier, -1, axis=1) & barrier
vacancyW = np.roll(barrierW, 1 , axis=1)
barrierNE = ~np.roll(barrier,  (1,1), axis=(0,1)) & barrier
vacancyNE = np.roll(barrierNE, (-1,-1), axis=(0,1))
barrierNW = ~np.roll(barrier, (1,-1), axis=(0,1)) & barrier
vacancyNW = np.roll(barrierNW, (-1,1), axis=(0,1))
barrierSE = ~np.roll(barrier, (-1,1), axis=(0,1)) & barrier
vacancySE = np.roll(barrierSE, (1,-1), axis=(0,1))
barrierSW = ~np.roll(barrier, (-1,-1), axis=(0,1)) & barrier
vacancySW = np.roll(barrierSW, (1,1), axis=(0,1))

# gradient calculation
edgeN = np.roll(barrierN, -2 , axis=0)
edgeS = np.roll(barrierS, 2 , axis=0)
edgeE = np.roll(barrierE, -2 , axis=1)
edgeW = np.roll(barrierW, 2 , axis=1)
extraPadN = np.roll(barrierN, 1 , axis=0)
extraPadN[0,:] = False
extraEdgN = np.roll(extraPadN, -2 ,axis=0)
extraPadS = np.roll(barrierS, -1 , axis=0)
extraPadS[-1,:] = False
extraEdgS = np.roll(extraPadS, 2 ,axis=0)
extraPadE = np.roll(barrierE, 1 , axis=1)
extraPadE[:,0] = False
extraEdgE = np.roll(extraPadE, -2 ,axis=1)
extraPadW = np.roll(barrierW, -1 , axis=1)
extraPadW[:,-1] = False
extraEdgW = np.roll(extraPadW, 2 ,axis=1)

viscosity1 = 0.17
omega1 = 1 / (3*viscosity1 + 0.5)     # "relaxation" parameter
viscosity2 = 0.17
omega2 = 1 / (3*viscosity2 + 0.5)
rho1 = 1 * (~barrier).astype('float64')
rho2 = np.zeros((height,width))
rad=5
for ry in range(rad*2,height,rad*4):
    for rx in range(rad*2,width,rad*4):
        if barrier[ry,rx]:
            continue
        for y in range(ry-rad,ry+rad):
            for x in range(rx-rad,rx+rad):
                if barrier[y,x]:
                    continue
                if (y-ry)**2+(x-rx)**2 <= rad**2:
                    rho2[y,x] = 1
# circleList=[(50,40,10),(50,70,15)]
# for cy,cx,rad in circleList:
    # for y in range(height):
        # for x in range(width):
            # if barrier[y,x]:
                # continue
            # if (y-cy)**2+(x-cx)**2 <= rad**2:
                # rho2[y,x] = 1
# for y in range(height):
    # for x in range(width):
        # if barrier[y,x]:
            # continue
        # if x < int(width/2):
            # rho2[y,x] = 1
rho1[rho2>0.5] = 0.0
flow1= np.expand_dims(w,axis=(1,2))*rho1
flow2= np.expand_dims(w,axis=(1,2))*rho2
c0,g11,g22,c1,c2=6,1,1,0.06,0.02
iF1,iF2,g12,g21,kappa1,kappa2=1,1,1,1,-100,30
def GradientX(phi):
    phi[barrier]=phi[0,0]
    phi[barrierE]=2*phi[vacancyE]-phi[edgeE]
    phi[barrierW]=2*phi[vacancyW]-phi[edgeW]
    phi[extraPadE]=phi[extraEdgE]
    phi[extraPadW]=phi[extraEdgW]
    dphidx=c1*(phi[1:-1,2:]-phi[1:-1,:-2])+c2*(phi[2:,2:]+phi[:-2,2:]-phi[2:,:-2]-phi[:-2,:-2])
    dphidx=np.pad(dphidx, ((1, ),(1, )), constant_values=((0, ),(0, )))
    return dphidx
def GradientY(phi):
    phi[barrier]=phi[0,0]
    phi[barrierN]=2*phi[vacancyN]-phi[edgeN]
    phi[barrierS]=2*phi[vacancyS]-phi[edgeS]
    phi[extraPadN]=phi[extraEdgN]
    phi[extraPadS]=phi[extraEdgS]
    dphidy=c1*(phi[2:,1:-1]-phi[:-2,1:-1])+c2*(phi[2:,2:]+phi[2:,:-2]-phi[:-2,2:]-phi[:-2,:-2])
    dphidy=np.pad(dphidy, ((1, ),(1, )), constant_values=((0, ),(0, )))
    return dphidy
def SecondGradient(dphidx, dphidy): # require kappa1, kapp2 = -100, 30
    Lphi=GradientX(dphidx)+GradientY(dphidy)
    GLphix,GLphiy = GradientX(Lphi),GradientY(Lphi)
    return GLphix,GLphiy
def SecondGradient2(dphidx, dphidy): # require kappa1, kapp2 = -20, 20
    dphidx[barrierE]=2*dphidx[vacancyE]-dphidx[edgeE]
    dphidx[barrierW]=2*dphidx[vacancyW]-dphidx[edgeW]
    dphidy[barrierN]=2*dphidy[vacancyN]-dphidy[edgeN]
    dphidy[barrierS]=2*dphidy[vacancyS]-dphidy[edgeS]
    Lphi=dphidx[1:-1,2:]-dphidx[1:-1,:-2]+dphidy[2:,1:-1]-dphidy[:-2,1:-1]
    Lphi=np.pad(Lphi, ((1, ),(1, )), constant_values=((0, ),(0, )))
    Lphi[barrierN]=2*Lphi[vacancyN]-Lphi[edgeN]
    Lphi[barrierS]=2*Lphi[vacancyS]-Lphi[edgeS]
    GLphiy=c1*(Lphi[2:,1:-1]-Lphi[:-2,1:-1])
    GLphiy=np.pad(GLphiy, ((1, ),(1, )), constant_values=((0, ),(0, )))
    Lphi[barrier]=Lphi[0,0]
    Lphi[barrierE]=2*Lphi[vacancyE]-Lphi[edgeE]
    Lphi[barrierW]=2*Lphi[vacancyW]-Lphi[edgeW]
    GLphix=c1*(Lphi[1:-1,2:]-Lphi[1:-1,:-2])
    GLphix=np.pad(GLphix, ((1, ),(1, )), constant_values=((0, ),(0, )))
    return GLphix,GLphiy
velocityThreshold=np.sqrt(2/3) #np.sqrt(1/3) insure all positive
def NormalizeVelocity(ux,uy):
    uLength = np.sqrt(ux * ux + uy * uy)
    overflow=uLength>velocityThreshold
    if not np.any(overflow):
        return ux, uy
    uxNorm = np.divide(ux, uLength, out=np.zeros((height,width)), where=uLength!=0)
    uyNorm = np.divide(uy, uLength, out=np.zeros((height,width)), where=uLength!=0)
    return np.where(overflow,uxNorm*velocityThreshold,ux),np.where(overflow,uyNorm*velocityThreshold,uy)
def EqVelocity(ux,uy):
    ustack = np.stack((ux,uy,ux*ux,uy*uy,ux*uy),axis=2)
    umulti= np.array([
        [0,0,-1.5,-1.5,0],
        [3,0,3,-1.5,0],
        [0,3,-1.5,3,0],
        [-3,0,3,-1.5,0],
        [0,-3,-1.5,3,0],
        [3,3,3,3,9],
        [-3,3,3,3,-9],
        [-3,-3,3,3,9],
        [3,-3,3,3,-9]
    ]).T
    return 1+np.moveaxis(ustack@umulti, -1, 0)

# Collide particles within each cell to redistribute velocities (could be optimized a little more):
def collide():
    global rho1, ux1, uy1, flow1, rho2, ux2, uy2, flow2
    rho1 = np.sum(flow1,axis=0)
    ux1 = np.divide(np.sum(flow1*np.expand_dims(np.array([0,1,0,-1,0,1,-1,-1,1]),axis=(1,2)),axis=0), rho1, out=np.zeros((height,width)), where=rho1!=0)
    uy1 = np.divide(np.sum(flow1*np.expand_dims(np.array([0,0,1,0,-1,1,1,-1,-1]),axis=(1,2)),axis=0), rho1, out=np.zeros((height,width)), where=rho1!=0)
    rho2 = np.sum(flow2,axis=0)
    ux2 = np.divide(np.sum(flow2*np.expand_dims(np.array([0,1,0,-1,0,1,-1,-1,1]),axis=(1,2)),axis=0), rho2, out=np.zeros((height,width)), where=rho2!=0)
    uy2 = np.divide(np.sum(flow2*np.expand_dims(np.array([0,0,1,0,-1,1,1,-1,-1]),axis=(1,2)),axis=0), rho2, out=np.zeros((height,width)), where=rho2!=0)
    
    # intermocular force
    phi1=np.sqrt(np.clip(rho1,0,None))
    phi2=np.sqrt(np.clip(rho2,0,None))
    dphi1dx, dphi1dy = GradientX(phi1),GradientY(phi1)
    dphi2dx, dphi2dy = GradientX(phi2),GradientY(phi2)
    F_11_x=2*phi1*dphi1dx
    F_11_y=2*phi1*dphi1dy
    F_22_x=2*phi2*dphi2dx
    F_22_y=2*phi2*dphi2dy
    F_12_x=-2*g12/np.sqrt(g11*g22)*phi1*dphi2dx
    F_12_y=-2*g12/np.sqrt(g11*g22)*phi1*dphi2dy
    F_21_x=-2*g21/np.sqrt(g11*g22)*phi2*dphi1dx
    F_21_y=-2*g21/np.sqrt(g11*g22)*phi2*dphi1dy
    
    # surface tension
    GLphi1x,GLphi1y = SecondGradient(dphi1dx, dphi1dy)
    GLphi2x,GLphi2y = SecondGradient(dphi2dx, dphi2dy)
    F_s1_x=2*kappa1/c0/g11*phi1*GLphi1x
    F_s1_y=2*kappa1/c0/g11*phi1*GLphi1y
    F_s2_x=2*kappa2/c0/g22*phi1*GLphi2x
    F_s2_y=2*kappa2/c0/g22*phi1*GLphi2y
    
    # add the force into u
    ux1+=np.divide((F_11_x*iF1+F_12_x+F_s1_x)/omega1, rho1, out=np.zeros((height,width)), where=rho1!=0)
    uy1+=np.divide((F_11_y*iF1+F_12_y+F_s1_y)/omega1, rho1, out=np.zeros((height,width)), where=rho1!=0)
    ux2+=np.divide((F_22_x*iF2+F_21_x+F_s2_x)/omega2, rho2, out=np.zeros((height,width)), where=rho2!=0)
    uy2+=np.divide((F_22_y*iF2+F_21_y+F_s2_y)/omega2, rho2, out=np.zeros((height,width)), where=rho2!=0)
    
    # normalization
    ux1,uy1 = NormalizeVelocity(ux1,uy1)
    ux2,uy2 = NormalizeVelocity(ux2,uy2)
    
    # applied force
    global forceY, forceX
    thickness = 1
    if np.sqrt(forceY*forceY + forceX*forceX)>0.1:
        ux1[chopstickY-thickness:chopstickY+thickness+1,chopstickX-thickness:chopstickX+thickness+1] = forceX * velocityThreshold
        uy1[chopstickY-thickness:chopstickY+thickness+1,chopstickX-thickness:chopstickX+thickness+1] = forceY * velocityThreshold
        ux2[chopstickY-thickness:chopstickY+thickness+1,chopstickX-thickness:chopstickX+thickness+1] = forceX * velocityThreshold
        uy2[chopstickY-thickness:chopstickY+thickness+1,chopstickX-thickness:chopstickX+thickness+1] = forceY * velocityThreshold
        forceY*=0.9
        forceX*=0.9
    else:
        forceY, forceX = 0, 0
    
    # update flow
    flow1*=(1-omega1)
    flow2*=(1-omega2)
    temp1=EqVelocity(ux1,uy1)
    temp2=EqVelocity(ux2,uy2)
    flow1+= omega1*np.expand_dims(w,axis=(1,2))*rho1*temp1
    flow2+= omega2*np.expand_dims(w,axis=(1,2))*rho2*temp2

# Move all particles by one step along their directions of motion (pbc):
def UpdateFlowPosition(flow):
    flow[1] = np.roll(flow[1], 1, axis=1)
    flow[2] = np.roll(flow[2], 1, axis=0)
    flow[3] = np.roll(flow[3], -1, axis=1)
    flow[4] = np.roll(flow[4], -1, axis=0)
    flow[5] = np.roll(flow[5], (1,1), axis=(0,1))
    flow[6] = np.roll(flow[6], (1,-1), axis=(0,1))
    flow[7] = np.roll(flow[7], (-1,-1), axis=(0,1))
    flow[8] = np.roll(flow[8], (-1,1), axis=(0,1))
    # Use tricky boolean arrays to handle barrier collisions (bounce-back):
    flow[4,vacancyN] = flow[2,barrierN]
    flow[2,vacancyS] = flow[4,barrierS]
    flow[3,vacancyE] = flow[1,barrierE]
    flow[1,vacancyW] = flow[3,barrierW]
    flow[7,vacancyNE] = flow[5,barrierNE]
    flow[5,vacancySW] = flow[7,barrierSW]
    flow[8,vacancyNW] = flow[6,barrierNW]
    flow[6,vacancySE] = flow[8,barrierSE]
    # clean the flow on contour
    flow[1,barrierE] = 0
    flow[2,barrierN] = 0
    flow[3,barrierW] = 0
    flow[4,barrierS] = 0
    flow[5,barrierNE] = 0
    flow[6,barrierNW] = 0
    flow[7,barrierSW] = 0
    flow[8,barrierSE] = 0
    # apply Force, require improvement

chopstick, chopstickY,chopstickX, forceY, forceX = False, 0, 0, 0, 0
def click_and_crop(event, x, y, flags, param):
    global chopstick, chopstickY, chopstickX,  forceY, forceX
    # print(x,y,event,flags)
    if flags == 0:
        return
    if event == 0:
        pass
    elif event == 4:
        chopstick = False
        chopstickY,chopstickX, forceY, forceX = 0, 0, 0, 0
    elif event == 1:
        chopstick = True
        chopstickY,chopstickX = y, x
    forceY = y - chopstickY
    forceX = x - chopstickX
    forceMagnitude = np.sqrt(forceY*forceY + forceX*forceX)
    if forceMagnitude>0:
        forceY/=forceMagnitude
        forceX/=forceMagnitude
    chopstickY,chopstickX = y, x

sizeIndex=3
sizeWidth=[150,200,400,600,900]
fileName='WaterOil'
cv2.namedWindow(fileName,cv2. WINDOW_KEEPRATIO)
cv2.resizeWindow(fileName,sizeWidth[sizeIndex],sizeWidth[sizeIndex])
cv2.setMouseCallback(fileName, click_and_crop)
blank=np.zeros((height,width))
while True:
    for step in range(10):
        collide()
        UpdateFlowPosition(flow1)
        UpdateFlowPosition(flow2)
    rho1_n=np.clip(rho1/1.5,0,1)
    rho2_n=np.clip(rho2/1.5,0,1)
    twoComp=np.stack((rho1_n,blank,rho2_n),axis=2) #it is bgr
    cv2.imshow(fileName, twoComp)
    key = cv2.waitKey(1)
    #q or esc for terminate the program
    if key == ord("q") or key == 27:
        break
    #+ for increasing size of circle and line
    elif key == 61:
        sizeIndex=min(sizeIndex+1,len(sizeWidth)-1)
        cv2.resizeWindow(fileName,sizeWidth[sizeIndex],sizeWidth[sizeIndex])
    #r for decreasing size of circle and line
    elif key == 45:
        sizeIndex=max(sizeIndex-1,0)
        cv2.resizeWindow(fileName,sizeWidth[sizeIndex],sizeWidth[sizeIndex])
cv2.destroyAllWindows()