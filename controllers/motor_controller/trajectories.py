#!/usr/bin/env python3

import numpy as np
import json
import math
import argparse
from abc import abstractmethod
import traceback

from numpy.core.fromnumeric import reshape

import robots


def buildTrajectory(type_name, start, knots, parameters=None):
    if type_name == "ConstantSpline":
        return ConstantSpline(knots, start)
    if type_name == "LinearSpline":
        return LinearSpline(knots, start)
    if type_name == "CubicZeroDerivativeSpline":
        return CubicZeroDerivativeSpline(knots, start)
    if type_name == "CubicWideStencilSpline":
        return CubicWideStencilSpline(knots, start)
    if type_name == "CubicCustomDerivativeSpline":
        return CubicCustomDerivativeSpline(knots, start)
    if type_name == "NaturalCubicSpline":
        return NaturalCubicSpline(knots, start)
    if type_name == "PeriodicCubicSpline":
        return PeriodicCubicSpline(knots, start)
    if type_name == "TrapezoidalVelocity":
        if parameters is None:
            raise RuntimeError("Parameters can't be None for TrapezoidalVelocity")
        return TrapezoidalVelocity(knots, parameters["vel_max"], parameters["acc_max"], start)
    raise RuntimeError("Unknown type: {:}".format(type_name))


def buildTrajectoryFromDictionary(dic):
    return buildTrajectory(dic["type_name"], dic["start"], np.array(dic["knots"]), dic.get("parameters"))


def buildRobotTrajectoryFromDictionary(dic):
    model = robots.getRobotModel(dic["model_name"])
    return RobotTrajectory(model, np.array(dic["targets"]), dic["trajectory_type"],
                           dic["target_space"], dic["planification_space"],
                           dic["start"], dic.get("parameters"))


def buildRobotTrajectoryFromFile(path):
    with open(path) as f:
        return buildRobotTrajectoryFromDictionary(json.load(f))


class Trajectory:
    """
    Describe a one dimension trajectory. Provides access to the value
    for any degree of the derivative

    Parameters
    ----------
    start: float
        The time at which the trajectory starts
    end: float or None
        The time at which the trajectory ends, or None if the trajectory never
        ends
    """
    def __init__(self, start=0):
        """
        The child class is responsible for setting the attribute end, if the
        trajectory is not periodic
        """
        self.start = start
        self.end = None

    @abstractmethod
    def getVal(self, t, d):
        """
        Computes the value of the derivative of order d at time t.

        Notes:
        - If $t$ is before (resp. after) the start (resp. after) of the
         trajectory, returns:
          - The position at the start (resp. end) of the trajectory if d=0
          - 0 for any other value of $d$

        Parameters
        ----------
        t : float
            The time at which the position is requested
        d : int >= 0
            Order of the derivative. 0 to access position, 1 for speed,
            2 for acc, etc...

        Returns
        -------
        x : float
            The value of derivative of degree d at time t.
        """

    def getStart(self):
        return self.start

    def getEnd(self):
        return self.end


class Spline(Trajectory):
    """
    Attributes
    ----------
    knots : np.ndarray shape (N,2+)
        The list of timing for all the N via points:
        - Column 0 represents time points
        - Column 1 represents the position
        - Additional columns might be used to specify other elements
          (e.g derivative)
    coeffs : np.ndarray shape(N-1,K+1)
        A list of n-1 polynomials of degree $K$: The polynomial at slice $i$ is
        defined as follows: $S_i(t) = \\sum_{j=0}^{k}coeffs[i,j] * (t-t_i)^(k-j)$
    """

    def __init__(self, knots, start=0):
        super().__init__(start)
        self.knots = knots
        self.start = start
        self.end = knots[-1,0]
        self.coeffs = np.ndarray((np.shape(knots)[0],4))
        self.updatePolynomials()
        
    @abstractmethod
    def updatePolynomials(self):
        """
        Updates the polynomials based on the knots and the interpolation method
        """

    def getDegree(self):
        """
        Returns
        -------
        d : int
            The degree of the polynomials used in this spline
        """
        return self.coeffs.shape[1]

    def getPolynomial(self, t):
        """
        Parameters
        ----------
        t : float
           The time at which the polynomial is requested

        Returns
        -------
        adjusted_t : float
            Normalized time for the slice considered
        p : np.ndarray shape(k+1,)
            The coefficients of the polynomial at time t, see coeffs
        """

        n = 0
        for k in range(self.knots.shape[0]-1):
            if self.knots[k,0] < t and self.knots[k+1,0] > t:
                n = k
        

        adjusted_t = t - self.knots[k,0]
        
        p = self.coeffs
        
        return adjusted_t, p

    def getVal(self, t, d=0):

        adjusted_t, coeffs = self.getPolynomial(t)
        pdegree = self.getDegree()
        sum = 0
        n = 0

        for k in range(self.knots.shape[0]-1):
            if self.knots[k,0] < t and self.knots[k+1,0] > t:
                n = k
        

        # slice_adjusted_t = adjusted_t - 
        # for k in range(self.knots.shape[0]-1):
        #     if  self.knots[k,0] > t:
        #         n= k*1


        adjusted_t = t 

        if(t<self.getStart()):
            
            if(d==0):
                for k in range(0,self.knots.shape[0]-1):
                    if self.knots[k,0] < self.start and self.knots[k+1,0] > self.start:
                        n = k
                return self.knots[n+1,1] 
            else:
                return 0

        
        if(t>self.getEnd()):
            
            if(d==0):
                for k in range(0,self.knots.shape[0]-1):
                    if self.knots[k,0] < self.start and self.knots[k+1,0] > self.start:
                        n = k                
                return self.knots[n-1,1]  
            else:
                return 0


        # print("\n adjust t =",adjusted_t)
        # print("\n etape n =",n)        

        if(d==0):
            for i in range(pdegree): 
                sum += coeffs[n,i]*adjusted_t**(i) 
                # print("for d = 0 coeff = {0}, puissance de t = {1}".format(coeffs[n,i],i))
        if(d==1):
            for i in range(pdegree-1):
                sum += (i+1) * coeffs[n,i+1]*adjusted_t**(i)
                # print("for d = 1 coeff = {0}, puissance de t = {1}".format( (i+1) *coeffs[n,i+1] ,i))
            if abs(sum) < 0.00001:
                sum = 0

        if(d==2):
            for i in range(pdegree-2):
                sum +=  (i+2)*(i+1) * coeffs[n,i+2]*adjusted_t**(i)
                # print("for d = 2 coeff = {0}, puissance de t = {1}".format( (i+2)*(i+1) *coeffs[n,i+2] ,i))
            if abs(sum )< 0.00001:
                sum = 0
        # print(self.coeffs)

        return sum


class ConstantSpline(Spline):
    def updatePolynomials(self):
        for i in range(self.knots.shape[0]):
            self.coeffs[i,0] = self.knots[i,1]


class LinearSpline(Spline):
    def updatePolynomials(self):
        for i in range(self.knots.shape[0]-1):
            a = (self.knots[i+1,1] - self.knots[i,1])/(self.knots[i+1,0] - self.knots[i,0]) 
            b = self.knots[i,1] - self.knots[i,0]*a
            self.coeffs[i,0] = b
            self.coeffs[i,1] = a

        self.coeffs[-1,0] = self.knots[-1,1] 
        # print(self.coeffs)
        


class CubicZeroDerivativeSpline(Spline):
    """
    Update polynomials ensuring derivative is 0 at every knot.
    """

    def updatePolynomials(self):

        for k in range(self.knots.shape[0]-1):
            # print(k)
            ti = self.knots[k,0]
            ti1 = self.knots[k+1,0]
            A = np.array([[ti**3, ti**2, ti, 1],[ti1**3, ti1**2, ti1, 1],[3*ti**2, 2*ti, 1 , 0],[3*ti1**2, 2*ti1, 1 , 0]])
            B = np.array([[self.knots[k,1], self.knots[k+1,1] ,0, 0]])[::-1]

            # print("ti =", ti)
            # print("ti1 =", ti1)

            # print("A = ", A)
            # print("B = ", B)

            res = np.linalg.solve(A,B.T)
            # print("res =",res)

            self.coeffs[k,:] = res.reshape((4,))[::-1]
        self.coeffs[-1,0] = self.knots[-1,1]

        # print(self.coeffs)


class CubicWideStencilSpline(Spline):
    """
    Update polynomials based on a larger neighborhood following the method 1
    described in http://www.math.univ-metz.fr/~croisil/M1-0809/2.pdf
    """
    #COMMENT LE TESTER?
    def updatePolynomials(self):
        shape = self.knots.shape[0]
        for k in range(shape-2):
            if(k==0):
                t0 = self.knots[k,0]
                t1 = self.knots[k+1,0]
                t2 = self.knots[k+2,0]
                t3 = self.knots[k+3,0]

                A = np.array([[t0**3, t0**2, t0, 1],[t1**3, t1**2, t1, 1],[t2**3, t2**2, t2, 1],[t3**3, t3**2, t3, 1]])

                B = np.array([[self.knots[k,1], self.knots[k+1,1] ,self.knots[k+2,1], self.knots[k+3,1]]])[::-1]

            # print(k)
            else:
                ti_moins_1 = self.knots[k-1,0]
                ti = self.knots[k,0]
                ti1 = self.knots[k+1,0]
                ti2 = self.knots[k+2,0]

                A = np.array([[ti_moins_1**3, ti_moins_1**2, ti_moins_1, 1],[ti**3, ti**2, ti, 1],[ti1**3, ti1**2, ti1, 1],[ti2**3, ti2**2, ti2, 1]])

                B = np.array([[self.knots[k-1,1], self.knots[k,1] ,self.knots[k+1,1], self.knots[k+2,1]]])[::-1]

            res = np.linalg.solve(A,B.T)
            # print("res =",res)

            self.coeffs[k,:] = res.reshape((4,))[::-1]
        
        #traitement de n-1
        tn_moins_3 = self.knots[shape-4,0]
        tn_moins_2 = self.knots[shape-3,0]
        tn_moins_1 = self.knots[shape-2,0]
        tn = self.knots[shape-1,0]

        A = np.array([[tn_moins_3**3, tn_moins_3**2, tn_moins_3, 1],[tn_moins_2**3, tn_moins_2**2, tn_moins_2, 1],[tn_moins_1**3, tn_moins_1**2, tn_moins_1, 1],[tn**3, tn**2, tn, 1]])
        #print(A)
        B = np.array([[self.knots[shape-4,1], self.knots[shape-3,1] ,self.knots[shape-2,1], self.knots[shape-1,1]]])[::-1]
        #print(B)
        res = np.linalg.solve(A,B.T)

        self.coeffs[shape-2,:] = res.reshape((4,))[::-1]

        #self.coeffs[-1,1] = self.knots[-1,1]

        # print(self.coeffs)


class CubicCustomDerivativeSpline(Spline):
    """
    For this splines, user is requested to specify the velocity at every knot.
    Therefore, knots is of shape (N,3)
    """
    def updatePolynomials(self):
        for k in range(self.knots.shape[0]-1):
            # print(k)
            ti = self.knots[k,0]
            ti1 = self.knots[k+1,0]
            A = np.array([[ti**3, ti**2, ti, 1],[ti1**3, ti1**2, ti1, 1],[3*ti**2, 2*ti, 1 , 0],[3*ti1**2, 2*ti1, 1 , 0]])
            B = np.array([[self.knots[k,1], self.knots[k+1,1] ,self.knots[k,2], self.knots[k+1,2]]])[::-1]

            res = np.linalg.solve(A,B.T)
            # print("res =",res)

            self.coeffs[k,:] = res.reshape((4,))[::-1]
        self.coeffs[-1,0] = self.knots[-1,1]

        # print(self.coeffs)



class NaturalCubicSpline(Spline):
    def updatePolynomials(self):
        n = self.knots.shape[0]
        # print("n =",n)
        S = np.zeros((4*n,4*n))
        B  = []
        # S1 = np.zeros((4*n,4*n))
        # S2 = np.zeros((4*n,4*n))
        for k in range(0,n-1):

            ti = self.knots[k,0]
            ti1 = self.knots[k+1,0]              

            # print(ti)                             
            A1 = np.array([[ti**3, ti**2, ti, 1],[ti1**3, ti1**2, ti1, 1]])
            A2 = np.array([[3*ti1**2, 2*ti1, 1 , 0, -3*ti1**2, -2*ti1, -1 , 0],[6*ti1, 2 , 0, 0, -6*ti1, -2 , 0 , 0]])

            #print("slicing = {0} : {1} , {2}  : {3}".format(2*k , 2*k+2    ,   4*k, 4*(k+1)))
            S[2*k       : 2*k+2     , 4*k : 4*(k+1)] = A1
            S[2*n +2*k  : 2*n+2*k+2 , 4*k : 4*(k+2)] = A2


            B.append( self.knots[k,1])
            B.append( self.knots[k+1,1]) 

        for i in range(0,2*n):
            B.append(0)

        # Last condition
        for i in range(0,2):
            B.append(0)

        B = np.array(B)

        # Bordure
        ti = self.knots[0,0]                                         
        S[-2,0] = 6 * ti
        S[-2,1] = 2

        res= np.linalg.pinv(S)@B.T

        for k in range(n-1):
            self.coeffs[k,:] = res[4*k:4*(k+1)][::-1]
        # print( self.coeffs ) 


class PeriodicCubicSpline(Spline):
    """
    Describe global splines where position, 1st order derivative and second
    derivative are always equal on both sides of a knot. This i
    """
    def updatePolynomials(self):
        n = self.knots.shape[0]
        S = np.zeros((4*n,4*n))
        B  = []
        for k in range(0,n-1):

            ti = self.knots[k,0]
            ti1 = self.knots[k+1,0]              

            # print(ti)                             
            A1 = np.array([[ti**3, ti**2, ti, 1],[ti1**3, ti1**2, ti1, 1]])
            A2 = np.array([[3*ti1**2, 2*ti1, 1 , 0, -3*ti1**2, -2*ti1, -1 , 0],[6*ti1, 2 , 0, 0, -6*ti1, -2 , 0 , 0]])

            #print("slicing = {0} : {1} , {2}  : {3}".format(2*k , 2*k+2    ,   4*k, 4*(k+1)))
            S[2*k       : 2*k+2     , 4*k : 4*(k+1)] = A1
            S[2*n +2*k  : 2*n+2*k+2 , 4*k : 4*(k+2)] = A2


            B.append( self.knots[k,1])
            B.append( self.knots[k+1,1]) 

        for i in range(0,2*n+2):
            B.append(0)

        B = np.array(B)
        
        # Bordure
        ti = self.knots[0,0]                                         
        S[-1,0] = 6 * ti  
        S[-1,1] = 2

        tn = self.knots[n-1,0]  
        S[-1,-4] = -6 * tn
        S[-1,-3] = -2

        S[-2,0:4] =np.array([3*ti**2, 2*ti, 1 , 0]) 
  
        S[-2,4*n-4 : 4*n] = - np.array([3*tn**2, 2*tn, 1 , 0])

        # print(S)
        res= np.linalg.pinv(S)@B.T
        # res = np.linalg.solve(S,B.T)


        for k in range(n-1):
            self.coeffs[k,:] = res[4*k:4*(k+1)][::-1]
        # print( self.coeffs ) 

    def getVal(self, t, d=0):
        adjusted_t, coeffs = self.getPolynomial(t)
        pdegree = self.getDegree()
        sum = 0
        n = 0

        for k in range(self.knots.shape[0]-1):
            if self.knots[k,0] < t and self.knots[k+1,0] > t:
                n = k
        

        # slice_adjusted_t = adjusted_t - 
        # for k in range(self.knots.shape[0]-1):
        #     if  self.knots[k,0] > t:
        #         n= k*1


        adjusted_t = t 

        if(t<self.getStart()):
            
            if(d==0):
                for k in range(0,self.knots.shape[0]-1):
                    if self.knots[k,0] < self.start and self.knots[k+1,0] > self.start:
                        n = k
                return self.knots[n+1,1] 
            else:
                return 0

        
        if(t>self.getEnd()):
            
            if(d==0):
                for k in range(0,self.knots.shape[0]-1):
                    if self.knots[k,0] < self.start and self.knots[k+1,0] > self.start:
                        n = k                
                return self.knots[n-1,1]  
            else:
                return 0

   

        if(d==0):
            for i in range(pdegree): 
                sum += coeffs[n,i]*adjusted_t**(i) 
                # print("for d = 0 coeff = {0}, puissance de t = {1}".format(coeffs[n,i],i))
        if(d==1):
            for i in range(pdegree-1):
                sum += (i+1) * coeffs[n,i+1]*adjusted_t**(i)
                # print("for d = 1 coeff = {0}, puissance de t = {1}".format( (i+1) *coeffs[n,i+1] ,i))
            if abs(sum) < 0.00001:
                sum = 0

        if(d==2):
            for i in range(pdegree-2):
                sum +=  (i+2)*(i+1) * coeffs[n,i+2]*adjusted_t**(i)
                # print("for d = 2 coeff = {0}, puissance de t = {1}".format( (i+2)*(i+1) *coeffs[n,i+2] ,i))
            if abs(sum )< 0.00001:
                sum = 0
        # print(self.coeffs)

        return sum


class TrapezoidalVelocity(Trajectory):
    def __init__(self, knots, vMax, accMax, start):
        self.knots = knots
        self.vMax = vMax
        self.accMax = accMax
        self.start = start

        self.offset_t = -0.2
        self.end = 6
        self.n = 0
        self.t_src = 0 

    def getVal(self, t, d):

    

        xsrc = self.knots[self.n]
        xend = self.knots[self.n+1]
        D  =  (xend - xsrc)
        T = xsrc
        V = 0
        Acc = np.sign(D)*self.accMax

        if t > self.start:
                
            
            
            if (np.abs(D) >= (self.vMax**2/(self.accMax))): 
                Talpha = self.vMax/self.accMax
                Dalpha = (self.accMax * Talpha**2)/2
                Tf = 2*Talpha + (np.abs(D) - 2*Dalpha)/self.vMax
            
                
                if t < self.t_src + Talpha:
                    T = xsrc + np.sign(D)*self.accMax*(t- self.t_src)*(t- self.t_src)/2
                    V = np.sign(D)*min(self.accMax * (t- self.t_src),self.vMax)




                elif t > self.t_src + Tf - Talpha:
                    T = xend - np.sign(D)*self.accMax*( Tf - t+ self.t_src)*( Tf - t+ self.t_src )/2
                    V = np.sign(D)*min(self.accMax * (Tf- t + self.t_src ),self.vMax)

                else:
                    V = np.sign(D)*self.vMax
                    T = xsrc + np.sign(D)*(Dalpha + self.vMax*(t - Talpha- self.t_src))
                
                
            else : 
                print("else")
                Talpha = np.sqrt(np.abs(D)/self.accMax)
                Dalpha = (self.accMax * Talpha**2)/2
                Tf = 2*Talpha 

                
                T = xsrc + np.sign(D)*(Dalpha + self.vMax*(t - Talpha - self.t_src))
                

                if t < self.t_src +Talpha:
                    T = xsrc + np.sign(D)*self.accMax*(t- self.t_src)*(t- self.t_src)/2
                    V = np.sign(D)*min(self.accMax * (t- self.t_src),self.vMax)

                if t >= self.t_src + Tf - Talpha:
                    T = xend - np.sign(D)*self.accMax*( Tf - t+ self.t_src)*( Tf - t+ self.t_src )/2
                    V = np.sign(D)*min(self.accMax * (Tf-t+ self.t_src),self.vMax)

    
            if t > self.t_src + Tf :

                if  self.n < self.knots.shape[0]-2:
                    self.n += 1
                    self.t_src += Tf
                
                T = xend
                V = 0

        
        else : 

            T = xsrc
            V = 0 
            Acc = 0


        if (d == 0) : 

            return T

        if (d == 1):            

            return V



        if(d == 2):

            

            if np.abs(D) <= np.abs(xsrc - T):
                return 0
            else:
                return Acc






            


            
        
        


class RobotTrajectory:
    """
    Represents a multi-dimensional trajectory for a robot.

    Attributes
    ----------
    model : control.RobotModel
        The model used for the robot
    planification_space : str
        Two space in which trajectories are planified: 'operational' or 'joint'
    trajectories : list(Trajectory)
        One trajectory per dimension of the planification space
    """

    supported_spaces = ["operational", "joint"]

    def __init__(self, model, targets, trajectory_type,
                 target_space, planification_space,
                 start=0, parameters=None):
        """
        model : robots.RobotModel
            The model of the robot concerned by this trajectory
        targets : np.ndarray shape(m,n) or shape(m,n+1)
            The multi-dimensional knots for the trajectories. One row concerns one
            target. Each column concern one of the dimension of the target space.
            For trajectories with specified time points (e.g. splines), the first
            column indicates time point.
        target_space : str
            The space in which targets are provided: 'operational' or 'joint'
        trajectory_type : str
            The name of the class to be used with trajectory
        planification_space : str
            The space in which trajectories are defined: 'operational' or 'joint'
        start : float
            The start of the trajectories [s]
        parameters : dictionary or None
            A dictionary containing extra-parameters for trajectories
        """
        self.robot = model
        self.targets = targets
        self.trajectory_type= trajectory_type
        self.target_space = target_space
        self.planification_space = planification_space
        self.start = 0
        self.parameters = None  
        
        self.joints = []
        self.traj = buildTrajectory(type_name, start, knots,parameters=parameters)




    def getVal(self, t, dim, degree, space):
        """
        Parameters
        ----------
        t : float
            The time at which the value is requested
        dim : int
            The dimension index
        degree : int
            The degree of the derivative requested (0 means position)
        space : str
            The space in which the value is requested: 'operational' or 'joint'

        Returns
        -------
        v : float or None
            The value of derivative of order degree at time t on dimension dim
            of the chosen space, None if computation is not implemented or fails
        """

        if space == "operational" :
            self.target = self.getPlanificationVal(t,degree)
            return self.joints[dim]

        if space == "joint" :
            return self.targets[dim]


       


        
        r

    def getPlanificationVal(self, t, degree):
        self.traj.updatePolynomials()


        return None

    def getOperationalTarget(self, t):
        return self.target

    def getJointTarget(self, t):

        self.joint =  self.robot.computeMGI(self.joints, self.target)
        return self.joints

    def getOperationalVelocity(self, t):

        self.target_vel = self.traj.getVal(t, 1)
        return None

    def getJointVelocity(self, t):

        self.joint_vel = self.robot.computeMGI(self.joints, self.target_vel)

        return None

    def getOperationalAcc(self, t):
        self.target_Acc = self.traj.getVal(t, 2)
        return None

    def getJointAcc(self, t):
        self.joint_vel = self.robot.computeMGI(self.joints, self.target_vel)
        return None

    def getStart(self):
        return self.start

    def getEnd(self):
        return self.end


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--robot", help="Consider robot trajectories and not 1D trajectories",
                        action="store_true")
    parser.add_argument("--degrees",
                        type=lambda s: np.array([int(item) for item in s.split(',')]),
                        default=[0, 1, 2],
                        help="The degrees of derivative to plot")
    parser.add_argument("trajectories", nargs="+", type=argparse.FileType('r'))
    args = parser.parse_args()
    trajectories = {}
    tmax = 0
    tmin = 10**10
    for t in args.trajectories:
        try:
            if args.robot:
                trajectories[t.name] = buildRobotTrajectoryFromDictionary(json.load(t))
            else:
                trajectories[t.name] = buildTrajectoryFromDictionary(json.load(t))
            tmax = max(tmax, trajectories[t.name].getEnd())
            tmin = min(tmin, trajectories[t.name].getStart())
        except KeyError:
            print("Error while building trajectory from file {:}:\n{:}".format(t.name, traceback.format_exc()))
            exit()
    order_names = ["position", "velocity", "acceleration", "jerk"]
    print("source,t,order,variable,value")
    for source_name, trajectory in trajectories.items():
        for t in np.arange(tmin - args.margin, tmax + args.margin, args.dt):
            for degree in args.degrees:
                order_name = order_names[degree]
                if (args.robot):
                    space_dims = {
                        "joint": trajectory.model.getJointsNames(),
                        "operational": trajectory.model.getOperationalDimensionNames()
                    }
                    for space, dim_names in space_dims.items():
                        for dim in range(len(dim_names)):
                            v = trajectory.getVal(t, dim, degree, space)
                            if v is not None:
                                print("{:}, {:}, {:}, {:}, {:}".format(source_name, t, order_name, dim_names[dim], v))
                else:
                    v = trajectory.getVal(t, degree)
                    print("{:}, {:}, {:}, {:}, {:}".format(source_name, t, order_name, "x", v))
