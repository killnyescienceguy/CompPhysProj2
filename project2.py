import math
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt

class FluidFlow:
    def __init__(self, V_0, nu, a, b, c, d, L):
        self.V_0 = V_0
        self.nu = nu
        self.a = a #large side length
        self.b = b #length before boundary starts
        self.c = c #width of obstruction
        self.d = d #height of obstruction
        self.L = L # L is number of grid points along a
        self.e_points = int(L*b/a)+1 #number points on E boundary
        self.c_points = int(L*c/a)+1 #'' ''
        self.a_points = int(L*(a-b-c)/a)+1#'' ''
        self.f_points=L+1#'' ''
        self.d_points=int(L*d/a)+1
        self.height_p=int(L*(a-d)/a)+1 #number of points above obstruction

    def create_matrix(self):
        L=self.L
        #grid before obstruction
        N = self.get_superindex(self.L, self.L) + 1
        A = np.zeros((N, N))
        b = np.zeros((N))
        for I in range(N):
            i, j = self.get_coords(I)
            if self.onboundary(I):
                A[I][I] = 1.0
                # b[I] = boundary(I)
            else:
                A[I][I] = -4.0
                A[I][self.get_superindex(i, j+1)] = 1.0
                A[I][self.get_superindex(i, j-1)] = 1.0
                A[I][self.get_superindex(i+1, j)] = 1.0
                A[I][self.get_superindex(i-1, j)] = 1.0
        self.A = A

    def onboundary(self,I): #checks if point on onboundary
        L=self.L
        i,j=self.get_coords(I)
        if i==0 or i==L or j==0 or j==L: #F,H,E,A,G bounds
            return True
        elif i==self.e_points-1 or i==self.e_points+self.c_points-2:
            #D,B bounds
            return True
        elif j==self.d_points-1: #C bounds
            return True
        else:
            return False

    def indextest(self):
        N = self.get_superindex(self.L, self.L) + 1
        for I in range(N):
            i, j = self.get_coords(I)
            print(str(I) + " " + str(self.get_superindex(i, j)))


    def get_superindex(self, i, j):
        L=self.L
        a=self.a
        b=self.b
        c=self.c
        d=self.d
        e_points = self.e_points
        c_points = self.c_points
        a_points = self.a_points
        f_points=self.f_points
        d_points=self.d_points
        height_p=self.height_p
        if i < e_points: #shifted to 0, incudes D boundary
            return i*f_points + j
        elif i < e_points + c_points - 2: #above obstruction
            if j >= d_points:
                Nbefore=f_points*e_points
                new_i = i - e_points - 1
                new_j = j - d_points - 1
                return Nbefore + new_i*(height_p) + new_j
            else:
                raise Exception("Inaccessible index")
        else:
            Nbefore = f_points*e_points + height_p*(c_points-2)
            new_i = i - e_points - (c_points - 2) - 1
            return Nbefore + new_i*(f_points) + j

    def get_coords(self, I): #gets coordinates from superindex I
        L=self.L
        a=self.a
        b=self.b
        c=self.c
        d=self.d
        e_points = int(L*b/a)+1 #number points on E boundary
        c_points = int(L*c/a)+1 #'' ''
        a_points = int(L*(a-b-c)/a)+1#'' ''
        f_points=L+1#'' ''
        d_points=int(L*d/a)+1
        height_p=int(L*(a-d)/a)+1 #points in space above obstruction
        i,j = 0,0
        if I <= f_points*e_points:
            i = int(I/f_points)
            j = I%(f_points)
        elif I <= f_points*e_points + height_p*(c_points-2):
            new_I = I - f_points*e_points
            i = int(new_I/height_p) + e_points
            j = new_I%(height_p) + d_points
        else:
            new_I = I - f_points*e_points - height_p*(c_points-2)
            i = int(new_I/f_points) + e_points + (c_points - 2)
            j = new_I%(f_points)
        return i,j





    # n is the number of iterations to perform SOR
    def SOR(n):
        return

def main():
    V_0 = 1
    nu = 0.1
    size_of_plate = 1.0
    front_of_plate = 0.25
    back_of_plate = 0.375
    top_of_plate = 0.25
    L = 16
    ff = FluidFlow(V_0, nu, size_of_plate, front_of_plate, back_of_plate - front_of_plate, top_of_plate, 4)
    # ff.create_matrix()
    # print(ff.A)
    ff.indextest()

if __name__ == "__main__":
    main()
