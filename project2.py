import math
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt

class FluidFlow:
    def __init__(self, V_0, nu, a, b, c, d, L, w):
        self.V_0 = V_0
        self.nu = nu
        self.a = a #large side length
        self.b = b #length before boundary starts
        self.c = c #width of obstruction
        self.d = d #height of obstruction
        self.L = L # L is number of grid points along a
        self.h = a/L
        self.w = w # w is the overrelaxation factor
        self.e_points = math.ceil(L*b/a)+1 #number points on E boundary
        self.c_points = math.ceil(L*c/a)+1 #'' ''
        self.a_points = int(L*(a-b-c)/a)+1#'' ''
        self.f_points=L+1#'' ''
        self.d_points=int(L*d/a)
        self.h_points=math.ceil(L*(a-d)/a)+1 #number of points above obstruction
        print("e_points: " + str(self.e_points))
        print("c_points: " + str(self.c_points))
        print("a_points: " + str(self.a_points))
        print("f_points: " + str(self.f_points))
        print("d_points: " + str(self.d_points))
        print("h_points: " + str(self.h_points))

    def create_matrix(self):
        L=self.L
        #grid before obstruction
        N = self.get_superindex(self.L, self.L) + 1
        phi_matrix = np.zeros((N, N)) #stream function matrix
        omega_matrix = np.zeros((N, N)) #vorticity matrix
        phi_b = np.zeros((N))
        omega_b = np.zeros((N))
        # fix all boundary conditions in which phi or omega are zero
        for I in range(N):
            i, j = self.get_coords(I)
            if self.onboundary(I):
                phi_matrix[I][I] = 1.0
                omega_matrix[I][I] = 1.0
                # b[I] = boundary(I)

            else:
                A[I][I] = -4.0
                A[I][self.get_superindex(i, j+1)] = 1.0
                A[I][self.get_superindex(i, j-1)] = 1.0
                A[I][self.get_superindex(i+1, j)] = 1.0
                A[I][self.get_superindex(i-1, j)] = 1.0
                b[I] = self.omega_matrix[I][I]
        self.A = A

    def initialize_free_flow(self):
        N = self.get_superindex(self.L, self.L) + 1
        for I in range(N):
            i, j = self.get_coords(I)
            self.phi_matrix[I][I] = 1
            self.omega_matrix[I][I] = 1
            self.phi_b[I] = V_0 * a*j/L
            self.omega_b[I] = 0

    def fix_boundaries(self):
        V_0=self.V_0
        L = self.L
        # fix A and E boundaries
        for i in range(L+1):
            if(i < self.e_points or i > self.e_points + self.c_points-2):
                I = self.get_superindex(i, 0)
                self.phi_b[I] = 0
            else: #C bound
                prev_I = self.get_superindex()
                self.omega_b[I] = -2/(self.h**2))

        # fix B and D boundaries
        for
        # fix G boundary
        for i in range(L+1):
            I = self.get_superindex(i, L)
            self.phi_matrix[I][I] = 1
            self.omega_matrix[I][I] = 1
            self.phi_b[I] = V_0 * a
            self.omega_b[I] = 0
        #



    def onboundary(self,I,fulloutput=False): #checks if point on onboundary
        if fulloutput==False: #just checks if it's on a boundary
            onbound=False
            L=self.L
            i,j=self.get_coords(I)
            if i==0 or i==L or j==0 or j==L: #F boundary
                onbound=True
            elif i==self.e_points-1 or i==self.e_points+self.c_points-2:
                #D,B bounds
                onbound=True
            elif j==self.d_points-1: #C bounds
                onbound=True
            return onbound
        else:
            if i==0:


    def indextest(self):
        N = self.get_superindex(self.L, self.L) + 1
        for I in range(N):
            i, j = self.get_coords(I)
            print(str(I) + " (i, j) = "  + str(i) + ", " + str(j) + "  I=" + str(self.get_superindex(i, j)))


    def get_superindex(self, i, j):
        e_points = self.e_points
        c_points = self.c_points
        a_points = self.a_points
        f_points=self.f_points
        d_points=self.d_points
        h_points=self.h_points
        if i < e_points: #shifted to 0, incudes D boundary
            return i*f_points + j
        elif i < e_points + c_points - 2: #above obstruction
            if j >= d_points:
                Nbefore=f_points*e_points
                new_i = i - e_points
                new_j = j - d_points
                return Nbefore + new_i*(h_points) + new_j
            else:
                raise Exception("Inaccessible index")
        else:
            Nbefore = f_points*e_points + h_points*(c_points-2)
            new_i = i - e_points - (c_points - 2)
            return Nbefore + new_i*(f_points) + j

    def get_coords(self, I): #gets coordinates from superindex I
        e_points = self.e_points
        c_points = self.c_points
        a_points = self.a_points
        f_points=self.f_points
        d_points=self.d_points
        h_points=self.h_points
        if I < f_points*e_points:
            i = int(I/f_points)
            j = I%(f_points)
            return i,j
        elif I < f_points*e_points + h_points*(c_points-2):
            new_I = I - f_points*e_points
            i = int(new_I/h_points) + e_points
            j = new_I%(h_points) + d_points
            return i,j
        else:
            new_I = I - f_points*e_points - h_points*(c_points-2)
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
    front_of_plate = 0.33
    back_of_plate = 0.66
    top_of_plate = 0.33
    L = 3
    w = 1.5
    print()
    ff = FluidFlow(V_0, nu, size_of_plate, front_of_plate,
        back_of_plate - front_of_plate, top_of_plate, L, w)
    ff.create_matrix()
    print(ff.A)

if __name__ == "__main__":
    main()
