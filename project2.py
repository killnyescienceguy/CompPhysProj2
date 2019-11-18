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
        self.h = a/L
        self.e_points = math.ceil(L*b/a)+1 #number points on E boundary
        self.c_points = math.ceil(L*c/a)+1 #'' ''
        self.a_points = int(L*(a-b-c)/a)+1#'' ''
        self.f_points=L+1#'' ''
        self.d_points=int(L*d/a)
        self.h_points=math.ceil(L*(a-d)/a)+1 #number of points above obstruction
        # print("e_points: " + str(self.e_points))
        # print("c_points: " + str(self.c_points))
        # print("a_points: " + str(self.a_points))
        # print("f_points: " + str(self.f_points))
        # print("d_points: " + str(self.d_points))
        # print("h_points: " + str(self.h_points))
        self.N = self.get_superindex(L, L) + 1
        self.psi_values = np.zeros((self.N)) #stream function matrix
        self.omega_values = np.zeros((self.N)) #vorticity matrix
        self.sortboundaries()

     # n is the number of relaxations performed, w is the overrelaxation factor
    def SOR(self, n, w=1.5):
        self.initialize_free_flow()
        self.apply_boundary_cond(["A", "B", "C", "D", "E", "F", "G", "H"])
        for i in range(n):
            self.update_psi_interior()
            self.update_omega_interior()
            self.apply_boundary_cond(["B", "C", "D", "H"])

    def initialize_free_flow(self):
        for I in range(self.N):
            i, j = self.get_coords(I)
            self.psi_values[I] = self.V_0 * self.a*j/self.L
            self.omega_values[I] = 0

    def apply_boundary_cond(self, boundaries):
        for boundary in boundaries:
            if boundary == "A":
                for I in self.Abound_points:
                    self.psi_values[I] = 0
                    self.omega_values[I] = 0
            if boundary == "E":
                for I in self.Ebound_points:
                    self.psi_values[I] = 0
                    self.omega_values[I] = 0
            if boundary == "C":
                for I in self.Cbound_points:
                    i, j = self.get_coords(I)
                    psi = self.psi_values[self.get_superindex(i, j + 1)]
                    self.omega_values[I] = -2/(self.h**2)*psi
                    self.psi_values[I] = 0
            if boundary == "B":
                for I in self.Bbound_points:
                    i, j = self.get_coords(I)
                    psi = self.psi_values[self.get_superindex(i - 1, j)]
                    self.psi_values[I] = 0
                    self.omega_values[I] = -2/(self.h**2)*psi
            if boundary == "D":
                for I in self.Dbound_points:
                    i, j = self.get_coords(I)
                    psi = self.psi_values[self.get_superindex(i + 1, j)]
                    self.psi_values[I] = 0
                    self.omega_values[I] = -2/(self.h**2)*psi
            if boundary == "G": #set to free flow conditions
                for I in self.Gbound_points:
                    i, j = self.get_coords(I)
                    self.psi_values[I] = self.V_0 * self.a*j/self.L
                    self.omega_values[I] = 0
            if boundary == "F": #set to free flow conditions
                for I in self.Fbound_points:
                    i, j = self.get_coords(I)
                    self.psi_values[I] = self.V_0 * self.a*j/self.L
                    self.omega_values[I] = 0
            if boundary == "H":
                for I in self.Hbound_points:
                    i, j = self.get_coords(I)
                    I_left = self.get_superindex(i-1, j)
                    self.psi_values[I] = self.psi_values[I_left]
                    self.omega_values[I] = self.omega_values[I_left]

    def update_psi_interior(self, w):
        for I in self.Interior_points:
            i, j = self.get_coords(I)
            self.psi_values[I] = (1-w)*self.psi_values[I] + \
                (w/4)*(self.psi_values[self.get_superindex(i+1,j)] + \
                self.psi_values[self.get_superindex(i-1,j)] + \
                self.psi_values[self.get_superindex(i,j+1)] + \
                self.psi_values[self.get_superindex(i,j-1)] - \
                self.omega_values[I])

    def update_omega_interior(self, w):
        for I in self.Interior_points:
            i, j = self.get_coords(I)
            d_psi_d_y = self.psi_values[self.get_superindex(i,j+1)] - \
                self.psi_values[self.get_superindex(i,j-1)]
            d_omega_d_y = self.omega_values[self.get_superindex(i,j+1)] - \
                self.omega_values[self.get_superindex(i,j-1)]
            d_psi_d_x = self.psi_values[self.get_superindex(i+1,j)] - \
                self.psi_values[self.get_superindex(i-1,j)]
            d_omega_d_x = self.omega_values[self.get_superindex(i+1,j)] - \
                self.omega_values[self.get_superindex(i-1,j)]

            self.omega_values[I] = (1-w)*self.omega_values[I] + \
                (w/4)*(self.omega_values[self.get_superindex(i+1,j)] + \
                self.omega_values[self.get_superindex(i-1,j)] + \
                self.omega_values[self.get_superindex(i,j+1)] + \
                self.omega_values[self.get_superindex(i,j-1)] - \
                (1/(2*nu))*(d_psi_d_y*d_omega_d_x - d_psi_d_x*d_omega_d_y))


    def onboundary(self,I,specific=False): #checks if point on onboundary
        onbound=False
        L=self.L
        i,j=self.get_coords(I)
        if specific==False:
            if i==0 or i==L or j==0 or j==L: #F boundary
                onbound=True
            elif i==self.e_points-1 or i==self.e_points+self.c_points-2:
                #D,B bounds
                onbound=True
            elif j==self.d_points-1: #C bounds
                onbound=True
            return onbound
        else:
            if i==0: #F boundary
                whichbound='F'
            elif i==L: #H boundary
                whichbound='H'
            elif j==0 and i<=self.e_points-1: #E boundary
                whichbound='E'
            elif j==0 and i>self.e_points+self.c_points-2: #A boundary
                whichbound='A'
            elif j==L: #G boundary
                whichbound='G'
            elif j==self.e_points-1 and i<=self.d_points: #D boundary
                whichbound='D'
            elif j==self.e_points+self.c_points-2 and i<=self.d_points:
                whichbound='B'
            else:
                whichbound='C'
            return whichbound

    def sortboundaries(self): #Imax is total number of verticies
        #sorts the verticies into different lists
        self.Interior_points=[]
        self.Abound_points=[]
        self.Bbound_points=[]
        self.Cbound_points=[]
        self.Dbound_points=[]
        self.Ebound_points=[]
        self.Fbound_points=[]
        self.Gbound_points=[]
        self.Hbound_points=[]
        for I in range(self.N):
            if self.onboundary(I)==False:
                self.Interior_points.append(I)
            else:
                if self.onboundary(I,True)=='A':
                    self.Abound_points.append(I)
                elif self.onboundary(I,True)=='B':
                    self.Bbound_points.append(I)
                elif self.onboundary(I,True)=='C':
                    self.Cbound_points.append(I)
                elif self.onboundary(I,True)=='D':
                    self.Dbound_points.append(I)
                elif self.onboundary(I,True)=='E':
                    self.Ebound_points.append(I)
                elif self.onboundary(I,True)=='F':
                    self.Fbound_points.append(I)
                elif self.onboundary(I,True)=='G':
                    self.Gbound_points.append(I)
                elif self.onboundary(I,True)=='H':
                    self.Hbound_points.append(I)

    def indextest(self):
        N = self.get_superindex(self.L, self.L) + 1
        for I in range(self.N):
            i, j = self.get_coords(I)
            print(str(I) + " (i, j) = "  + str(i) + ", " + str(j) + "  I=" + str(self.get_superindex(i, j)))


    def get_superindex(self, i, j):
        if i < self.e_points: #shifted to 0, incudes D boundary
            return i*self.f_points + j
        elif i < self.e_points + self.c_points - 2: #above obstruction
            if j >= self.d_points:
                Nbefore=self.f_points*self.e_points
                new_i = i - self.e_points
                new_j = j - self.d_points
                return Nbefore + new_i*(self.h_points) + new_j
            else:
                raise Exception("Inaccessible index")
        else:
            Nbefore = self.f_points*self.e_points + self.h_points*(self.c_points-2)
            new_i = i - self.e_points - (self.c_points - 2)
            return Nbefore + new_i*(self.f_points) + j

    def get_coords(self, I): #gets coordinates from superindex I
        if I < self.f_points*self.e_points:
            i = int(I/self.f_points)
            j = I%(self.f_points)
            return i,j
        elif I < self.f_points*self.e_points + self.h_points*(self.c_points-2):
            new_I = I - self.f_points*self.e_points
            i = int(new_I/self.h_points) + self.e_points
            j = new_I%(self.h_points) + self.d_points
            return i,j
        else:
            new_I = I - self.f_points*self.e_points - self.h_points*(self.c_points-2)
            i = int(new_I/self.f_points) + self.e_points + (self.c_points - 2)
            j = new_I%(self.f_points)
            return i,j

    def print_psi(self):
        matrix = np.zeros((self.L+1, self.L+1))
        for I in range(self.N):
            i, j = self.get_coords(I)
            matrix[self.L - j][i] = self.psi_values[I]
        print(matrix)

def main():
    V_0 = 1
    nu = 0.1
    region_dim = 1.0
    front_of_plate = 0.0
    back_of_plate = 0.0
    top_of_plate = 0.0
    L = 3
    ff = FluidFlow(V_0, nu, region_dim, front_of_plate,
        back_of_plate - front_of_plate, top_of_plate, L)
    ff.SOR(0)
    print(ff.print_psi())

if __name__ == "__main__":
    main()
