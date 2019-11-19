import math
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
import sys

class FluidFlow:
    # b is length before boundary starts
    # c is width of obstruction
    # d is height of obstruction
    def __init__(self, V_0, nu, a, b, c, d, L):
        self.V_0 = V_0
        self.nu = nu
        self.a = a #large side length

        self.L = L # L is number of grid points along each side
        self.h = a/L # h is the grid spacing

        self.before_plate = math.ceil(L*b/a)
        self.on_plate = math.ceil(L*c/a)
        self.after_plate = int(L*(a-b-c)/a)
        assert(self.before_plate + self.on_plate + self.after_plate == L + 1)

        self.plate_height = int(L*d/a)

        self.psi_matrix = np.zeros((L+1,L+1)) #stream function matrix
        self.omega_matrix = np.zeros((L+1,L+1)) #vorticity matrix
        self.residual_matrix = np.zeros((L+1,L+1)) #residual matrix
        self.sortboundaries()

     # n is the number of relaxations performed, w is the overrelaxation factor
    def SOR(self, n, w = 1.5):
        self.initialize_free_flow()
        self.apply_boundary_cond(["A", "B", "C", "D", "E", "F", "G", "H"])
        self.print_psi()
        self.print_omega()
        for i in range(n):
            self.update_psi_interior(w)
            self.update_omega_interior(w)
            self.apply_boundary_cond(["B", "C", "D", "H"])
            self.print_psi()
            self.print_omega()

    def update_psi_interior(self, w):
        for point in self.Interior_points:
            i, j = point[0], point[1]
            self.psi_matrix[i][j] = (1-w)*self.psi_matrix[i][j] + \
                (w/4)*(self.psi_matrix[i+1][j] + \
                self.psi_matrix[i-1][j] + \
                self.psi_matrix[i][j+1] + \
                self.psi_matrix[i][j-1] - \
                self.omega_matrix[i][j])

    def update_omega_interior(self, w):
        old_omega_matrix = self.omega_matrix
        for point in self.Interior_points:
            i, j = point[0], point[1]
            d_psi_d_y = (self.psi_matrix[i][j+1] - self.psi_matrix[i][j-1])/(2*self.h)
            d_omega_d_y = (old_omega_matrix[i][j+1] - old_omega_matrix[i][j-1])/(2*self.h)
            d_psi_d_x = (self.psi_matrix[i+1][j] - self.psi_matrix[i-1][j])/(2*self.h)
            d_omega_d_x = (old_omega_matrix[i+1][j] - old_omega_matrix[i-1][j])/(2*self.h)

            self.omega_matrix[i][j] = (1-w)*self.omega_matrix[i][j] + \
                (w/4)*(self.omega_matrix[i+1][j] + self.omega_matrix[i-1][j] + \
                self.omega_matrix[i][j+1] + self.omega_matrix[i][j-1] - \
                (1/(self.nu))*(d_psi_d_y*d_omega_d_x - d_psi_d_x*d_omega_d_y))

    def compute_residual(self):
        for point in self.Interior_points:
            i, j = point[0], point[1]
            self.residual_matrix[i][j] = -4*self.psi_matrix[i][j] + \
            self.psi_matrix[i+1][j] + self.psi_matrix[i-1][j] + \
            self.psi_matrix[i][j+1] + self.psi_matrix[i][j-1] + \
            self.omega_matrix[i][j]

    def residual_norm(self):
        sum = 0
        for i in range(self.L+1):
            for j in range(self.L+1):
                sum += self.residual_matrix[i][j]*self.residual_matrix[i][j]
        return math.sqrt(sum)

    def initialize_free_flow(self):
        for point in self.Interior_points:
            i, j = point[0], point[1]
            self.psi_matrix[i][j] = self.V_0 * self.a*j/self.L
            self.omega_matrix[i][j] = 0

    def apply_boundary_cond(self, boundaries):
        for boundary in boundaries:
            if boundary == "A":
                for coords in self.Abound_points:
                    i, j = coords[0], coords[1]
                    self.psi_matrix[i][j] = 0
                    self.omega_matrix[i][j] = 0
            if boundary == "E":
                for coords in self.Ebound_points:
                    i, j = coords[0], coords[1]
                    self.psi_matrix[i][j] = 0
                    self.omega_matrix[i][j] = 0
            if boundary == "C":
                for coords in self.Cbound_points:
                    i, j = coords[0], coords[1]
                    psi = self.psi_matrix[i][j+1]
                    self.omega_matrix[i][j] = -2/(self.h**2)*psi
                    self.psi_matrix[i][j] = 0
            if boundary == "B":
                for coords in self.Bbound_points:
                    i, j = coords[0], coords[1]
                    psi = self.psi_matrix[i+1][j]
                    self.psi_matrix[i][j] = 0
                    self.omega_matrix[i][j] = -2/(self.h**2)*psi
            if boundary == "D":
                for coords in self.Dbound_points:
                    i, j = coords[0], coords[1]
                    psi = self.psi_matrix[i-1][j]
                    self.psi_matrix[i][j] = 0
                    self.omega_matrix[i][j] = -2/(self.h**2)*psi
            if boundary == "G": #set to free flow conditions
                for coords in self.Gbound_points:
                    i, j = coords[0], coords[1]
                    self.psi_matrix[i][j] = self.V_0 * self.a*j/self.L
                    self.omega_matrix[i][j] = 0
            if boundary == "F": #set to free flow conditions
                for coords in self.Fbound_points:
                    i, j = coords[0], coords[1]
                    self.psi_matrix[i][j] = self.V_0 * self.a*j/self.L
                    self.omega_matrix[i][j] = 0
            if boundary == "H":
                for coords in self.Hbound_points:
                    i, j = coords[0], coords[1]
                    self.psi_matrix[i][j] = self.psi_matrix[i-1][j]
                    self.omega_matrix[i][j] = self.omega_matrix[i-1][j]


    def sortboundaries(self): #Imax is total number of verticies
        #sorts the verticies into different lists
        L=self.L
        self.Interior_points=[]
        self.Plate_interior_points=[]
        self.Abound_points=[]
        self.Bbound_points=[]
        self.Cbound_points=[]
        self.Dbound_points=[]
        self.Ebound_points=[]
        self.Fbound_points=[]
        self.Gbound_points=[]
        self.Hbound_points=[]
        for i in range(self.L+1):
            for j in range(self.L+1):
                if i==0: #F boundary
                    self.Fbound_points.append((i,j))
                elif i==L: #H boundary
                    self.Hbound_points.append((i,j))
                elif j==L: #G boundary
                    self.Gbound_points.append((i,j))
                elif j==0 and i<self.before_plate: #E boundary
                    self.Ebound_points.append((i,j))
                elif j==0 and i>=self.before_plate+self.on_plate: #A boundary
                    self.Abound_points.append((i,j))
                elif i >= self.before_plate and i<self.before_plate+self.on_plate   \
                        and j<=self.plate_height: #grid points on the plate
                    if self.plate_height == 0:
                        self.Ebound_points.append((i,j))
                    elif j==self.plate_height:
                        self.Cbound_points.append((i,j)) #C boundary
                    elif i==self.before_plate:
                        self.Dbound_points.append((i,j)) #D boundary
                    elif i==self.before_plate+self.on_plate-1:
                        self.Bbound_points.append((i,j)) #B boundary
                    else:
                        self.Plate_interior_points.append((i,j))
                else:
                    self.Interior_points.append((i,j))


    def print_psi(self):
        matrix = np.zeros((self.L+1, self.L+1))
        for i in range(self.L+1):
            for j in range(self.L+1):
                matrix[self.L - j][i] = self.psi_matrix[i][j]
        np.set_printoptions(precision=3)
        print('\u03C8:')
        print(matrix)

    def print_omega(self):
        matrix = np.zeros((self.L+1, self.L+1))
        for i in range(self.L+1):
            for j in range(self.L+1):
                matrix[self.L - j][i] = self.omega_matrix[i][j]
        np.set_printoptions(precision=2)
        print('\u03C9:')
        print(matrix)

    def graph_residual(self):
        x_values = np.linspace(0, 1.0, self.L+1)
        y_values = np.linspace(0, 1.0, self.L+1)
        X, Y = np.meshgrid(x_values, y_values)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, self.residual_matrix, cmap="viridis")
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'Residual')
        ax.set_title(r"Surface plot of the residual versus $x$ and $y$")
        plt.show()

def main():
    V_0 = 1
    nu = 0.1
    region_dim = 1.0
    front_of_plate = 0.25
    back_of_plate = 0.55
    top_of_plate = 0.5
    L = int(sys.argv[1])
    ff = FluidFlow(V_0, nu, region_dim, front_of_plate,
        back_of_plate - front_of_plate, top_of_plate, L)
    num_relaxations = int(sys.argv[2])
    ff.SOR(num_relaxations)
    ff.compute_residual()
    print("residual norm: " + str(ff.residual_norm()))
    ff.graph_residual()


if __name__ == "__main__":
    main()
