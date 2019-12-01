import math
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
import sys

# The FluidFlow class models the fluid flow around an obstruction by creating
# a grid that computes values of psi and vorticity at each discrete gridpoint.
# These values are computed using the Successive Overrelation Method (SOR).
class FluidFlow:
    # a is the length of the square plate, b is length before boundary starts,
    # c is width of obstruction, and d is height of obstruction
    def __init__(self, V_0, nu, a, b, c, d, L):
        self.V_0 = V_0
        self.nu = nu
        self.a = a
        self.L = L # L is number of grid points along each side
        self.h = a/L # h is the grid spacing

        self.before_plate = math.ceil(L*b/a)
        self.on_plate = math.ceil(L*c/a)
        self.after_plate = L + 1 - (self.before_plate + self.on_plate)
        self.plate_height = int(L*d/a)

        self.sort_points()
        self.initialize_grid()

    # This function sorts each vertex as a pair of coordinates into whatever
    # boundary it lies on or into the interior.
    def sort_points(self):
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
                        self.Plate_interior_points.append((i,j)) # interior of plate
                else:
                    self.Interior_points.append((i,j))


    #  Initializes matrices storing values of psi, omega, and the residual.
    def initialize_grid(self):
        self.psi_matrix = np.zeros((self.L+1,self.L+1)) #stream function matrix
        self.omega_matrix = np.zeros((self.L+1,self.L+1)) #vorticity matrix
        self.residual_matrix = np.zeros((self.L+1,self.L+1)) #residual matrix
        self.initialize_free_flow()
        self.apply_boundary_conditions(["A", "E", "F", "G", "H"])

    # Initializes the interior of the psi and omega matrices to free flow
    # conditions.
    def initialize_free_flow(self):
        for point in self.Interior_points:
            i, j = point[0], point[1]
            self.psi_matrix[i][j] = self.V_0 * self.a*j/self.L
            self.omega_matrix[i][j] = 0

    # Accepts an array of strings, each of which corresponds to a boundary
    # at which the associated conditions of that boundary are applied.
    def apply_boundary_conditions(self, boundaries):
        for boundary in boundaries:
            if boundary == "A":
                for point in self.Abound_points:
                    i, j = point[0], point[1]
                    self.psi_matrix[i][j] = 0
                    self.omega_matrix[i][j] = 0
            if boundary == "E":
                for point in self.Ebound_points:
                    i, j = point[0], point[1]
                    self.psi_matrix[i][j] = 0
                    self.omega_matrix[i][j] = 0
            if boundary == "C":
                for point in self.Cbound_points:
                    i, j = point[0], point[1]
                    psi = self.psi_matrix[i][j+1]
                    self.omega_matrix[i][j] = -2/(self.h**2)*psi
                    self.psi_matrix[i][j] = 0
            if boundary == "B":
                for point in self.Bbound_points:
                    i, j = point[0], point[1]
                    psi = self.psi_matrix[i+1][j]
                    self.omega_matrix[i][j] = -2/(self.h**2)*psi
                    self.psi_matrix[i][j] = 0
            if boundary == "D":
                for point in self.Dbound_points:
                    i, j = point[0], point[1]
                    psi = self.psi_matrix[i-1][j]
                    self.omega_matrix[i][j] = -2/(self.h**2)*psi
                    self.psi_matrix[i][j] = 0
            if boundary == "G": #set to free flow conditions
                for point in self.Gbound_points:
                    i, j = point[0], point[1]
                    self.psi_matrix[i][j] = self.V_0 * self.a*j/self.L
                    self.omega_matrix[i][j] = 0
            if boundary == "F": #set to free flow conditions
                for point in self.Fbound_points:
                    i, j = point[0], point[1]
                    self.psi_matrix[i][j] = self.V_0 * self.a*j/self.L
                    self.omega_matrix[i][j] = 0
            if boundary == "H":
                for point in self.Hbound_points:
                    i, j = point[0], point[1]
                    self.psi_matrix[i][j] = self.psi_matrix[i-1][j]
                    self.omega_matrix[i][j] = self.omega_matrix[i-1][j]
            if boundary == "Obstruction":
                for point in self.obstruction_points:
                    i, j = point[0], point[1]
                    self.psi_matrix[i][j] = 0
                    self.omega_matrix[i][j] = 0

     # This function performs n relaxations using an over-relaxation factor w.
     # Each relaxation consists of the following sequence of events: update psi
     # interior, apply boundary conditions along the boundary, update omega
     # interior, and lastly apply the boundary conditions along the back wall.
    def SOR(self, n, w = 1.5):
        for i in range(n):
            self.update_psi_interior(w)
            self.apply_boundary_conditions(["B", "C", "D"])
            self.update_omega_interior(w)
            self.apply_boundary_conditions(["H"])
        self.update_residual()

    # This function updates the interior values of psi according to the
    # overrelaxation factor w.
    def update_psi_interior(self, w):
        for point in self.Interior_points:
            i, j = point[0], point[1]
            self.psi_matrix[i][j] = (1-w)*self.psi_matrix[i][j] + \
                (w/4)*(self.psi_matrix[i+1][j] + \
                self.psi_matrix[i-1][j] + \
                self.psi_matrix[i][j+1] + \
                self.psi_matrix[i][j-1] + \
                self.h*self.h*self.omega_matrix[i][j])

    # This function updates the interior values of omega according to the
    # overrelaxation factor w.
    def update_omega_interior(self, w):
        for point in self.Interior_points:
            i, j = point[0], point[1]
            d_psi_d_y = (self.psi_matrix[i][j+1] - self.psi_matrix[i][j-1])/(2*self.h)
            d_omega_d_y = (self.omega_matrix[i][j+1] - self.omega_matrix[i][j-1])/(2*self.h)
            d_psi_d_x = (self.psi_matrix[i+1][j] - self.psi_matrix[i-1][j])/(2*self.h)
            d_omega_d_x = (self.omega_matrix[i+1][j] - self.omega_matrix[i-1][j])/(2*self.h)

            self.omega_matrix[i][j] = (1-w)*self.omega_matrix[i][j] + \
                (w/4)*(self.omega_matrix[i+1][j] + self.omega_matrix[i-1][j] + \
                self.omega_matrix[i][j+1] + self.omega_matrix[i][j-1] + \
                (self.h*self.h/self.nu)*(d_psi_d_y*d_omega_d_x - d_psi_d_x*d_omega_d_y))

    # This function computes the residual in the interior points and stores
    # these values in self.residual_matrix.
    def update_residual(self):
        for point in self.Interior_points:
            i, j = point[0], point[1]
            self.residual_matrix[i][j] = (-4*self.psi_matrix[i][j] + \
            self.psi_matrix[i+1][j] + self.psi_matrix[i-1][j] + \
            self.psi_matrix[i][j+1] + self.psi_matrix[i][j-1])/(self.h*self.h) + \
            self.omega_matrix[i][j]

    # This function returns the overall norm of the residual matrix.
    def residual_norm(self):
        sum = 0
        for i in range(self.L+1):
            for j in range(self.L+1):
                sum += self.residual_matrix[i][j]*self.residual_matrix[i][j]
        return math.sqrt(sum)

    def graph_residual(self):
        x_values = np.linspace(0, 1.0, self.L+1)
        y_values = np.linspace(0, 1.0, self.L+1)
        X, Y = np.meshgrid(x_values, y_values)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, self.residual_matrix, cmap="viridis")

        ax.set_xlabel(r'$y$')
        ax.set_ylabel(r'$x$')
        ax.set_zlabel(r'Residual')
        ax.set_title(r"Surface plot of the residual versus $x$ and $y$")
        plt.show()

    def fluid_flow_contour_plot(self):
        x_values = np.linspace(0, 1.0, self.L+1)
        y_values = np.linspace(0, 1.0, self.L+1)
        X, Y = np.meshgrid(x_values, y_values)

        fig,ax=plt.subplots(1,1)
        levels = np.arange(0,1.1,0.05)
        cp = ax.contourf(Y, X, self.psi_matrix,levels=levels)
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Filled Contours Plot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(r"$\psi$")
        plt.show()

    def vorticity_contour_plot(self):
        x_values = np.linspace(0, 1.0, self.L+1)
        y_values = np.linspace(0, 1.0, self.L+1)
        X, Y = np.meshgrid(x_values, y_values)

        fig,ax=plt.subplots(1,1)
        levels = np.arange(-1,1,0.05)
        cp = ax.contourf(Y, X, 0.1*self.omega_matrix)
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Filled Contours Plot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(r"$\omega$")
        plt.show()

    def residual_norm_vs_w(self, w_0, w_f, n, num_relaxations, graphs=False):
        w_values = np.linspace(w_0, w_f, n)
        res_norm_values = np.zeros((n))
        for i in range(n):
            self.initialize_free_flow()
            self.SOR(num_relaxations, w_values[i])
            res_norm_values[i] = self.residual_norm()
        print("optimal value of w: " + str(w_values[np.argmin(res_norm_values)]))
        if graphs==True:
            plt.plot(w_values, res_norm_values)
            plt.xlabel(r'$w$')
            plt.ylabel(r'Residual Norm $R$')
            plt.title(r'Plot of Residual Norm $R$ as a function of $w$')
            plt.show()

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

def main():
    V_0 = 1
    nu = 0.1
    region_dim = 1.0
    front_of_plate = 0.25
    back_of_plate = 0.55
    top_of_plate = float(sys.argv[4])
    w=float(sys.argv[3])
    L = int(sys.argv[1])
    ff = FluidFlow(V_0, nu, region_dim, front_of_plate,
        back_of_plate - front_of_plate, top_of_plate, L)
    num_relaxations = int(sys.argv[2])

    ff.SOR(num_relaxations,w)

    #ff.graph_residual()
    ff.fluid_flow_contour_plot()
    ff.vorticity_contour_plot()
    #ff.residual_norm_vs_w(1, 2, 20, num_relaxations)


if __name__ == "__main__":
    main()
