from learning import *

# c,r = Input.inputCR()
# m = Input.inputMatrix(c,r)

c,r = 3,3
m = np.array([[1,0,0],
              [0,1,0],
              [0,0,1]])

print(Matrix.trace(m), Matrix.det(m))