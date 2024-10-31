import numpy as np

class Input:
    def inputCR():
        print(f"Please input H,W")
        return map(int, input().split())

    def inputMatrix(C,R):
        print(f"Please input H*W of Grid")
        rlt = np.array([np.array(list(map(int, input().split()))) for c in range(C)])
        return rlt.reshape((C,R))

    def inputVector():
        print(F"Please 1*N of vector")
        vector = np.array(list(map(int,input().split())))
        return vector.reshape((len(vector)))

class Vector:
    def inner_product(w,v) -> str | int | float:
        if len(w) != len(v): return "not same length!!"
        return np.sum(list(map(lambda x,y:x*y, w, v)))

    def l2norm(w, v) -> str | int | float:
        if len(w) != len(v): return "not same length!!"
        return np.sqrt(np.sum(list(map(lambda x,y:(x-y)**2, w, v))))

class Matrix:
    def trace(m: list[list[int|float]]) -> str | int | float:
        if len(m) != len(m[0]): return "not same length of column,row !!"
        return np.sum(np.array([m[i][i] for i in range(len(m))]))
    
    def det(m: list[int|float]) -> str | int | float:
        if len(m) != len(m[0]): return "not same length of column,row !!"
        length = len(m)
        if length==2:
            return np.float64(m[0][0]*m[1][1]-m[0][1]*m[1][0])
        if length==3:
            plus,minus = np.array([]),np.array([])
            for i in range(length):
                ptmp,mtmp = 1,1
                for j in range(length):
                    ptmp *= m[j][(i+j)%length]
                    mtmp *= m[length-1-j][j]
                plus = np.append(plus,ptmp)
                minus = np.append(minus,mtmp)
            return np.sum(plus)-np.sum(minus)
        else:
            pass
    
    def dot(m1: list[list[int|float]], m2: list[list[any]]) -> list[list[any]]:
        # 計算できない場合はエラー発生
        if m1.shape[1] != m2.shape[0]:
            return "not calc mul of m1, m2 !"
        
        c,r = m1.shape[0], m2.shape[1]
        
        rlt = np.array([])
        for cc in range(c):
            tmp = np.array([])
            for rr in range(r):
                tmp = np.append(tmp, np.sum(Vector.inner_product(m1[cc], m2[:,rr])))
                
            rlt = np.append(rlt, tmp)
        return np.array(rlt).reshape((c,r))