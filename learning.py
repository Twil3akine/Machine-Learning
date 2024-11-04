import numpy as np

class Input:
    def inputCR():
        print(f"Please input H,W")
        return map(int, input().split())
    
    def inputVector():
        print(F"Please 1*N of vector")
        vector = np.array(list(map(int,input().split())))
        return vector.reshape((len(vector)))

    def inputMatrix(C,R):
        print(f"Please input H*W of Grid")
        rlt = np.array([np.array(list(map(int, input().split()))) for c in range(C)])
        return rlt.reshape((C,R))
    
    def inputDataDiv2():
        print(f"please input 2*N(=y,x) of vector. when finish input, input 0")
        L = []
        cnt = 0
        while True:
            tmp = list(map(int,input().split()))
            if len(tmp)==1:
                if tmp[0]==0:
                    return np.array(L).reshape((cnt,2))
                else:
                    print(f"not allow that control")
                    continue
            elif len(tmp)==2:
                cnt += 1
                L.append(tmp)
            else:
                print(f"not allow that control")
                continue
            
    def inputDataDivOver2():
        print(f"please input number of total variable!")
        while True:
            num = tuple(map(int,input().split()))
            if len(num) != 1:
                print("please input again")
                continue
            else:
                break
            
        print(f"please input 2*N(=y,x...) of vector. when finish input, input 0")
        L = []
        cnt = 0
        while True:
            tmp = list(map(int,input().split()))
            if len(tmp)==1:
                if tmp[0]==0:
                    return np.array(L).reshape((cnt,num+1))
                else:
                    print(f"not allow that control")
                    continue
            elif len(tmp)==num:
                cnt += 1
                tmp.append(1)
                L.append(tmp)
            else:
                print(f"not allow that control")
                continue

class Vector:
    # 平均
    def mean(v):
        s = np.sum(v)
        return s/len(v)
    
    # 分散
    def variance(v):
        bar = Vector.mean(v)
        return Vector.mean((v-bar)**2)
    
    # 標準偏差
    def standardDeviation(v):
        return np.sqrt(Vector.variance(v))
    
    # 共分散
    def covariance(w,v):
        w_bar = Vector.mean(w)
        v_bar = Vector.mean(v)
        return Vector.mean((w-w_bar)*(v-v_bar))
    
    # 内積
    def inner_product(w,v):
        if len(w) != len(v): return "not same length!!"
        return np.sum(list(map(lambda x,y:x*y, w, v)))

    # L2ノルム
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
            plus,minus = np.zeros(3), np.zeros(3)
            for i in range(length):
                ptmp,mtmp = 1,1
                for j in range(length):
                    ptmp *= m[j][(i+j)%length]
                    mtmp *= m[length-1-j][j]
                plus[i][j] = ptmp
                minus[i][j] = mtmp
            return np.sum(plus)-np.sum(minus)
        else:
            return "Sorry, I can calc det of any equal or less than 3...(under developing)"
    
    def dot(m1, m2) -> np.ndarray:
        m1 = np.array(m1)
        m2 = np.array(m2)
        
        # 計算できない場合はエラー発生
        if m1.shape[1] != m2.shape[0]:
            return "not calc mul of m1, m2 !"
        
        c,r = m1.shape[0], m2.shape[1]
        
        rlt = np.zeros((c,r))
        for cc in range(c):
            for rr in range(r):
                rlt[cc][rr] = np.sum(Vector.inner_product(m1[cc], m2[:,rr]))
                
        return np.array(rlt).reshape((c,r))
    
class MachineLearning:
    # 回帰分析
    def regressionAnalysis(data, target=None):
        Y = data[:, 0]
        X = data[:, 1]
        w = Vector.covariance(X,Y)/Vector.variance(X)
        b = Vector.mean(Y)-(w*Vector.mean(X))
        
        if target==None: return (w,b)
        else: return (w*target+b)
        
    # 重回帰分析
    def multipleRegressionAnalysis(data, target=None):
        data = np.hstack((np.array(data), np.ones(len(data)).reshape(-1,1)))
        Y = np.array(data[:, 0]).reshape(-1, 1)
        Z = np.array(data[:, 1:])
        
        v = Matrix.dot(np.linalg.inv(Matrix.dot(Z.T, Z)), Matrix.dot(Z.T, Y))
        w = np.array(v[:-1]).flatten()
        b = v[-1,0]
        
        if target is None: return (w,b)
        else: return Matrix.dot(w,target) + b