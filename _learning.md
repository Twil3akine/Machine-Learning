## L2ノルム
$$
\begin{align}
    \boldsymbol{x} 
    &=
    \begin{bmatrix}
        x_{1} \\ x_{2}
    \end{bmatrix}
    \\
    \boldsymbol{y} 
    &=
    \begin{bmatrix}
        y_{1} \\ y_{2}
    \end{bmatrix}
    \\
    \boldsymbol{x} - \boldsymbol{y} 
    &=
    \begin{bmatrix}
        x_{1} - y_{1} \\ 
        x_{2} - y_{2}
    \end{bmatrix}
    \\
    \|\boldsymbol{x}-\boldsymbol{y}\|
    &=
    \sqrt{{(x_{1}-y_{1})}^{2} + {(x_{2}-y_{2})}^{2}}
\end{align}
$$

## 内積
$$
\begin{align}
    \boldsymbol{a} \cdot \boldsymbol{b}
    &=
    \boldsymbol{a}^{\top}\boldsymbol{b}=
    \begin{bmatrix}
        a_{1},a_{2}
    \end{bmatrix}
    \begin{bmatrix}
        b_{1} \\ b_{2}
    \end{bmatrix}
    =a_{1}b_{1}+a_{2}b_{2}
    =\sum\limits_{i=1}^{2}a_{i}b_{i}
\end{align}
$$

## 行列の積
$$
\begin{align}
    \begin{bmatrix}
        x_{11} & \cdots & x_{1D} \\
        \vdots & \ddots & \vdots \\
        x_{M1} & \cdots & x_{MD}
    \end{bmatrix}
    \begin{bmatrix}
        y_{11} & \cdots & y_{1N} \\
        \vdots & \ddots & \vdots \\
        y_{D1} & \cdots & y_{DN}
    \end{bmatrix}
    &=
    \begin{bmatrix}
        a_{11} & \cdots & a_{1N} \\
        \vdots & \ddots & \vdots \\
        a_{M1} & \cdots & a_{MN}
    \end{bmatrix} \\
    &=
    \begin{bmatrix}
        \sum\limits_{j=1}^{D}x_{1j}y_{j1} & 
        \cdots & 
        \sum\limits_{j=1}^{D}x_{1j}y_{jN}
        \\
        \vdots & 
        \ddots & 
        \vdots 
        \\
        \sum\limits_{j=1}^{D}x_{Mj}y_{j1} & 
        \cdots & 
        \sum\limits_{j=1}^{D}x_{Mj}y_{jN}
    \end{bmatrix}
\end{align}
$$