Title: Linear Algebra Basics with Numpy
Date: 2016-10-25 15:00
Modified: 2016-10-25 23:00
Category: Data Science
Tags: linear algebra, numpy, python
Slug: linear_algebra_basic
Authors: Kimi Yuan
Summary: Basic operands like addition and multiplication between vectors and matrices and also use Numpy for calculation.

[TOC]

# Matrices and Vectors

## Definitions

**Matrix**: Rectaugular array of numbers.

**Dimension of matrix**: number of rows x number of columns.

**Matrix Elements** ( entries of matrix)

$A_{ij}$ = "i, j element " in the $^{ith}$ row, $j^{th}$ column.
$$
A = \begin{bmatrix}
	 1402 & 191\\
	 1371& 821 \\
	 949& 1437 \\
          147 & 1448
	 \end{bmatrix}
$$
A is 4 rows x 2 columns matrix.


$A_{11} = 1402$

$A_{12} = 191$

$A_{32} = 1437$

$A_{41} = 147$


**Vector**: An n x 1 matrix.
$$
y = \begin{bmatrix}
	1 \\
	2 \\
	3 \\
	4
	\end{bmatrix}
$$
where y is a 4 dimension vector and $y_i = i^{th}$ element.



## Code Example

```python
In [1]: import numpy as np

In [2]: matrix_a = np.array([[1402,191], [1371, 821],
                             [949,1437], [147, 1448]])

In [3]: matrix_a
Out[3]:
array([[1402,  191],
       [1371,  821],
       [ 949, 1437],
       [ 147, 1448]])

In [4]: matrix_a.shape
Out[4]: (4, 2)

In [5]: matrix_a[0][0]	# matrix element of row 1 and colomn 1
Out[5]: 1402

In [6]: matrix_a[0][1]	# matrix element of row 1 and colomn 2
Out[6]: 191

In [7]: matrix_a[2][1]	# matrix element of row 3 and colomn 2
Out[7]: 1437

In [8]: matrix_a[3][0]	# matrix element of row 4 and colomn 1
Out[8]: 147
```



# Addition and Scalar Multiplication

**Matrix Addition**

3 x 2 matrix  add  3 x 2 matrix equals 3 x 2 matrix
$$
\begin{bmatrix}
1 & 0 \\
2 & 5 \\
3 & 1
\end{bmatrix}  + \begin{bmatrix}
4 & 0.5 \\
2 & 5 \\
0 & 1
\end{bmatrix}  = \begin{bmatrix}
5 & 0.5 \\
4 & 10 \\
3 & 2
\end{bmatrix}
$$
3 x 2 matrix add 2 x 2 matrix gets ERROR.

```python
In [9]: np.array([[1,0], [2,5], [3,1]]) +
                np.array([[4,0.5], [2,5], [0,1]])
Out[9]:
array([[  5. ,   0.5],
       [  4. ,  10. ],
       [  3. ,   2. ]])

In [10]: np.array([[1,0], [2,5], [3,1]]) +
                 np.array([[4,0.5], [2,5]])
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-10-4c8e1d407c3f> in <module>()
----> 1 np.array([[1,0], [2,5], [3,1]]) + np.array([[4,0.5], [2,5]])

ValueError: operands could not be broadcast together with shapes (3,2) (2,2)
```



**Scalar Multiplication**


$$
3 \times \begin{bmatrix} 1 & 0 \\ 2 & 5 \\ 3 & 1 \end{bmatrix} = \begin{bmatrix} 3 & 0 \\ 6 & 15 \\ 9 & 3 \end{bmatrix}  = \begin{bmatrix} 1 & 0 \\ 2 & 5 \\ 3 & 1 \end{bmatrix} \times 3
$$
​			
**Combination of Operands**
$$
3 \times \begin{bmatrix} 1 \\ 4 \\ 2 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \\ 5 \end{bmatrix} - \begin{bmatrix} 3 \\ 0 \\ 2 \end{bmatrix} /3
$$

```python
In [11]: 3 * np.array([[1],[4],[2]]) + np.array([[0],[0],[5]]) -
    		np.array([[3],[0],[2]])/3
Out[11]:
array([[  2.        ],
       [ 12.        ],
       [ 10.33333333]])
```



# Matrix - Vector Multiplication

**Details**:			
$A \times    x = y$

m x n matrix *times*  n x 1 matrix (n-dimensional vector) equals m-dimensional vector.
$$
\begin{bmatrix} A_{11} & A_{12} & \dots & A_{1n} \\ A_{21} & A_{22} & \dots & A_{2n} \\ \vdots & \vdots & \dots & \vdots\\
A_{m1} & A_{m2} &\dots & A_{mn} \end{bmatrix} \times \begin{bmatrix} x_1 \\ x_2 \\\vdots \\x_{n-1}\\ x_n \end{bmatrix} =
\begin{bmatrix} y_1 \\ y_2 \\\vdots \\ y_m \end{bmatrix}
$$
To get $y_i$, multiply $A$'s $i^ith$ row with elements of vector $x$, and add them up.

**Example**:
$$
\begin{bmatrix} 1 & 3 \\ 4 & 0 \\ 2 & 1\end{bmatrix} \begin{bmatrix} 1 \\ 5\end{bmatrix}  = \begin{bmatrix} 16 \\4 \\ 7\end{bmatrix}
$$

```python
In [12]: np.matmul(np.array([[1,3], [4,0], [2,1]]), np.array([[1],[5]]))
Out[12]:
array([[16],
       [ 4],
       [ 7]])
```



# Matrix - Matrix Multiplication

**Details**:			
$A \times    B = C$

m x n matrix *times*  n x o matrix  equals m x o  matrix.
$$
\begin{bmatrix} A_{11} & A_{12} & \dots & A_{1n} \\ A_{21} & A_{22} & \dots & A_{2n} \\ \vdots & \vdots & \dots & \vdots\\
A_{m1} & A_{m2} &\dots & A_{mn} \end{bmatrix} \times \begin{bmatrix} B_{11}& B_{12} & \dots & B_{1,o-1}& B_{1o} \\ B_{21} & B_{22} & \dots & B_{2,o-1}& B_{2o}\\ \vdots & \vdots &\dots &\vdots &\vdots\\B_{n-1,1} &  B_{n-1,2} &\dots &B_{n-1,o-1}& B_{n-1,o}  \\B_{n,1} &  B_{n,2} &\dots &B_{n,o-1}& B_{n,o} \end{bmatrix} = \begin{bmatrix} C_{11} & C_{12} & \dots & C_{1o} \\ C_{21} & C_{22} & \dots & C_{2o} \\ \vdots & \vdots & \dots & \vdots\\
C_{m1} & C_{m2} &\dots & C_{mo} \end{bmatrix}
$$
The $i^ith$ column of the matrix C is obtained by multiplying A with the $i^ith$ column of B. (for i = 1,2, …, o)

**Example**:
$$
\begin{bmatrix} 1 & 3 &2 \\ 4 & 0 &1\end{bmatrix} \begin{bmatrix} 1 & 3  \\ 0 &1\\5 &2\end{bmatrix} =
\begin{bmatrix} 11 & 10  \\ 9 &14\end{bmatrix}
$$

```python
In [14]: np.matmul(np.array([[1,3,2], [4,0,1]]), np.array([[1,3],[0,1],[5,2]]))
Out[14]:
array([[11, 10],
       [ 9, 14]])
```



# Matrix Multiplication Properties

**not commutative**: Let A and B be matrices. Then in general, $ A \times B \ne B \times A$

**Associative**: A x (B x C) = (A x B) x C		
​					

## Identity Matrix

Denoted $I$ (or $I_{n\times n}$), examples:
$$
\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

$$
\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 &0 &1 \end{bmatrix}
$$


$$
\begin{bmatrix} 1 & 0 & 0 &0 \\ 0 & 1 &0& 0 \\ 0 &0 &1&0 \\ 0&0&0&1 \end{bmatrix}
$$

For any matrix A,

$A_{mn} \times I_{nn} = I_{mm} \times A_{mn} = A_{mn}$



```python
In [15]: np.identity(3)
Out[15]:
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
```



# Inverse and Transpose

## Inverse

If A is an m x m matrix, and if it has an inverse $A^{-1}$,

$A (A^{-1}) = A^{-1} A = I$



```python
In [17]: matrix_a = np.array([[3,4],[2,16]])

In [18]: matrix_a_inverse = np.linalg.inv(matrix_a)

In [19]: np.matmul(matrix_a, matrix_a_inverse)
Out[19]:
array([[ 1.,  0.],
       [ 0.,  1.]])
```



## Transpose

Let A be an m x n matrix, and let B = A^T^,  then B is an n x m matrix and $B_{ij} =A_{ji}$.

Example:
$$
A = \begin{bmatrix} 1 & 2 &0\\3&5&9\end{bmatrix}
$$

$$
B = A^T = \begin{bmatrix} 1 & 3 \\ 2&5\\0&9\end{bmatrix}
$$



```python
In [20]: matrix_a = np.array([[1,2,0],[3,5,9]])

In [21]: matrix_a.T
Out[21]:
array([[1, 3],
       [2, 5],
       [0, 9]])
```
