---
title: Automatic Differentiation II
description: The more the merrier (especially dimensions)
---

In the last essay, we saw how we can calculate the gradients of scalar functions, by implementating backpropagation through a graph data structure. The implementation process was preceded by the theory behind both the graph data structure and the chain rule. The final algorithm was very useful in calculating *scalar* values, but we must not forget multi-dimensional inputs. At times (most of the times actually), we need gradients on vector valued functions, not scalar. Therefore, in this continuation essay, we shall explore *Vector calulus* and how it fits in Automatic Differentiation.

## Vector Calculus

There isn't as much difference in this topic, compared to normal calculus. Topics that are generally taught under this concept such as Divergence, Curl, Line Integrals .. etc. won't be covered here (anyone interested can explore<i><a href="https://www.khanacademy.org/math/multivariable-calculus" style="font-weight: 0;">Khan academy</a></i>, for their excellent courses), as they are not relevant in ML/DL. We shall only go over the general cases, and put them in our algorithm. A vector is simply a list  of numbers put together,

$$x=\begin{bmatrix}
x_{1} & x_{2} & x_{3} & \dotsc  & x_{n}
\end{bmatrix} ,\ x\ \epsilon \ \mathbb{R}^{n}$$

Where **R** denotes the set of all real numbers. What we are interested in, is what is the derivative of a vector with respect to another vector. As I covered this in another essay as well (Neural Networks through Linear Algebra), it results in what we call a *Jacobian*. Let *v* and *x* be two vectors,

$$ \begin{array}{l}
\mathbf{v} =\begin{bmatrix}
v_{1} & v_{2} & v_{3} & \dotsc  & v_{n}
\end{bmatrix} ,\ \mathbf{v} \ \epsilon \ \mathbb{R}^{n}\\\\\\\\
x=\begin{bmatrix}
x_{1} & x_{2} & x_{3} & \dotsc  & x_{m}
\end{bmatrix} ,\ x\ \epsilon \ \mathbb{R}^{m}
\end{array}$$

If we need to find out the derivative of **v** with respect to **x**, we get a *Jacobian Matrix*.

$$\frac{\delta \mathbf{v}}{\delta \mathbf{x}} =\begin{bmatrix}
\frac{\delta v_{1}}{\delta x_{1}} & \frac{\delta v_{2}}{\delta x_{1}} & \frac{\delta v_{3}}{\delta x_{1}} & \dotsc  & \frac{\delta v_{n}}{\delta x_{1}}\\\\
\frac{\delta v_{1}}{\delta x_{2}} &  &  &  & \\\\
\frac{\delta v_{1}}{\delta x_{3}} &  & \ddots  &  & \vdots \\\\
\vdots  &  &  &  & \\\\
\frac{\delta v_{1}}{\delta x_{m}} &  & \dotsc  &  & \frac{\delta v_{n}}{\delta x_{m}}
\end{bmatrix}_{n*m}$$

Generally when we deal with multi-dimensional data, we need to pay special attention to the dimensions. In the above example, we see how the matrix combines the dimensions of the two vectors, to give us an *n * m* dimensional matrix. The reason is simple: we simply calculated the derivative of each element in v with respect to each element in x, and put them all in the matrix. When it comes to the derivative of a matrix with respect to a vector, the same process applies, and we get a three-dimensional *Jacobian cube* structure. But a more easier process would be to simply reshape the matrix into a vector. Let **M** be a matrix, of dimensions *r * y*.

$$\mathbf{M} \ =\ \begin{bmatrix}
m_{11} & m_{21} & m_{31} & \dotsc  & m_{x1}\\\\
m_{12} &  &  &  & \\\\
m_{13} &  & \ddots  &  & \vdots \\\\
\vdots  &  &  &  & \\\\
m_{1y} &  & \dotsc  &  & m_{xy}
\end{bmatrix}_{r*y}$$

Reshaping the matrix into a vector would give us a vector of *ry* dimensions. This can be done because there exists an isomorphism (they are similiar, or have similiar/same properties) between **R**<sup>*r * y*</sup> and **R**<sup>*ry*</sup>. The matrix **M** after reshaping looks like this,

$$\mathbf{M} \ =\begin{bmatrix}
m_{11} & m_{12} & m_{13} & \dotsc  & m_{1y} & m_{21} & \dotsc  & m_{2y} & \dotsc  & m_{x1} & \dotsc  & m_{ry}
\end{bmatrix} \\\\\ \mathbf{M} \ \epsilon \ \mathbb{R}^{ry}$$

After reshaping, calculating the derivative with respect to any other vector is simply to follow the previous steps, to get a *Jacobian* matrix. 

$$\mathbf{M} \ =\begin{bmatrix}
m_{1} & m_{2} & m_{3} & \dotsc  & m_{ry}
\end{bmatrix} ,\ \ \mathbf{M} \ \epsilon \ \mathbb{R}^{ry}$$

$$\frac{\delta \mathbf{M}}{\delta \mathbf{x}} =\begin{bmatrix}
\frac{\delta m_{1}}{\delta x_{1}} & \frac{\delta m_{2}}{\delta x_{1}} & \frac{\delta m_{3}}{\delta x_{1}} & \dotsc  & \frac{\delta m_{ry}}{\delta x_{1}}\\\\\\\\
\frac{\delta m_{1}}{\delta x_{2}} &  &  &  & \\\\\\\\
\frac{\delta m_{1}}{\delta x_{3}} &  & \ddots  &  & \vdots \\\\\\\\
\vdots  &  &  &  & \\\\\\\\
\frac{\delta m_{1}}{\delta x_{m}} &  & \dotsc  &  & \frac{\delta m_{ry}}{\delta x_{m}}
\end{bmatrix}_{ry*m}$$

Doing so, we basically get the same Jacobian cube as before, as you can check with the dimensions (*r * y * m* = *ry * m*), because all we did was reshape the matrix for easier calculations and understanding. With this, we can move forward to see how to backpropagate with vector-valued functions.

## Backpropagation with Vectors

As we did before, it would benefit us if we applied this theory with a numerical example in mind. Let us define two vectors, 

$$ \begin{array}{l}
\mathbf{x} =\begin{bmatrix}
1.0 & 3.0 & 6.0 & 2.0 & 0.0
\end{bmatrix} ,\ x\ \epsilon \ \mathbb{R}^{5}\\\\
\mathbf{y} =\begin{bmatrix}
2.0 & 7.0 & 5.0 & 1.0 & 1.0
\end{bmatrix} ,\ y\ \epsilon \ \mathbb{R}^{5}
\end{array}$$

And going further, we can define an equation using them as our primary variable,

$$ \begin{array}{l}
\mathbf{z} \ =\ \mathbf{x^{2} +y^{2}}\\\\\\\\
\mathbf{z} \ =\ \left(\begin{bmatrix}
1.0 & 3.0 & 6.0 & 2.0 & 0.0
\end{bmatrix}\right)^{2} +\left(\begin{bmatrix}
2.0 & 7.0 & 5.0 & 1.0 & 1.0
\end{bmatrix}\right)^{2}\\\\\\\\
\mathbf{z} \ =\ \begin{bmatrix}
5.0 & 58.0 & 61.0 & 5.0 & 1.0
\end{bmatrix} ,\ \ \ z\ \epsilon \ \mathbb{R}^{5}
\end{array}$$

Thus we have the value of **z** vector, with the dimensions being (*1 x 5*). Going further, before we begin with calculating the gradients, let's visualize it in a graph data structure, where we need two more sub-variables: **u** = *x*<sup>*2*</sup> and **v** = *y*<sup>*2*</sup>

$$ \begin{array}{l}
\mathbf{u} \ =\ \begin{bmatrix}
1.0 & 9.0 & 36.0 & 4.0 & 0.0
\end{bmatrix}\\\\
\mathbf{v} \ =\ \begin{bmatrix}
4.0 & 49.0 & 25.0 & 1.0 & 1.0
\end{bmatrix}
\end{array}$$

<img src='/media/pic1-AD2.png'>

Upto here, we are in similiar territory as *scalar* valued functions. But as we shall see, while backpropagating from here, we observe a change in dimensions which needs to be dealt with.
Going back as usual leads us to the following equations:

$$ \begin{array}{l}
\frac{\delta \mathbf{z}}{\delta \mathbf{u}} =1\ ,\ \frac{\delta \mathbf{z}}{\delta \mathbf{v}} =1\\\\\\\\
\frac{\delta \mathbf{u}}{\delta \mathbf{x}} =2\mathbf{x} =\begin{bmatrix}
2.0 & 6.0 & 12.0 & 4.0 & 0.0
\end{bmatrix} ,\\\\\\\\ \frac{\delta \mathbf{v}}{\delta \mathbf{y}} =2\mathbf{y} =\begin{bmatrix}
4.0 & 14.0 & 10.0 & 2.0 & 2.0
\end{bmatrix}\\\\\\\\
\frac{\delta \mathbf{z}}{\delta \mathbf{x}} =\frac{\delta \mathbf{z}}{\delta \mathbf{u}}\frac{\delta \mathbf{u}}{\delta \mathbf{x}} =2\mathbf{x}\\\\\\\\
\frac{\delta \mathbf{z}}{\delta \mathbf{y}} =\frac{\delta \mathbf{z}}{\delta \mathbf{v}}\frac{\delta \mathbf{v}}{\delta \mathbf{y}} =2\mathbf{y}
\end{array}$$

As was in the previous essay, the graph looks like this:

<img src='/media/pic2-AD2.png'>

One thing that we must remember is that the expressions of partial derivatives represent *matrices* and not simple vectors. This can prove somewhat troublesome while implementing practically, as the gradients as passed backwards, they are needed for further computations, which can become complicated quickly as the dimensions increase rapidly, and sort of accumulate. For instance,

$$\frac{\delta \mathbf{z}}{\delta \mathbf{u}} \ =\ \begin{bmatrix}
\frac{\delta z_{1}}{\delta u_{1}} & \frac{\delta z_{2}}{\delta u_{1}} & \frac{\delta z_{3}}{\delta u_{1}} & \frac{\delta z_{4}}{\delta u_{1}} & \frac{\delta z_{5}}{\delta u_{1}}\\\\\\\\
\frac{\delta z_{1}}{\delta u_{2}} &  &  &  & \\\\\\\\
\frac{\delta z_{1}}{\delta u_{3}} &  & \ddots  &  & \vdots \\\\\\\\
\frac{\delta z_{1}}{\delta u_{4}} &  &  &  & \\\\\\\\
\frac{\delta z_{1}}{\delta u_{5}} &  & \dotsc  &  & \frac{\delta z_{5}}{\delta u_{5}}
\end{bmatrix}_{5*5}$$

Is already a matrix **R**<sup>*5 * 5*</sup>, and will be used to calculate the gradients as we go further up the graph, and be multiplied by other matrices. Apart from that, if we had started **x** or **y** to be matrices, it would have resulted in the same difficulty. Due to this reasons, we utilize a **gradient tensor** at the beginning of the algorithm. It it nothing but a vector passed onwards from the starting point, following the same process as before. 

<img src='/media/pic3-AD2.png'>

Passing along this vector won't change our answers at all, for example, our Jacobian was *J*<sub>*x*</sub> (or *J*<sub>*y*</sub>), and with passing along this vector, we simply end up with *[1, 1, 1, 1, 1]*<sup>*T*</sup>. *J*<sub>*x*</sub> , which will give us the original vector (the *gradient* of *x* or *y*), as doing the above mentioned operations reduces the matrix to a vector, which is what we desired in the first place (the dimension are (*1  * 5*) x (*5 * 5*) = *1 * 5*), giving us a vector. The main reason for passing in a vector in the last node was to reduce our output from a matrix to a vector, or whatever the dimension was of our input. Note that the gradient tensor has to match the dimensions of the function value (**z**). 

Another added advantage is that we get control over the *sensitivity* of the output with respect to the input. If instead we passed in a gradient tensor *[1, 1, 2, 1, 1]*, we will get the same output, except the third element will be twice of the original answer, as was intended. Overall, passing in a gradient tensor proves quite useful when calculating gradients of vector-valued functions. With this, we are done with the theory for Automatic Differentiation !

<p style="text-align: center;">
<strong><a href='/'>Home</a></strong>
</p>