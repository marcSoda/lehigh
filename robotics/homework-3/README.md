# CSE298 - Foundations of Robotics - Homework 3

**Due Date: 7/26/2021 EOD**

## Objectives:

The purpose of this assignment is to derive a motion model for an idealized robot. You can find a solution to this if you search online, but try to do it yourself first before looking for the answer.

## The Assignment

For this assignment you will derive the relevant matrices needed to compute a prediction and measurement update for a robot moving on a 2D plane. On this plane, the robot has an `x` coordinate, a `y` coordinate, and a `theta` orientation. It moves with linear velocity `v`, and angular velocity `omega`.

The state vector `s_t` for this robot is

```
s_t = [x y theta]'
```

The input control `u_t` for the robot is

```
u_t = [v omega]'
```

The robot can make an observation `z_t` of a landmark using a LIDAR sensor. The sensor reports the range of the landmark `rho` and the bearing to the landmark `phi`.

```
z_t = [rho phi]'
```

You will need some trig for this assignment, so if you need to brush up on your sines and cosines, do so now. Remember: SOHCAHTOA.

## Model

### Question 1

Write down the range of the robot `rho` in terms of the landmark's coordinates `(m_x, m_y)`, and the robot's coordiantes `(x,y)`.

```
`rho` = sqrt((x_m-x)^2(y_m-y)^2)
```

### Question 2

Write down the bearing of the robot `phi` in terms of the landmark's coordinates `(m_x, m_y)`, and the robot's coordiantes `(x,y)`.

```
`phi` = atan2((y_m - y), (x_m - x)) - theta
```

### Question 3

With the answers to 1 and 2, write down the function `h(s_t)`.

```
h(s_t) = [rho phi]
       = [(sqrt((x_m-x)^2(y_m-y)^2)) (atan2((y_m - y), (x_m - x)))]
```

### Question 4

The robot can be thought of moving along circumference of a circle with radius

```
r = v/omega
```

The velocity vector of the robot can be broken down into x and y components. What are `vx` and `vy`, given `theta`?

```
vx = vsin(theta)
vy = vcos(theta)
```

--

Let's say that `v` and `omega` remain constant over a small time period `dt`. If the robot starts at `x` and `y` with orientation `theta`, and it's traveling  at `v` and `omega`, where will the robot be after `dt`?

```
w = omega
(x + -(v/w)sin(theta),y + (v/w)cos(theta))
```

### Question 5

- x update
```
x_t = x_(t-1) + (-(v/w)sin(theta) + (v/w)sin(theta+wdt))
```
- y update
```
y_t = y_(t-1) + ((v/w)cos(theta) + (v/w)cos(theta+wdt))
```
- theta update
```
theta_t = theta_(t-1) + (wdt)
```
- note: dt is the timestep

### Question 6

With the answers to 5, write down the function g(s_t,u_t).

```
g(s_t,u_t) = [x_t y_t theta_t]
           = [(x_(t-1) + (-(v/w)sin(theta) + (v/w)sin(theta+wdt))) (y_(t-1) + ((v/w)cos(theta) + (v/w)cos(theta+wdt))) (theta_(t-1) + wdt)]

```
## Linearization

The model you made in the previous part is nonlinear. Those pesky sines and cosines ruin it for us. In order to use an EKF, we need to linearize the model by calculating the [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant).

## Question 9

Calculate the following:

```
dg(1)/dx = 1
dg(2)/dx = 0
dg(3)/dx = 0
dg(1)/dy = 0
dg(2)/dy = 1
dg(3)/dy = 0
dg(1)/dtheta = (v(cos(wdt + theta) - cos(theta))) / w
dg(2)/dtheta = (v(sin(theta + wdt) + sin(theta))) / w
dg(3)/dtheta = 1
```

## Question 10

```
Write down the Jacobian matrix G using the answers to question 9
G = [(dg(1)/dx) (dg(1)/dy) (dg(1)/dtheta) ; (dg(2)/dx) (dg2)/dy) (dg(2)/dtheta) ; (dg(3)/dx) (dg(3)/dy) (dg(3)/dtheta)]
  = [(1) (0) ((v(cos(wdt + theta) - cos(theta))) / w) ; (0) (1) ((v(sin(theta + wdt) + sin(theta))) / w) ; (0) (0) (1)]
```

## Question 11

Calculate the following:

```
dh(1)/dx = y - y_m
dh(2)/dx = (y_m - y) / ((x_m - x)^2 + (y_m - y)^2)
dh(1)/dy = x - x_m
dh(2)/dy = -(x_m - x) / ((y - y_m)^2 + (x_m - x)^2)
dh(1)/dtheta = 0
dh(2)/dtheta = - 1
```

## Question 12

Write down the Jacobian matrix H using the answers to question 11

```
H = [(dh(1)/dx) (dh(1)/dy) (dh(1)/dtheta) ; (dh(2)/dx) (dh(2)/dy) (dh(2)/dtheta)]
  = [(y - y_m) (x - x_m) (0) ; ((y_m - y) / ((x_m - x)^2 + (y_m - y)^2)) (-(x_m - x) / ((y - y_m)^2 + (x_m - x)^2)) (0)]
```
