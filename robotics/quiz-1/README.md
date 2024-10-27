# CSE298 - Foundations of Robotics - Quiz 1

**Due: 7/14/2021 by EOD**

Make at least one commit per question. You can use any resources you like. You can write your answers in this file, `README.md` file. You can use the Gitlab interface to edit this file if you prefer.

## Question 1

Describe the method by which LIDAR measures the distance to objects.

- LiDAR (Light Detection and Ranging) measures distances using time of flight.
- It sends out a beam of light then uses the time it takes for the light to reflect back to calculate the distance.
- The distance equation would be d=ct where:
    * d is distance
    * c is the speed of light
    * t is time of flight

## Question 2

What do accelerometers measure? What do gyroscopes measure?

- Accelerometers measure non-gravitational acceleration in XYZ or NED.
- Gyroscopes use the Earth's gravity to determine orientation by measuring the rate of rotation around axes.

## Question 5

Given the following data, find the least squares best fit for a line that passes through it. Provide values for Beta1 and Beta2.

```Matlab
data = [
    0.8345   39.6147
    2.7439   39.7819
    2.9933   35.9441
    5.3286   33.1060
    4.7773   31.5177
    4.6543   30.9445
    7.2957   28.2147
    9.8732   26.9777
    7.5641   24.0761
    9.7092   24.7355
   12.2058   18.1678
    9.6958   17.5425
   13.7505   15.9074
   14.9036   14.0982
   15.2562   13.6018
   15.0757   10.2414
   16.9914    7.8736
   18.1118    6.0755
   20.1223    4.8266
   20.7962    1.2416]
```

* Beta1 = -1.9308
* Beta2 = 41.7573
* y = Beta2(x) + Beta1