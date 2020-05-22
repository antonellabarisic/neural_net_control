# Korištenje

Postaviti u neki paket (npr. mmuav_gazebo). U files direktoriju se nalaze neki fileovi koji se trebaju dodati u mmuav\_gazebo/mmuav\_description/urdf. Nalazi se i launch file koji treba dodati u mmuav\_gazebo/mmuav\_gazebo/launch

seg\_control je 1. algoritam, seg\_control3 je 2. algoritam. Pokrenuti izvođenje kao npr.

```
roslaunch mmuav_gazebo uav_attitude_position_world_test.launch x:=250.0 y:=-114.0 z:=16.0
rosrun neural_net_control <dir>/fcn8_mnv2.xml
```
Ako se želi vidjeti pogled s kamere pokrenuti
```
rosrun image_view image_view image:=/uav/camera1/image_raw
```

Ako se želi mjenjati početna pozicija (x, y, z), treba se u config/seg_ctrl1.yaml promjeniti poziciju. Prilikom pokretanja launch datoteke zadaje se također ta postavljena pozicija. U yaml datoteci se mogu i mijenjati drugi parametri za algoritme.

Ako se želi mjenjati nagib kamere treba se u uav.gazebo.xacro datoteci (linija 81) promjeniti broj (označeno je).
