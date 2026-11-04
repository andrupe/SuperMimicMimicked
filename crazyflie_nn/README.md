# Crazyflie RL Controller

## Training 
Run the `trainPPO.py`

When it finishes the neural network in c headerfile format will be available at `logs/params/matrices.h`

## Flashing the Firmware
We now need to add the controller to the crazyflie.

First run `git clone --recursive https://github.com/bitcraze/crazyflie-firmware.git` to clone the official firmware.

### Creating a New App
Start by copying an existing app, 

`cp -r examples/app_hello_world examples/my_controller`

`cd examples/my_controller/`

Then rename it,

`mv src/hello_world.c src/my_controller.c`

We have to tell kbuild that we renamed the file. Open `src/Kbuild` in your and update it to
`obj-y += my_controller.o`

Next, replace the code of the `my_controller.c` file with the code from `firmware/rl_controller.c`, then copy the `prototypes.h` and the `matrices.h` (Neural network weights) and paste them into the same folder as the `my_controller.c`

Finally we need to tell the firmware that we have implemented the out-of-tree controller and that it should be added to the list. We do this by adding CONFIG_CONTROLLER_OOT to the app-config file. When you are done it should look like this:


`CONFIG_APP_ENABLE=y`


`CONFIG_APP_PRIORITY=1`

`CONFIG_APP_STACKSIZE=350`

`CONFIG_CONTROLLER_OOT=y`

As a last step we have to change the high level commander rate from 100Hz to 250Hz This is done by going to the `crazyflie-firmware/src/modules/interface/stabilizer_types.h`

and chainging the,

`#define RATE_HL_COMMANDER RATE_100_HZ`

to

`#define RATE_HL_COMMANDER RATE_250_HZ`


Finally compile,

`make -j$(nproc)`

and flash it to the drone ,

`make cload`

*Note: The instuctions above are based on the official documentation [here](https://www.bitcraze.io/2023/02/adding-an-estimator-or-controller/)*
