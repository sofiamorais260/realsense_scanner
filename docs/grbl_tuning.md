# GRBL Tuning Notes

This project uses GRBL joystick jogging through many short `$J` commands rather
than long continuous moves. Because of that, machine responsiveness depends
strongly on the acceleration values stored on the controller itself.

## Why these settings were changed

During joystick testing, the machine felt sluggish, as if motion was being held
back, especially on the Z axis. The limiting factor was not the maximum feed
rate. Instead, the controller acceleration was too low, so each short jog spent
too much of its travel ramping up instead of moving at the intended speed.

For that reason, the GRBL acceleration settings were increased in UGS.

## Relevant parameters

- `$110`, `$111`, `$112`: maximum feed rate for X, Y, and Z
- `$120`, `$121`, `$122`: acceleration for X, Y, and Z
- `$100`, `$101`, `$102`: steps per millimetre for X, Y, and Z

## Values used during joystick tuning

Previous acceleration values:

```text
$120=10
$121=10
$122=10
```

Updated acceleration values:

```text
$120=80
$121=80
$122=30
```

The maximum rate settings used at the time were:

```text
$110=3000
$111=3000
$112=1000
```

## Practical interpretation

- X and Y acceleration were increased from `10` to `80 mm/s^2` to make planar
  joystick motion more responsive.
- Z acceleration was increased more conservatively from `10` to `30 mm/s^2`
  because the vertical axis is mechanically more sensitive and should not be
  made as aggressive without testing.

## Important note

These values are stored in the GRBL controller, not in the Python application.
They persist after disconnecting the machine, closing UGS, or power-cycling the
system. They must be changed again only if the controller configuration is reset
or overwritten.
