#include <Arduino_RouterBridge.h>

// TB6612FNG driver pins (tested pin mapping, confirmed working)
#define PWMA 5
#define AIN1 8
#define AIN2 9

#define PWMB 6
#define BIN1 10
#define BIN2 11

#define STBY 7

const int SPEED = 180;

// Motion modes, set from Python via Bridge.call("set_motion", mode)
enum Motion {
  MOTION_STOP = 0,
  MOTION_FORWARD = 1,
  MOTION_PIVOT_RIGHT = 2,
  MOTION_PIVOT_LEFT = 3,
};

void stop_motors() {
  analogWrite(PWMA, 0);
  analogWrite(PWMB, 0);
}

void drive_forward() {
  digitalWrite(AIN1, HIGH);
  digitalWrite(AIN2, LOW);
  digitalWrite(BIN1, HIGH);
  digitalWrite(BIN2, LOW);
  analogWrite(PWMA, SPEED);
  analogWrite(PWMB, SPEED);
}

// Pivot in place by driving only one side (matches the "right"/"left" test
// pattern the user already validated: one motor on, the other idle).
void pivot_right() {
  digitalWrite(AIN1, HIGH);
  digitalWrite(AIN2, LOW);
  analogWrite(PWMA, SPEED);
  analogWrite(PWMB, 0);
}

void pivot_left() {
  digitalWrite(BIN1, HIGH);
  digitalWrite(BIN2, LOW);
  analogWrite(PWMB, SPEED);
  analogWrite(PWMA, 0);
}

// Single RPC entry point: Python is the state machine, the sketch just
// executes whatever motion mode it's told, right away.
void set_motion(int mode) {
  switch (mode) {
    case MOTION_FORWARD:
      drive_forward();
      break;
    case MOTION_PIVOT_RIGHT:
      pivot_right();
      break;
    case MOTION_PIVOT_LEFT:
      pivot_left();
      break;
    default:
      stop_motors();
      break;
  }
}

void setup() {
  pinMode(PWMA, OUTPUT);
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);

  pinMode(PWMB, OUTPUT);
  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);

  pinMode(STBY, OUTPUT);
  digitalWrite(STBY, HIGH);

  stop_motors();  // fail-safe: stay stopped until Python commands motion

  Bridge.begin();
  Bridge.provide("set_motion", set_motion);
}

void loop() {
  // Everything is event-driven through Bridge RPC.
}
