#include <Arduino_RouterBridge.h>

// TB6612FNG driver pins (tested pin mapping, confirmed working)
#define PWMA 5
#define AIN1 8
#define AIN2 9

#define PWMB 6
#define BIN1 10
#define BIN2 11

#define STBY 7

// HC-SR04 ultrasonic distance sensor.
// Powered from the 3.3V pin (not 5V) - the UNO Q's GPIO is 3.3V logic and
// isn't confirmed 5V-tolerant, so Echo must never see more than ~3.3V.
// Confirmed by testing: Trig -> pin 3, Echo -> pin 2 on this sensor/wiring.
#define TRIG_PIN 3
#define ECHO_PIN 2

// Lowered from 180 -> 130 -> 50: at higher speed the camera frames were
// blurry/shaky enough that a genuinely broken wire would flicker to INTACT
// on individual frames, and the car was generally moving faster than makes
// sense for a wire-by-wire inspection pass anyway.
const int SPEED = 50;

// The car drifted right when driving straight at equal PWM on both sides -
// pivot_right() (motor A alone) turns the car right, so A is the left
// wheel; trimming it down brings the two sides back into balance. If it
// now drifts left instead, flip this to trim PWMB down by the same amount
// instead; retune the magnitude by feel.
const int TRIM_A = 8;

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
  analogWrite(PWMA, SPEED - TRIM_A);
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

// Returns distance in cm; 2.0 if something is closer than reliably
// measurable; -1.0 if no echo came back at all (out of range / stuck pin).
//
// Hand-rolled instead of using pulseIn(): observed pulseIn() on this
// Zephyr/STM32 core occasionally blocking for several seconds instead of
// respecting its timeout argument when Echo doesn't toggle as expected,
// which stalled the Bridge RPC call well past its own timeout. Polling
// micros() directly with an explicit deadline on both wait phases guarantees
// this function can never block longer than ~2*TIMEOUT_US regardless.
float get_distance_cm() {
  // HC-SR04 itself asserts Echo for up to ~38ms when nothing is in range
  // before giving up, so our own wait has to comfortably clear that.
  const unsigned long TIMEOUT_US = 45000UL;
  // The sensor can't reliably time anything closer than ~2cm (~116us round
  // trip) - a shorter pulse means either electrical noise on the Echo line
  // (e.g. coupling right at the Trig edge) or a real object sitting inside
  // that blind zone. We can't tell which from the pulse alone, so we report
  // it as "very close" (MIN_VALID_CM) rather than -1.0 ("nothing there") -
  // treating it as "clear" let the car drive straight into anything closer
  // than 2cm, which is the opposite of what an obstacle sensor should do.
  // Bogus single-tick noise is instead filtered by the caller requiring
  // consecutive close readings before reacting.
  const unsigned long MIN_VALID_US = 100UL;
  const float MIN_VALID_CM = 2.0;

  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(15);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long wait_start = micros();
  while (digitalRead(ECHO_PIN) == LOW) {
    if (micros() - wait_start > TIMEOUT_US) return -1.0;
  }

  unsigned long pulse_start = micros();
  while (digitalRead(ECHO_PIN) == HIGH) {
    if (micros() - pulse_start > TIMEOUT_US) return -1.0;
  }
  unsigned long pulse_end = micros();
  unsigned long duration_us = pulse_end - pulse_start;

  if (duration_us < MIN_VALID_US) return MIN_VALID_CM;
  return duration_us / 58.0;  // standard HC-SR04 conversion: round-trip us / 58 = cm
}

// Diagnostic only: raw idle Echo pin state, no trigger pulse sent. A
// properly powered, connected HC-SR04 should read LOW consistently at idle.
// If this flickers between 0/1 with nothing triggering it, the pin is
// floating (sensor not actually powered / GND or VCC not really connected).
bool read_echo_raw() {
  return digitalRead(ECHO_PIN) == HIGH;
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

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);

  stop_motors();  // fail-safe: stay stopped until Python commands motion

  Bridge.begin();
  Bridge.provide("set_motion", set_motion);
  Bridge.provide("get_distance_cm", get_distance_cm);
  Bridge.provide("read_echo_raw", read_echo_raw);
}

void loop() {
  // Everything is event-driven through Bridge RPC.
}
