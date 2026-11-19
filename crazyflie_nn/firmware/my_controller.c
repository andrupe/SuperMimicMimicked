/* Out-Of-Tree SuperMimic NN Controller for Bitcraze Crazyflie 2.1 (STM32F405) */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "app.h"
#include "FreeRTOS.h"
#include "task.h"
#define DEBUG_MODULE "RLCONTROLLER"
#include "debug.h"
#include "stabilizer_types.h"

#include "supermimic_policy.h"

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

#define SM_POS_ERR_CLIP 1.5f
#define SM_ARM_LEN_OVER_SQRT2 (0.046f / 1.41421356f)
#define SM_KM_OVER_KF (1.34e-10f / 2.25e-8f)
#define SM_HIST_STRIDE 2
#define SM_RING_LEN ((SM_HIST_LEN - 1) * SM_HIST_STRIDE + 1)

static float obs_ring[SM_RING_LEN][SM_FRAME_DIM];
static float obs_vec[SM_OBS_DIM];
static float last_raw_action[SM_ACT_DIM] = {0.0f, 0.0f, 0.0f, 0.0f};
static bool hist_initialized = false;

static float body_torque_x = 0.0f;
static float body_torque_y = 0.0f;
static float body_torque_z = 0.0f;
static float total_thrust  = 0.0f;

static inline float clampf(float v, float lo, float hi) {
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

/* Rotate world vector v_w into body frame using unit quaternion q=(w,x,y,z): R^T * v_w */
static inline void world_to_body(const quaternion_t *q, float wx, float wy, float wz, float out[3]) {
  float w = q->w, x = q->x, y = q->y, z = q->z;
  const float norm = sqrtf(w * w + x * x + y * y + z * z);
  if (norm > 1e-6f) {
    const float inv = 1.0f / norm;
    w *= inv; x *= inv; y *= inv; z *= inv;
  } else {
    w = 1.0f; x = 0.0f; y = 0.0f; z = 0.0f;
  }
  const float r00 = 1.0f - 2.0f * (y * y + z * z);
  const float r01 = 2.0f * (x * y - w * z);
  const float r02 = 2.0f * (x * z + w * y);
  const float r10 = 2.0f * (x * y + w * z);
  const float r11 = 1.0f - 2.0f * (x * x + z * z);
  const float r12 = 2.0f * (y * z - w * x);
  const float r20 = 2.0f * (x * z - w * y);
  const float r21 = 2.0f * (y * z + w * x);
  const float r22 = 1.0f - 2.0f * (x * x + y * y);
  out[0] = r00 * wx + r10 * wy + r20 * wz;
  out[1] = r01 * wx + r11 * wy + r21 * wz;
  out[2] = r02 * wx + r12 * wy + r22 * wz;
}

void appMain(void) {
  DEBUG_PRINT("SuperMimic OOT RL controller ready (100 Hz, stride=2).\n");
  while (1) {
    vTaskDelay(M2T(2000));
  }
}

void controllerOutOfTreeInit(void) {
  memset(obs_ring, 0, sizeof(obs_ring));
  memset(obs_vec, 0, sizeof(obs_vec));
  memset(last_raw_action, 0, sizeof(last_raw_action));
  hist_initialized = false;
  total_thrust = 0.0f;
  body_torque_x = 0.0f;
  body_torque_y = 0.0f;
  body_torque_z = 0.0f;
  DEBUG_PRINT("SuperMimic OOT controller initialized!\n");
}

bool controllerOutOfTreeTest(void) {
  return true;
}

void controllerOutOfTree(control_t *control,
                         const setpoint_t *setpoint,
                         const sensorData_t *sensors,
                         const state_t *state,
                         const uint32_t tick) {
  if (setpoint->mode.z == modeDisable) {
    hist_initialized = false;
    memset(last_raw_action, 0, sizeof(last_raw_action));
    control->thrustSi = 0.0f;
    control->torqueX  = 0.0f;
    control->torqueY  = 0.0f;
    control->torqueZ  = 0.0f;
    control->controlMode = controlModeForceTorque;
    return;
  }

  if (RATE_DO_EXECUTE(RATE_100_HZ, tick)) {
    float frame[SM_FRAME_DIM];
    float g_b[3], v_b[3], pos_err_b[3], ref_vel_b[3];
    const quaternion_t *q = &state->attitudeQuaternion;

    world_to_body(q, 0.0f, 0.0f, -1.0f, g_b);
    world_to_body(q, state->velocity.x, state->velocity.y, state->velocity.z, v_b);
    world_to_body(q,
                  setpoint->position.x - state->position.x,
                  setpoint->position.y - state->position.y,
                  setpoint->position.z - state->position.z,
                  pos_err_b);
    world_to_body(q, setpoint->velocity.x, setpoint->velocity.y, setpoint->velocity.z, ref_vel_b);

    frame[0] = state->position.x;
    frame[1] = state->position.y;
    frame[2] = state->position.z;
    frame[3] = g_b[0];
    frame[4] = g_b[1];
    frame[5] = g_b[2];
    frame[6] = v_b[0];
    frame[7] = v_b[1];
    frame[8] = v_b[2];
    frame[9]  = sensors->gyro.x * (M_PIf / 180.0f);
    frame[10] = sensors->gyro.y * (M_PIf / 180.0f);
    frame[11] = sensors->gyro.z * (M_PIf / 180.0f);
    frame[12] = last_raw_action[0];
    frame[13] = last_raw_action[1];
    frame[14] = last_raw_action[2];
    frame[15] = last_raw_action[3];
    frame[16] = clampf(pos_err_b[0], -SM_POS_ERR_CLIP, SM_POS_ERR_CLIP);
    frame[17] = clampf(pos_err_b[1], -SM_POS_ERR_CLIP, SM_POS_ERR_CLIP);
    frame[18] = clampf(pos_err_b[2], -SM_POS_ERR_CLIP, SM_POS_ERR_CLIP);
    frame[19] = ref_vel_b[0];
    frame[20] = ref_vel_b[1];
    frame[21] = ref_vel_b[2];
    frame[22] = setpoint->attitudeRate.yaw * (M_PIf / 180.0f);

    if (!hist_initialized) {
      for (int r = 0; r < SM_RING_LEN; ++r) {
        memcpy(obs_ring[r], frame, sizeof(frame));
      }
      hist_initialized = true;
    } else {
      memmove(obs_ring[0], obs_ring[1], (SM_RING_LEN - 1) * SM_FRAME_DIM * sizeof(float));
      memcpy(obs_ring[SM_RING_LEN - 1], frame, sizeof(frame));
    }

    for (int h = 0; h < SM_HIST_LEN; ++h) {
      memcpy(&obs_vec[h * SM_FRAME_DIM], obs_ring[h * SM_HIST_STRIDE], SM_FRAME_DIM * sizeof(float));
    }

    float thrust_n[SM_ACT_DIM];
    sm_policy_forward_with_raw(obs_vec, thrust_n, last_raw_action);

    /* Rotor ordering from crazyflie.xml: f0=FL (+d,+d), f1=FR (+d,-d), f2=RL (-d,+d), f3=RR (-d,-d) */
    const float f0 = thrust_n[0];
    const float f1 = thrust_n[1];
    const float f2 = thrust_n[2];
    const float f3 = thrust_n[3];
    const float t_tot = f0 + f1 + f2 + f3;
    const float tx    = SM_ARM_LEN_OVER_SQRT2 * ( f0 - f1 + f2 - f3);
    const float ty    = SM_ARM_LEN_OVER_SQRT2 * (-f0 - f1 + f2 + f3);
    const float tz    = SM_KM_OVER_KF         * (-f0 + f1 - f2 + f3);
    if (isfinite(t_tot) && isfinite(tx) && isfinite(ty) && isfinite(tz)) {
      total_thrust  = clampf(t_tot, 0.0f, 0.6168f);
      body_torque_x = tx;
      body_torque_y = ty;
      body_torque_z = tz;
    } else {
      total_thrust = 0.0f;
      body_torque_x = 0.0f;
      body_torque_y = 0.0f;
      body_torque_z = 0.0f;
    }
  }

  control->thrustSi = total_thrust;
  control->torqueX  = body_torque_x;
  control->torqueY  = body_torque_y;
  control->torqueZ  = body_torque_z;
  control->controlMode = controlModeForceTorque;
}
