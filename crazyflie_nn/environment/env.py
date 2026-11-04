import jax
import jax.numpy as jnp
from brax import envs
from brax.envs import State
import dynamics.dynamics as dyn # Drone dynamics

class DroneBrax(envs.Env):
    '''Environment for training a position controller for a drone random start return to (0, 0, 0)'''

    def __init__(self, backend='jax', dt=0.01, action_hist_len=1):
        self.dt = dt  
        self.substeps = 1 

        self.act_dim = 4 
        self.state_dim = 18 # (pos[3], Rwb_flat[9], vel[3], angvel[3])
        
        # Action history setup
        self.action_hist_len = action_hist_len
        self.obs_dim = self.state_dim + (self.act_dim * self.action_hist_len)

        self.hover_rpm = dyn.HOVER_RPM 
        self.max_rpm = dyn.MAX_THRUST

        self.min_alive = 0 * self.hover_rpm 
        self.thrust_scale = 0.4 * self.hover_rpm 

        # Reward parameters
        self.reward_parameters = {
            'position_weight': 2.5 ,
            'orientation_weight': 1.5,  
            'velocity_weight' :0.1,
            'action_weight': 0.05, 
            'smoothing_weight': 0.005,
            'survival': 0
        }

        # Noise will be added to the observation
        state_noise_std = jnp.array([
            0.005, 0.005, 0.005, # x, y, z
            0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, # Rwb_flat
            0.005, 0.005, 0.005, # vx, vy, vz
            0.005, 0.005, 0.005, # wx, wy, wz
        ])
        
        # Zero noise for the action history part of the observation
        if self.action_hist_len > 0:
            action_noise_std = jnp.zeros(self.act_dim * self.action_hist_len)
            self.noise_std = jnp.concatenate([state_noise_std, action_noise_std])
        else:
            self.noise_std = state_noise_std

    @property
    def observation_size(self):
        return self.obs_dim
    
    @property
    def action_size(self):
        return self.act_dim
    
    @property
    def backend(self):
        return 'jax'
    
    def reset(self, rng: jax.Array) -> State:
        rng, key_pos, key_att, key_vel, key_ang, key_step, key_dr, key_next, key_stay_still = jax.random.split(rng, 9)
        
        def get_random_Rwb(k):
            a = jax.random.uniform(k, (3,), minval=-jnp.pi/4, maxval=jnp.pi/4)
            Rx = jnp.array([[1, 0, 0], [0, jnp.cos(a[0]), -jnp.sin(a[0])], [0, jnp.sin(a[0]), jnp.cos(a[0])]])
            Ry = jnp.array([[jnp.cos(a[1]), 0, jnp.sin(a[1])], [0, 1, 0], [-jnp.sin(a[1]), 0, jnp.cos(a[1])]])
            Rz = jnp.array([[jnp.cos(a[2]), -jnp.sin(a[2]), 0], [jnp.sin(a[2]), jnp.cos(a[2]), 0], [0, 0, 1]])
            return (Rz @ Ry @ Rx).flatten()

        pos, Rwb_flat, vel, ang = jax.lax.cond(
            jax.random.uniform(key_stay_still) <= 0.05,
            lambda _: (jnp.zeros(3),
                        jnp.eye(3).flatten(),
                        jax.random.uniform(key_vel, (3,), minval=-1, maxval=1),
                        jax.random.uniform(key_ang, (3,), minval=-1, maxval=1)),
            lambda _: (
                jax.random.uniform(key_pos, (3,), minval=-0.2, maxval=0.2),
                get_random_Rwb(key_att),
                jax.random.uniform(key_vel, (3,), minval=-1, maxval=1),
                jax.random.uniform(key_ang, (3,), minval=-1, maxval=1)
            ),
        operand=None
        )
        state = jnp.concatenate([pos, Rwb_flat, vel, ang]) 
        
        # Add actual motor speeds (omega) to the pipeline_state for the integrator
        omega_init = jnp.ones(4) * self.hover_rpm
        pipeline_state = jnp.concatenate([state, omega_init])

        # Initialize action history with zeros
        action_history = jnp.zeros(self.act_dim * self.action_hist_len)

        obs = self._get_obs(pipeline_state, action_history)
        obs_noise = jax.random.normal(key_step, shape=obs.shape)
        obs = obs + obs_noise * self.noise_std        

        metrics = {
            'pos_x': jnp.float32(0.0),
            'pos_z': jnp.float32(0.0),
            'reward_pos': jnp.float32(0.0),
            'reward_survive': jnp.float32(0.0)
        }
        
        info = {
            'action': jnp.zeros(4), 
            'action_history': action_history, # Store history in info
            'rng': key_next, 
            'key_dr': key_dr
        }
        
        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            metrics=metrics,
            info=info
        )
    
    def step(self, state: State, action: jax.Array) -> State:
        current_state = state.pipeline_state 
        
        # Map current actions to rpm
        u = jnp.clip(self.hover_rpm + action * self.thrust_scale, self.min_alive, self.max_rpm) 
        
        # Retrieve previous states from info
        action_prev = state.info['action']
        action_history = state.info['action_history']
        
        rng = state.info['rng']
        rng, key_noise, key_next = jax.random.split(rng, 3)
        domain_random_key = state.info['key_dr'] # This changes only on reset, each episode has the same parameters

        def body_fun(i, x):
            return dyn.rk4(x, u, domain_random_key, self.dt / self.substeps)

        next_state = jax.lax.fori_loop(0, self.substeps, body_fun, current_state) 
        next_pipeline_state = jnp.concatenate([next_state])

        pos = next_state[0:3]
        Rwb_flat = next_state[3:12]
        vel = next_state[12:15]

        # Extract Euler Angles for limits
        Rwb = jnp.reshape(Rwb_flat, (3, 3))
        sy = jnp.sqrt(Rwb[0, 0]**2 + Rwb[1, 0]**2)
        singular = sy < 1e-6
        roll = jnp.where(singular, jnp.arctan2(-Rwb[1, 2], Rwb[1, 1]), jnp.arctan2(Rwb[2, 1], Rwb[2, 2]))
        pitch = jnp.where(singular, jnp.arctan2(-Rwb[2, 0], sy), jnp.arctan2(-Rwb[2, 0], sy))
        
        target_p = jnp.zeros(3)
        dist = jnp.linalg.norm(target_p - pos)
        vel_error = jnp.square(jnp.linalg.norm(vel))
        act_mag = jnp.linalg.norm(action)
        r11 = Rwb_flat[0]
        r21 = Rwb_flat[3]

        orient_error = jnp.linalg.norm(jnp.eye(3) - Rwb) 

        # Reward components
        r_pos    = -self.reward_parameters['position_weight'] * jnp.square(dist)
        r_act    = -self.reward_parameters['action_weight']   * jnp.square(act_mag)
        r_vel    = -self.reward_parameters['velocity_weight'] *(vel_error)
        r_orient = -self.reward_parameters['orientation_weight'] * jnp.square((orient_error))
        r_smooth = -self.reward_parameters['smoothing_weight'] * jnp.square(jnp.linalg.norm(action - action_prev))
        
        reward = r_pos  + r_vel+ r_act + r_orient + r_smooth + 2.0

        # Termination conditions
        angle_limit = jnp.deg2rad(60.0) 
        crashed = pos[2] < -0.6
        too_high = pos[2] > 0.6
        too_far = (jnp.abs(pos[0]) > 0.6) | (jnp.abs(pos[1]) > 0.6)
        too_tilt = (jnp.abs(roll) > angle_limit) | (jnp.abs(pitch) > angle_limit)
        done = (crashed | too_high | too_far | too_tilt).astype(jnp.float32)
        reward = jnp.where(done > 0.5, -10.0, reward) 

        # Update action history
        if self.action_hist_len > 0:
            new_action_history = jnp.concatenate([action_history[self.act_dim:], action])
        else:
            new_action_history = action_history

        # Construct new observation
        obs = self._get_obs(next_pipeline_state, new_action_history)
        raw_noise = jax.random.normal(key_noise, shape=obs.shape)
        noisy_obs = obs + (raw_noise * self.noise_std)

        new_metrics = {'pos_x': pos[0], 'pos_z': pos[2], 'reward_pos': r_pos}
        metrics = {**state.metrics, **new_metrics}
        
        info = {
            **state.info, 
            'action': action, 
            'action_history': new_action_history, # Update info for the next step
            'rng': key_next
        }
        
        return state.replace(
            pipeline_state=next_pipeline_state,
            obs=noisy_obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info
        )
    
    def _get_obs(self, pipeline_state, action_history):
        """Constructs observation from the full pipeline state and action history."""

        state = pipeline_state[:self.state_dim]
        if self.action_hist_len > 0:
            return jnp.concatenate([state, action_history])
        return state
