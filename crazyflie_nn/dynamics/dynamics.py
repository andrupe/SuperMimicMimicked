import jax
import jax.numpy as jnp

###########################
# Constants
###########################
DT = 0.005
L = 0.046
KF = 2.25e-8 # N/(rad / s)^2
KM = 1.34e-10 # N*m/ (rad/s)^2
G = 9.81
M_NOMINAL = 0.033
HOVER_RPM = jnp.sqrt(G * M_NOMINAL / (4 * KF)) 
MAX_THRUST = 2618
print(f"Hover RPM is {HOVER_RPM:.0f}, which is {HOVER_RPM * 100 / MAX_THRUST:.0f}% of max RPM")
DRAG_COEFF_XY = 9.1785e-7
DRAG_COEFF_Z = 10.311e-7
CD = jnp.diag(jnp.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z]))
I_NOMINAL = jnp.array([[16.6e-6, 0.83e-6, 0.72e-6],
                [0.83e-6, 16.6e-6, 1.8e-6],
                [0.72e-6, 1.8e-6, 29.3e-6]])
Kaero = jnp.array([[9.1785e-7, 0.,        0.],
                   [0.        ,9.1785e-7, 0],
                   [0.        ,0.        ,10.311e-7]])

KF_BOUND = 0.1 * KF
KM_BOUND = 0.1 * KM
TAU_M = 0.1

#############################
# Dynamics
#############################
@jax.jit
def dynamics(x, u, key):
    m_key, i_key, KF1_key, KF2_key,KF3_key,KF4_key, KM_key = jax.random.split(key,7)

    KF1 = KF + jax.random.uniform(KF1_key, minval= -KF_BOUND, maxval= KF_BOUND)
    KF2 = KF + jax.random.uniform(KF2_key, minval= -KF_BOUND, maxval= KF_BOUND)
    KF3 = KF + jax.random.uniform(KF3_key, minval= -KF_BOUND, maxval= KF_BOUND)
    KF4 = KF + jax.random.uniform(KF4_key, minval= -KF_BOUND, maxval= KF_BOUND)
    
    KM_R = KM + jax.random.uniform(KM_key, minval=-KM_BOUND, maxval=KM_BOUND)
    M = M_NOMINAL + jax.random.uniform(m_key, minval=-0.1*M_NOMINAL, maxval=0.1*M_NOMINAL)

    diag_noise = jax.random.uniform(i_key, shape=(3,), minval=0.9, maxval=1.1)
    I_diag = jnp.diag(I_NOMINAL) * diag_noise
    I = I_NOMINAL - jnp.diag(jnp.diag(I_NOMINAL)) + jnp.diag(I_diag)
    I_inv = jnp.linalg.inv(I)    
    
    rw = x[0:3]
    Rwb = jnp.reshape(x[3:12], (3, 3))
    vb = x[12:15]
    wb = x[15:18]
    omega = x[18:22]

    omega_dot = (1/TAU_M) * (u - omega)

    rw_dot = Rwb @ vb 

    F_gravity = Rwb.T @ jnp.array([0, 0, -M*G]) 
    F_thrust = jnp.array([0, 0, KF1 * omega[0]**2 + KF2 * omega[1]**2 + KF3 * omega[2]**2 + KF4 * omega[3]**2])

    Faero = jnp.sum(jnp.abs(omega)) * (Kaero @ vb)
    
    Fb = F_gravity + F_thrust - Faero
    vb_dot = (1/M) * Fb - jnp.cross(wb, vb)

    skew_wb = jnp.array([
        [0, -wb[2], wb[1]],
        [wb[2], 0, -wb[0]],
        [-wb[1], wb[0], 0]
    ])
    Rwb_dot = Rwb @ skew_wb
    Rwb_dot_flat = Rwb_dot.flatten()

    tau_phi   = L * (-KF1*omega[0]**2 - KF2*omega[1]**2 + KF3*omega[2]**2 + KF4*omega[3]**2) / jnp.sqrt(2)
    tau_theta = L * (-KF1*omega[0]**2 + KF2*omega[1]**2 + KF3*omega[2]**2 - KF4*omega[3]**2) / jnp.sqrt(2)

    tau = jnp.array([tau_phi, tau_theta,  KM_R*(-omega[0]**2 + omega[1]**2 - omega[2]**2 + omega[3]**2)])

    wb_dot = I_inv @ (tau - jnp.cross(wb, I @ wb))

    return jnp.concatenate([rw_dot, Rwb_dot_flat, vb_dot, wb_dot, omega_dot])

#############################
# Integration
#############################

def skew(w):
    return jnp.array([
        [  0.0, -w[2],  w[1]],
        [ w[2],   0.0, -w[0]],
        [-w[1],  w[0],   0.0]
    ])

def exp_so3(w):
    theta_sq = jnp.dot(w, w)
    theta = jnp.sqrt(theta_sq)
    A = jnp.where(theta_sq < 1e-6, 1.0 - theta_sq / 6.0, jnp.sin(theta) / jnp.where(theta > 0, theta, 1.0))
    B = jnp.where(theta_sq < 1e-6, 0.5 - theta_sq / 24.0, (1.0 - jnp.cos(theta)) / jnp.where(theta > 0, theta_sq, 1.0))
    K = skew(w)
    return jnp.eye(3) + A * K + B * (K @ K)

def skew(w):
    return jnp.array([
        [  0.0, -w[2],  w[1]],
        [ w[2],   0.0, -w[0]],
        [-w[1],  w[0],   0.0]
    ])

def exp_so3(w):
    theta_sq = jnp.dot(w, w)
    theta = jnp.sqrt(theta_sq)
    A = jnp.where(theta_sq < 1e-6, 1.0 - theta_sq / 6.0, jnp.sin(theta) / jnp.where(theta > 0, theta, 1.0))
    B = jnp.where(theta_sq < 1e-6, 0.5 - theta_sq / 24.0, (1.0 - jnp.cos(theta)) / jnp.where(theta > 0, theta_sq, 1.0))
    K = skew(w)
    return jnp.eye(3) + A * K + B * (K @ K)

def rk4(x, u, key, dt):
    '''RK4 for integrating both euclidean and SE3 quantities, Formulas taken from https://arxiv.org/pdf/2205.12572, page 29 '''
    def split_state(s):
        return s[0:3], s[3:12].reshape((3, 3)), s[12:15], s[15:18], s[18:22]

    def pack_state(rw, Rwb, vb, wb, omega):
        return jnp.concatenate([rw, Rwb.flatten(), vb, wb, omega])

    rw_k, R_k, vb_k, wb_k, omega_k = split_state(x)

    # Step 1
    dx1 = dynamics(x, u, key)
    rw_dot1, _, vb_dot1, wb_dot1, omega_dot1 = split_state(dx1) 

    # Step 2
    rw_temp2 = rw_k + 0.5 * dt * rw_dot1
    vb_temp2 = vb_k + 0.5 * dt * vb_dot1
    wb_temp2 = wb_k + 0.5 * dt * wb_dot1
    omega_temp2 = omega_k + 0.5 * dt * omega_dot1
    R_temp2 = R_k @ exp_so3(0.5 * dt * wb_k)  
    x2 = pack_state(rw_temp2, R_temp2, vb_temp2, wb_temp2, omega_temp2)
    
    dx2 = dynamics(x2, u, key)
    rw_dot2, _, vb_dot2, wb_dot2, omega_dot2 = split_state(dx2)

    # Step 3
    rw_temp3 = rw_k + 0.5 * dt * rw_dot2
    vb_temp3 = vb_k + 0.5 * dt * vb_dot2
    wb_temp3 = wb_k + 0.5 * dt * wb_dot2
    omega_temp3 = omega_k + 0.5 * dt * omega_dot2
    R_temp3 = R_k @ exp_so3(0.5 * dt * wb_temp2) 
    x3 = pack_state(rw_temp3, R_temp3, vb_temp3, wb_temp3, omega_temp3)
    
    dx3 = dynamics(x3, u, key)
    rw_dot3, _, vb_dot3, wb_dot3, omega_dot3 = split_state(dx3)

    # Step 4
    rw_temp4 = rw_k + dt * rw_dot3
    vb_temp4 = vb_k + dt * vb_dot3
    wb_temp4 = wb_k + dt * wb_dot3
    omega_temp4 = omega_k + dt * omega_dot3
    R_temp4 = R_k @ exp_so3(dt * wb_temp3) 
    x4 = pack_state(rw_temp4, R_temp4, vb_temp4, wb_temp4, omega_temp4)
    
    dx4 = dynamics(x4, u, key)
    rw_dot4, _, vb_dot4, wb_dot4, omega_dot4 = split_state(dx4)

    # Final Integration Updates
    rw_next = rw_k + (dt / 6.0) * (rw_dot1 + 2.0 * rw_dot2 + 2.0 * rw_dot3 + rw_dot4)
    vb_next = vb_k + (dt / 6.0) * (vb_dot1 + 2.0 * vb_dot2 + 2.0 * vb_dot3 + vb_dot4)
    wb_next = wb_k + (dt / 6.0) * (wb_dot1 + 2.0 * wb_dot2 + 2.0 * wb_dot3 + wb_dot4)
    omega_next = omega_k + (dt / 6.0) * (omega_dot1 + 2.0 * omega_dot2 + 2.0 * omega_dot3 + omega_dot4)

    w_final = dt * wb_k + (dt**2 / 6.0) * (wb_dot1 + wb_dot2 + wb_dot3)
    R_next = R_k @ exp_so3(w_final)

    return pack_state(rw_next, R_next, vb_next, wb_next, omega_next)