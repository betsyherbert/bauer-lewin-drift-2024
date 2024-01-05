from modules import *

def get_path(folder_name, file_name):
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    folder_path = os.path.join(grandparent_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    return file_path

def circular_gaussian(N, theta, amp=2, sigma=20, baseline=0):
    theta_y = np.linspace(0, 180, N)  # center of tuning curves 
    d = np.abs(theta - theta_y)       # distance to input theta
    d_plus = d + 180
    d_minus = d - 180
    y = amp * ( np.exp(-(d**2)/(2*sigma**2)) + np.exp(-(d_plus**2)/(2*sigma**2)) + np.exp(-(d_minus**2)/(2*sigma**2))) + baseline
    return y

def ori_matrix_different_vars(N, vars, diagonal=True):
    x = np.linspace(0, 180, N)
    matrix = np.zeros((N, N))
    for i in range(N):
        mean = x[i]
        matrix[:, i] = stats.norm.pdf(x, mean, vars[i]) + stats.norm.pdf(x, mean+180, vars[i]) + stats.norm.pdf(x, mean-180, vars[i]) 
        if diagonal==False: matrix[i, i] = 0
    return matrix

def circular_distance(x, y):
    return np.minimum(np.abs(x-y), 180-np.abs(x-y))

def directional_circular_distance(x, y):
    difference = x-y
    difference[difference > 90] -= 180
    difference[difference < -90] += 180
    return difference

def convert_to_array(*args):
    for arg in args:
        arg = np.array(arg)
    return args

def propensity(w, a):
    return np.tanh(a*w)

def get_preferred_orientations(N, W, n_angles):
    posts = np.zeros((N, n_angles))
    for i, angle in enumerate(np.linspace(0, 181, n_angles)):
        y = circular_gaussian(N, angle, amp=1, sigma=5, baseline=0)
        posts[:, i] = W.T.dot(y)
    return 180 * np.argmax(posts, axis=1) / (n_angles)

def initialise_W(N, vars):
    W = ori_matrix_different_vars(N, vars, diagonal=True) / N   # initialise network with tuning curves distributed across stimulis space 
    W /= np.sum(W, axis=0)
    return W

def single_hebbian_component(N, W_old, theta_stim, type):    
    if type == 'baseline' or type == 'test': theta = np.random.uniform(0, 180)
    if type == 'stripe_rearing': theta = theta_stim
    u = circular_gaussian(N, theta, amp=0.62, sigma=60, baseline=0)
    v = W_old.T.dot(u)
    return np.outer(u, v)

def normalisation(W):
        W /= np.sum(W, axis=0) + 1e-10      # divisive normalisation and rectification

def prerun(W_init, theta_stim, a, hebb_scaling, rand_scaling, learning_rate, n_steps_per_norm, n_trials):
    N = W_init.shape[0]
    for t in range(n_trials):
        w = W_init
        H = single_hebbian_component(N, w, theta_stim, type='baseline'); eta = np.random.randn(N, N); prop_function = propensity(w, a)
        W_init = w + (hebb_scaling * H * prop_function + rand_scaling * eta * prop_function) * learning_rate
        if t % n_steps_per_norm == 0: normalisation(W_init)
    return W_init


def get_POs_over_trials(W_init, n_steps, a, hebb_scaling, rand_scaling, learning_rate, theta_stim, n_steps_per_norm, n_norm_per_day, n_test_angles, type):
    N = W_init.shape[0]
    POs = []; W = np.zeros((N, N, n_steps+1)); W[:, :, 0] = W_init
    for t in tqdm(range(n_steps)):
        W_old = W[:, :, t]
        H = single_hebbian_component(N, W_old, theta_stim, type=type)
        eta = np.random.randn(N, N)
        prop_function = propensity(W_old, a)
        hebb = hebb_scaling * H * prop_function
        rand = rand_scaling * eta * prop_function
        W_new = W_old + (hebb + rand) * learning_rate
        if t % n_steps_per_norm == 0: 
            normalisation(W_new)
            if t % (n_steps_per_norm * n_norm_per_day) == 0:
                PO = get_preferred_orientations(N, W_new, n_angles=n_test_angles); POs.append(PO)    # Save preferred orientations for each day
        W[:, :, t+1] = W_new
    return POs


def get_metrics(N, n_days, theta_stim, POs):
    preferences = np.array(POs).T 
    initial_preferences = np.linspace(0, 180, N)
    drift_magnitude = np.array([circular_distance(preferences[:, day], initial_preferences) for day in range(n_days)])
    drift_rate = np.array([circular_distance(preferences[:, day+1], preferences[:, day]) for day in range(n_days-1)])
    initial_distances = np.abs(initial_preferences - theta_stim)
    distances = np.abs(preferences - theta_stim)
    convergence = np.array([initial_distances - distances[:, day] for day in range(n_days-1)])
    return drift_magnitude, drift_rate, convergence

def evolve_weights(N, W_old, t, type, theta_stim, a, learning_rate, hebb_scaling, rand_scaling, n_steps_per_norm):
    H = single_hebbian_component(N, W_old, theta_stim, type=type)
    eta = np.random.randn(N, N)
    prop_function = propensity(W_old, a)
    hebb =  hebb_scaling * H * prop_function
    rand =  rand_scaling * eta * prop_function
    W_new = W_old + (hebb + rand) * learning_rate
    if t % n_steps_per_norm == 0: 
        normalisation(W_new)
    return W_new

def get_r_values(POs, theta_stim, n_days, N):
    preferences = np.array(POs).T
    initial_preferences = np.linspace(0, 180, N) 
    initial_distances = np.abs(initial_preferences - theta_stim)
    total_drift = np.array([circular_distance(preferences[:, day], initial_preferences) for day in range(n_days)]).T
    r_values = np.array([stats.spearmanr(initial_distances, total_drift[:, day])[0] for day in range(n_days)])
    return r_values