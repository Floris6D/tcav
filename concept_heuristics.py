import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression



# def reshape2(acts_concept: np.ndarray, acts_random: np.ndarray, printflag=False):
#     # Get original shapes
#     original_shape_concept = acts_concept.shape
#     original_shape_random = acts_random.shape
    
#     # Flatten each activation to 2D (nr_of_samples, X)
#     acts_concept = acts_concept.reshape(len(acts_concept), -1)
#     acts_random = acts_random.reshape(len(acts_random), -1)
    
#     # Print original sizes if reshaping was needed
#     if original_shape_concept != acts_concept.shape:
#         print(f"acts_concept reshaped from {original_shape_concept} to {acts_concept.shape}")
#     if original_shape_random != acts_random.shape:
#         print(f"acts_random reshaped from {original_shape_random} to {acts_random.shape}")
#     return acts_concept, acts_random


def reshape(acts, printflag=False):
    original_shape = acts.shape
    acts = acts.reshape(len(acts), -1)
    if original_shape != acts.shape and printflag:
        print(f"acts reshaped from {original_shape} to {acts.shape}")
    return acts



def compute_SVM_CAV(acts_concept: np.ndarray, acts_random: np.ndarray):
    """Calculates a CAV using a linear SVM.
    flattens the activations and trains a linear SVM to separate the concept activations from the random activations.
    acts_concept: numpy array, of which the first dimension is the number of samples and the rest are the activations.
    acts_random: numpy array, of which the first dimension is the number of samples and the rest are the activations.
    Returns: numpy array, the CAV.
    """
    acts_concept, acts_random = reshape(acts_concept), reshape(acts_random)
    
    # Create labels: 1 for acts_concept, 0 for acts_random
    X = np.vstack((acts_concept, acts_random))
    y = np.hstack((np.ones(len(acts_concept)), np.zeros(len(acts_random))))
    
    # Train a linear SVM
    svm = SVC(kernel='linear')
    svm.fit(X, y)
    cav = svm.coef_[0] / np.linalg.norm(svm.coef_[0])
    return cav


def compute_pattern_CAV(acts_concept: np.ndarray, acts_random: np.ndarray, model_type = "linear"):
    """
    Computes a pattern-based Concept Activation Vector (CAV).
    
    Parameters:
    - acts_concept: np.ndarray of shape (n_samples_c, *activation_shape)
    - acts_random: np.ndarray of shape (n_samples_r, *activation_shape)
    
    Returns:
    - pattern_cav: np.ndarray of shape (flattened_activation_dim,)
    """

    acts_concept, acts_random = reshape(acts_concept), reshape(acts_random)
    # Stack activations into one dataset
    A = np.vstack([acts_concept, acts_random]) 

    # Create labels: concept activations -> 1, random activations -> 0
    t = np.concatenate([np.ones(acts_concept.shape[0]), np.zeros(acts_random.shape[0])])  # Shape: (n_samples,)
    if model_type.lower() == "linear":
        # Fit a linear regression model
        model = LinearRegression(fit_intercept=True)
        model.fit(t.reshape(-1, 1), A)  
        # Extract pattern CAV: the estimated weight vector
        pattern_cav = model.coef_.flatten() / np.linalg.norm(model.coef_)
    elif model_type.lower() == "svm":
        model = SVR(kernel='linear')
        pattern_cav = np.zeros(A.shape[1])  # Initialize pattern vector
        
        for i in range(A.shape[1]):  # Train a separate SVR for each feature
            model.fit(t.reshape(-1, 1), A[:, i])  
            pattern_cav[i] = model.coef_[0][0]  # Extract coefficient
        
        pattern_cav /= np.linalg.norm(pattern_cav)  # Normalize the CAV
    else:
        print(f"Model type {model_type} not supported. Please choose 'linear' or 'svm'")
        return None

    return pattern_cav


def compute_RCV(acts, concept_values, model_type = "linear"):
    """
    Computes the Regression Concept Vector (RCV) for a given concept.
    acts: np.ndarray of shape (n_samples, *activation_shape)
    concept_values: np.ndarray of shape (n_samples,)
    """
    acts = reshape(acts)
    if model_type.lower() == "linear":
        model = LinearRegression(fit_intercept=True)
        model.fit(acts, concept_values)
        rcv = model.coef_ / np.linalg.norm(model.coef_)
        return rcv
    elif model_type.lower() == "svm":
        model = SVR(kernel='linear')
        model.fit(acts, concept_values)
        rcv = model.coef_
        return rcv / np.linalg.norm(rcv)
    else: 
        print(f"Model type {model_type} not supported. Please choose 'linear' or 'svm'")
        return None
    

def compute_CAR(acts_concept, acts_random):
    """
    Computes the Concept Activation Regions (CAR) for a given concept.
    acts_concept: np.ndarray of shape (n_samples_c, *activation_shape)
    acts_random: np.ndarray of shape (n_samples_r, *activation_shape)
    Retuns SVM object
    """
    acts_concept, acts_random = reshape(acts_concept), reshape(acts_random)
    X = np.vstack([acts_concept, acts_random])
    y = np.hstack([np.ones(acts_concept.shape[0]), np.zeros(acts_random.shape[0])])
    svm = SVC() #TODO whettehel is this kernel
    svm.fit(X, y)
    return svm #TODO how to go from svm to loss function

def compute_RCR(acts, concept_values):
    """
    Computes the Regression Concept Regions (RCR) for a given concept.
    acts: np.ndarray of shape (n_samples, *activation_shape)
    concept_values: np.ndarray of shape (n_samples,)
    Returns: SVR object
    """
    acts = reshape(acts)
    svr = SVR()
    svr.fit(acts, concept_values)
    return svr
    



def _sanity_check(X,Y):
    filter_cav = compute_SVM_CAV(X, Y)
    pattern_cav = compute_pattern_CAV(X, Y)
    pattern_SVM_cav = compute_pattern_CAV(X, Y, model_type = "svm")
    labels = np.hstack([(X[:,0] -1 )**2, np.zeros(Y.shape[0])]) * np.random.randint( 0, 5, X.shape[0] + Y.shape[0])
    XY = np.vstack([X, Y])
    rcv = compute_RCV(XY, labels)
    car = compute_CAR(X, Y)
    rcr = computte_RCR(XY, labels)
    from matplotlib import pyplot as plt
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.scatter(X[:,0], X[:,1], label= "Concept activations", alpha = 0.5, color = "Blue")
    plt.scatter(Y[:,0], Y[:,1], label = "Random activations", alpha = 0.5, color = "Gray")
    plt.plot([0, filter_cav[0]], [0, filter_cav[1]], label = "filter vector", color = "red") 
    plt.plot([0, pattern_cav[0]], [0, pattern_cav[1]], label = "pattern vector",  color= "green")
    plt.plot([0, pattern_SVM_cav[0]], [0, pattern_SVM_cav[1]], label = "pattern SVM vector",  color= "purple")
    plt.plot([0, rcv[0]], [0, rcv[1]], label = "RCV",  color= "orange")
    plt.legend()
    plt.tight_layout()
    plt.show()

def random_sample(mu, sigma, n: int) -> np.ndarray:
    assert n > 0, "Number of samples must be positive"
    if not( isinstance(mu, float) or isinstance(mu, int)):
        mu = np.array(mu)
        sigma = np.array(sigma)
        assert mu.shape == sigma.shape, "Shape of mu and sigma must be the same"
    return np.random.normal(mu, sigma, (n, len(mu)))


# #Settings for proving cav wrong over pattern-cav
X = random_sample([1,1], [1, 0.1], 100)
Y = random_sample([0,0], [1, 0.1], 100)
_sanity_check(X,Y)