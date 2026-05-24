import numpy as np
from scipy.special import expit

def sample_uniform_disjoint(low, high, size):
    """Sample uniformly from [-high, -low] \cup [low, high]"""
    signs = 2 * (np.random.choice(2, size=size) - 0.5)
    vals = np.random.uniform(low=low, high=high, size=size)
    return signs * vals

class LinearTwoMediatorsProxDGP:
    """Linear data generating process with two mediators (M1 and M2) and proxies Z, W
    X → A → M1 → M2 → Y
    X → A → M1 → Z
    X → M1 → W
    X → A, M1, M2, W → Y
    """
    
    def __init__(
        self,
        xdim,       # dimension of covariates X
        zdim,       # dimension of proxy Z (for M1)
        wdim,       # dimension of proxy W (for M2)
        m1dim,      # dimension of mediator M1
        m2dim,      # dimension of mediator M2
        ydim=1,     # dimension of outcome Y
        l=0.5,      # lower bound for parameter values
        u=1.0,      # upper bound for parameter values
        var=0.5,    # noise variance
        nonnegative=False,
        azwy_nonnegative=False,
        seed=0,
    ):
        np.random.seed(seed)
        
        self.xdim = xdim
        self.zdim = zdim
        self.wdim = wdim
        self.m1dim = m1dim
        self.m2dim = m2dim
        self.ydim = ydim
        
        sampler = np.random.uniform if nonnegative else sample_uniform_disjoint
        azwy_sampler = np.random.uniform if azwy_nonnegative else sampler
        
        # Treatment assignment: A = 1 if sigmoid(X @ Wxa + noise) > 0.5
        self.Wxa = sampler(low=0.4*l, high=0.4*u, size=(xdim, 1)) * (1./np.sqrt(xdim))
        
        # M1 (first mediator): M1 = X @ Wxm1 + A @ Wam1 + eps_m1
        self.Wxm1 = sampler(low=l, high=u, size=(xdim, m1dim)) * (1./np.sqrt(xdim))
        self.Wam1 = sampler(low=l, high=u, size=(1, m1dim))
        
        # M2 (second mediator): M2 = X @ Wxm2 + A @ Wam2 + M1 @ Wm1m2 + eps_m2
        self.Wxm2 = sampler(low=l, high=u, size=(xdim, m2dim)) * (1./np.sqrt(xdim))
        self.Wam2 = sampler(low=l, high=u, size=(1, m2dim))
        self.Wm1m2 = sampler(low=l, high=u, size=(m1dim, m2dim)) * (1./np.sqrt(m1dim))
        
        # Proxy Z: Z = X @ Wxz + A @ Waz + M1 @ Wm1z + eps_z
        self.Wxz = sampler(low=l, high=u, size=(xdim, zdim)) * (1./np.sqrt(xdim))
        self.Waz = azwy_sampler(low=l, high=u, size=(1, zdim))
        self.Wm1z = sampler(low=l, high=u, size=(m1dim, zdim)) * (1./np.sqrt(m1dim))
        
        # Proxy W: W = X @ Wxw + M1 @ Wm1w + eps_w
        self.Wxw = sampler(low=l, high=u, size=(xdim, wdim)) * (1./np.sqrt(xdim))
        self.Wm1w = sampler(low=l, high=u, size=(m1dim, wdim)) * (1./np.sqrt(m1dim))
        
        # Outcome Y: Y = X @ Wxy + A @ Way + M1 @ Wm1y + M2 @ Wm2y + W @ Wwy + eps_y
        self.Wxy = 2 * sampler(low=l, high=u, size=(xdim, ydim)) * (1./np.sqrt(xdim))
        self.Way = sampler(low=l, high=u, size=(1, ydim))
        self.Wm1y = sampler(low=l, high=u, size=(m1dim, ydim)) * (1./np.sqrt(m1dim))
        self.Wm2y = sampler(low=l, high=u, size=(m2dim, ydim)) * (1./np.sqrt(m2dim))
        self.Wwy = azwy_sampler(low=l, high=u, size=(wdim, ydim)) * (1./np.sqrt(wdim))
        
        # Covariances
        self.xcov = np.eye(xdim)
        self.m1cov = var * np.eye(m1dim)
        self.m2cov = var * np.eye(m2dim)
        self.ycov = var * np.eye(ydim)
        self.zcov = var * np.eye(zdim)
        self.wcov = var * np.eye(wdim)
        self.acov = var  # noise for A (used in the threshold model)
    
    def sample_dataset(self, n, seed=None):
        """Sample a dataset of size n"""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate noise terms
        eps_m1 = np.random.multivariate_normal(np.zeros(self.m1dim), self.m1cov, n)
        eps_m2 = np.random.multivariate_normal(np.zeros(self.m2dim), self.m2cov, n)
        eps_y = np.random.multivariate_normal(np.zeros(self.ydim), self.ycov, n)
        eps_z = np.random.multivariate_normal(np.zeros(self.zdim), self.zcov, n)
        eps_w = np.random.multivariate_normal(np.zeros(self.wdim), self.wcov, n)
        
        # Generate covariates X
        X = np.random.multivariate_normal(np.zeros(self.xdim), self.xcov, n)
        
        # Generate treatment A (binary) - using Bernoulli distribution
        # A ~ Bernoulli(p = sigmoid(X @ Wxa + noise))
        a_noise = np.random.normal(0, self.acov, n).reshape(-1, 1)
        a_logit = X @ self.Wxa + a_noise
        A_probs = expit(a_logit).flatten()
        A = np.random.binomial(1, A_probs).reshape(-1, 1)
        
        # Generate mediators M1 and M2 (linear)
        M1 = X @ self.Wxm1 + A @ self.Wam1 + eps_m1
        M2 = X @ self.Wxm2 + A @ self.Wam2 + M1 @ self.Wm1m2 + eps_m2
        
        # Generate proxies Z and W
        Z = X @ self.Wxz + A @ self.Waz + M1 @ self.Wm1z + eps_z
        W = X @ self.Wxw + M1 @ self.Wm1w + eps_w
        
        # Generate outcome Y
        Y = (
            X @ self.Wxy
            + A @ self.Way
            + M1 @ self.Wm1y
            + M2 @ self.Wm2y
            + W @ self.Wwy
            + eps_y
        )
        
        data = {
            'x': X,
            'z': Z,
            'w': W,
            'a': A,
            'm1': M1,
            'm2': M2,
            'y': Y,
            'a_p': A_probs.reshape(-1, 1),  # Treatment probabilities
        }
        
        return data
    
    def true_psi(self, data):
        """Compute true value of potential outcome Y(M2(1, M1(0)), M1(0), 0)"""
        X = data['x']
        n = X.shape[0]
        
        # Step 1: Generate M1 under intervention A=0
        A0 = np.zeros((n, 1))
        M1_A0 = X @ self.Wxm1 + A0 @ self.Wam1
        
        # Step 2: Generate M2 under interventions A=1 and M1(0)
        A1 = np.ones((n, 1))
        M2_A1_M1_A0 = X @ self.Wxm2 + A1 @ self.Wam2 + M1_A0 @ self.Wm1m2
        
        # Step 3: Generate Y under interventions A=0, M1(0), and M2(1, M1(0))
        # Note: W depends on M1(0)
        W = X @ self.Wxw + M1_A0 @ self.Wm1w
        
        Y = (
            X @ self.Wxy
            + A0 @ self.Way           # A=0
            + M1_A0 @ self.Wm1y       # M1(0)
            + M2_A1_M1_A0 @ self.Wm2y # M2(1, M1(0))
            + W @ self.Wwy            # W (based on M1(0))
        )
        
        return Y.mean()  # Return the expected value E[Y(M2(1, M1(0)), M1(0), 0)]
    
    def true_psi_x(self, data):
        """Compute true value of potential outcome Y(M2(1, M1(0)), M1(0), 0) pointwise"""
        X = data['x']
        n = X.shape[0]
        
        # Step 1: Generate M1 under intervention A=0
        A0 = np.zeros((n, 1))
        M1_A0 = X @ self.Wxm1 + A0 @ self.Wam1
        
        # Step 2: Generate M2 under interventions A=1 and M1(0)
        A1 = np.ones((n, 1))
        M2_A1_M1_A0 = X @ self.Wxm2 + A1 @ self.Wam2 + M1_A0 @ self.Wm1m2
        
        # Step 3: Generate Y under interventions A=0, M1(0), and M2(1, M1(0))
        # Note: W depends on M1(0)
        W = X @ self.Wxw + M1_A0 @ self.Wm1w
        
        Y = (
            X @ self.Wxy
            + A0 @ self.Way           # A=0
            + M1_A0 @ self.Wm1y       # M1(0)
            + M2_A1_M1_A0 @ self.Wm2y # M2(1, M1(0))
            + W @ self.Wwy            # W (based on M1(0))
        )
        
        return Y  # Return the pointwise values E[Y(M2(1, M1(0)), M1(0), 0) | X]

    def get_true_nested_effect(self, data):
        """Get the true value of the nested potential outcome"""
        return {
            'E_Y_nested': self.true_psi(data)
        }
        

def main():
    # Test the new DGP
    datagen = LinearTwoMediatorsProxDGP(
        xdim=10,    # dimension of covariates
        zdim=2,     # dimension of proxy for M1
        wdim=2,     # dimension of proxy for M2
        m1dim=3,    # dimension of first mediator
        m2dim=3,    # dimension of second mediator
        seed=1
    )
    
    data_fit = datagen.sample_dataset(n=100, seed=1)
    
    print("Dataset shapes:")
    for key, value in data_fit.items():
        print(f"{key}: shape {value.shape}, dtype {value.dtype}")
    
    # Test the calculation of nested potential outcomes
    nested_result = datagen.true_psi_x(data_fit)
    print(nested_result.shape)
    # print(f"\nTrue value of nested potential outcome E[Y(M2(1, M1(0)), M1(0), 0)]: {nested_result.mean():.4f}")


if __name__ == "__main__":
    main()