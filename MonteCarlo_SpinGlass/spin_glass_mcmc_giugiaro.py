import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def generate_data(rho, n, p_e):
    """
    arguments : 
        - rho   : proba that the sign is flipped
        - n     : number of nodes
        - p_e   : proba of an edge 
    returns   : 
        - s    : vector of size n containing the labels 
        - y    : n x n symmetric matrix with 0 diagonal
        - edges: n x n matrix encoding the edges
    """
    s = np.random.choice([1., -1.], size=n)
    y = np.zeros((n, n))
    edges = np.zeros((n,n))

    for i in range(n):
        for j in range(i+1, n):
            edges[i,j] = np.random.binomial(1, p_e)

            if edges[i,j] == 1:
                y[i,j] = -(2 * np.random.binomial(1, (rho)) -1) * s[i] * s[j]
                y[j,i] = y[i,j]
        
            edges[j,i] = edges[i,j]

        
    return s, y, edges 

def log_ratio_posterior(s, i, y, rho, edges):
    """
    arguments : 
        - s     : vector containing the labels
        - i     : index such that s[i] is flipped
        - y     : n x n matrix
        - rho   : flip probability
        - edges : matrix encoding the edges in the graph
    returns   : 
        - log( P(s' | Y) / P(s | Y) ) where s' is the vector defined as s'[i] = -s[i] and s[j] = s[j] for j != i
    """
    n = np.size(s) 
    # alpha = 0
    # beta = 0

    # for j in range(n):
    #     if y[j,i] == s[j]*s[i]:
    #         alpha += 1
    #     if y[j,i] == -s[j]*s[i]:
    #         beta += 1

    alpha = np.count_nonzero(y[:,i] == s[i]*s)
    beta = np.count_nonzero(y[:,i] == - s[i]*s)

    log_eta = (alpha - beta) * np.log(rho/(1-rho))

    return log_eta

def run_pca(y):
    """
    arguments : 
        - y : n x n matrix
    returns   : 
        - eigenvector corresponding to the highest eigenvalue of y
    """

    pca = PCA()
    pca.fit(y)
    s_pca = pca.components_[0]

    return s_pca

def run_mcmc(y, T, rho, edges):
    """
    arguments : 
        - y   : n x n matrix
        - T   : number of iterations for mcmc
        - rho : flip probability
        - edges : matrix encoding the edges in the graph
    returns   : 
        - list of length T containing the mcmc iterates
    """
    n = np.size(y, 0)
    s0 = 2 * np.random.binomial(1, p=0.5, size=n) - 1
    s = np.copy(s0)

    # overlaps = []
    state_history = [np.copy(s0)]

    for t in range(T):
        i = np.random.randint(low=0, high=n)
        log_eta = log_ratio_posterior(s, i, y, rho, edges)
        
        prob_accept = min([1, np.exp(log_eta)])

        if (np.random.rand() < prob_accept):   #accept trial state
            s[i] = -s[i]  
            
        state_history.append(np.copy(s))
        # overlaps.append(np.mean(s * s_star))
        
    return state_history


def question_5():
    """
    Here goes the code for question 5 : plot the mixing time and overlap of the MCMC
    """
    T = 200000
    n = 800
    p_e = 4/n
    rhos = np.linspace(0.1, 0.3, 10)
    rhos = np.concatenate((rhos, np.linspace(0.35, 0.4, 2)))

    # rho values for single graph
    # rhos = np.array([0.1, 0.15, 0.2, 0.25, 0.35])

    Q_BO = []

   
    for rho in rhos:
        s_star, y, edges = generate_data(rho, n, p_e)

        # T_therm estimaton
        s0 = s_star
        s = np.copy(s0)

        configs_star = [np.copy(s0)]

        for t in range(T):
            i = np.random.randint(low=0, high=n)
            log_eta = log_ratio_posterior(s, i, y, rho, edges)
        
            prob_accept = min([1, np.exp(log_eta)])

            if (np.random.rand() < prob_accept):
                s[i] = -s[i]  
            
            configs_star.append(np.copy(s))
        
        Q_t_star = np.abs(np.einsum('ij,j->i', configs_star, s_star)) / n
        Q_t_star = np.array(Q_t_star)

        # simulation
        configs = run_mcmc(y, T, rho, edges)
        configs = np.array(configs)
    
        Q_t = np.abs(np.einsum('ij,j->i', configs, s_star)) / n
        Q_t = np.array(Q_t)
        
        T_therm = np.where(Q_t_star < Q_t)[0][0]
        print(T_therm)

        s_BO = np.sign(np.mean(configs[100000:, :], axis=0))
        Q_BO.append(np.abs(np.einsum('i,i', s_BO, s_star)) / n)

        # for multiple graphs
        plt.plot(Q_t)
        plt.plot(Q_t_star, label = "$\\rho$ = %.3f"%rho) 
        plt.xlabel("t")
        plt.ylabel("overlap")
        plt.ylim((0,1.05))
        plt.title(f"$n={n} \\rho={rho:.3f}$")
        plt.savefig(f"rho={rho}.png")
        plt.clf()

        # for single graph
    #     plt.plot(Q_t, label = "$\\rho$ = %.3f"%rho)
    #     plt.xlabel("t")
    #     plt.ylabel("overlap")
    #     plt.legend()
    #     plt.ylim((0,1.05))
    #     plt.title(f"$n={n}$")
    # plt.show()

    # for Q_BO vs rho
    plt.plot(rhos, Q_BO, label=r"$Q(s_{BO})$")
    plt.xlabel(r"$\rho$")
    plt.ylabel("overlap")
    plt.title(f"$n={n}$")
    plt.legend()
    plt.savefig("Q_BO_vs_rho.png")
    plt.clf()
    

def question_6():
    """
    Here goes the code for question 6 : compare the performance of PCA and BO
    """
    T = 200000
    n = 800
    p_e_val = np.linspace(2/n, 6/n, 100)
    rho = 0.25
    Q_BO = []
    Q_pca = []

    for p_e in p_e_val:
        s_star, y, edges = generate_data(rho, n, p_e)

        # PCA
        s_pca = np.sign(run_pca(y))
        Q_pca.append(np.abs(np.einsum('i,i', s_pca, s_star)) / n)

        # BO
        configs = run_mcmc(y, T, rho, edges)
        configs = np.array(configs)

        s_BO = np.sign(np.mean(configs[(100000):, :], axis=0))
        Q_BO.append(np.abs(np.einsum('i,i', s_BO, s_star)) / n)

    k = 5
    Q_BO_mobile_average = np.cumsum(np.asarray(Q_BO), dtype=float)
    Q_BO_mobile_average[k:] = Q_BO_mobile_average[k:] - Q_BO_mobile_average[:-k]
    Q_BO_mobile_average = Q_BO_mobile_average[k - 1:] / k

    Q_pca_mobile_average = np.cumsum(np.asarray(Q_pca), dtype=float)
    Q_pca_mobile_average[k:] = Q_pca_mobile_average[k:] - Q_pca_mobile_average[:-k]
    Q_pca_mobile_average = Q_pca_mobile_average[k - 1:] / k


    plt.plot(p_e_val, Q_BO, label=r"$Q(s_{BO})$")
    plt.plot(p_e_val, Q_pca, label=r"$Q(s_{PCA})$")
    plt.xlabel(r"$p_E$")
    plt.ylabel("overlap")
    plt.title(f"$n={n}$")
    plt.legend()
    plt.show()

    p_e_avg = p_e_val[2 : -2]

    plt.plot(p_e_avg, Q_BO_mobile_average, label=r"$Q(s_{BO})$")
    plt.plot(p_e_avg, Q_pca_mobile_average, label=r"$Q(s_{PCA})$")
    plt.xlabel(r"$p_E$")
    plt.ylabel("overlap")
    plt.title(f"$n={n}$")
    plt.legend()
    plt.show()


np.random.seed(1)
question_5()
# question_6()