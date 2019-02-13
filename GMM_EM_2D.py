import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


def generate_samples(mu, sigma):
    samples = np.zeros(4000)
    np.random.seed(0)
    for i in range(len(mu)):
        samples[i*1000:(i+1)*1000] = np.random.normal(mu[i], sigma[i], 1000)
    print('sample size = ', np.shape(samples))
    plt.hist(samples, bins=70)
    plt.ylabel('Probability')
    plt.show()
    return samples


def initiate():
    pis = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    np.random.seed(0)
    mus = 5 * np.random.rand(4)
    np.random.seed(0)
    sigmas = np.random.rand(4)
    return pis, mus, sigmas


def solve(data, max_iter, pi, mu, sigma):

    print("\nstarting with:")
    print("pis = ", pi)
    print("mus = ", mu)
    print("vars = ", sigma)
    print()

    plot_data(data, mu, sigma, pi)

    converged = False
    wait = 0
    for it in range(max_iter):
        if not converged:
            """E-Step"""
            r, m_c, pi = e_step(data, mu, sigma, pi)

            """M-Step"""
            mu0, sigma = m_step(data, r, m_c)

            print("iteration", it+1, ":")
            print("pis = ", pi)
            print("mus = ", mu)
            print("sigmas = ", sigma)
            print()

            if it % 8 == 0:
                plot_data(data, mu, sigma, pi)

            """convergence condition"""
            shift = np.linalg.norm(np.array(mu) - np.array(mu0))
            mu = mu0
            if shift < 0.0001:
                wait += 1
                if wait > 10:
                    converged = True
            else:
                wait = 0


def e_step(data, mu, sigma, pi):
    clusters_number = len(pi)

    """creating estimations gaussian density functions"""
    gaussian_pdf_list = []
    for j in range(clusters_number):
        gaussian_pdf_list.append(norm(loc=mu[j], scale=sigma[j]))

    """Create the array r with dimensionality nxK"""
    r = np.zeros((len(data), clusters_number))

    """Probability for each data point x_i to belong to gaussian g """
    for c, g, p in zip(range(clusters_number), gaussian_pdf_list, pi):
        r[:, c] = p * g.pdf(data)

    """Normalize the probabilities 
    each row of r sums to 1 and weight it by mu_c == the fraction of points belonging to cluster c"""
    for i in range(len(r)):
        sum1 = np.dot([1 for i in range(clusters_number)], r[i, :].reshape(clusters_number, 1))
        r[i] = r[i] / sum1

    """calculate m_c
    For each cluster c, calculate the m_c and add it to the list m_c"""
    m_c = []
    for c in range(clusters_number):
        m = np.sum(r[:, c])
        m_c.append(m)

    """calculate pi
    probability of occurrence for each cluster"""
    for k in range(clusters_number):
        pi[k] = (m_c[k] / np.sum(m_c))

    return r, m_c, pi


def m_step(data, r, m_c):
    clusters_number = len(m_c)

    mu = []
    """calculate mu"""
    for k in range(clusters_number):
        mu.append(np.dot(r[:, k].reshape(len(data)), data.reshape(len(data)).T) / m_c[k])

    sigma = []
    """calculate sigma"""
    for c in range(clusters_number):
        sigma.append(np.sqrt(np.dot((r[:, c].reshape(len(data))).T, ((data.reshape(len(data)) - mu[c]) ** 2)) / m_c[c]))

    return mu, sigma


def plot_data(data, mu, var, p):
    """Plot the data"""
    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(111)

    """plot the hist"""
    ax0.hist(data, bins=50, alpha=0.2, density=True)

    """creating gausians"""
    gaussians_list = []
    g = 0
    for j in range(4):
        gaussians_list.append(p[j]*(norm(loc=mu[j], scale=var[j])).pdf(np.linspace(0.5, 4, num=60)))
        g += p[j] * (norm(loc=mu[j], scale=var[j])).pdf(np.linspace(0.5, 4, num=60))

    """Plot the gaussians"""
    for g1 in gaussians_list:
        ax0.plot(np.linspace(0.5, 4, num=60), g1, c='black')
    ax0.plot(np.linspace(0.5, 4, num=60), g, c='black')
    plt.show()


def main():
    """generate samples from different distributions"""
    mus = [1.0, 1.3, 2.0, 2.6]
    sigmas = [0.100, 0.316, 0.200, 0.224]
    samples1 = generate_samples(mus, sigmas)

    """initialize pi mu and sigma of distributions"""
    pis, mus, sigmas = initiate()

    """solve the problem using EM algorithm"""
    solve(samples1, 1500, pis, mus, sigmas)


if __name__ == "__main__":
    main()
