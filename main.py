import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def calcP4(S_n_max, queue_lens, length, V, Tau_t):
    x_t = [1 if i+1 in S_n_max else 0 for i in range(0, length)]
    factor = max(x_t[i] * Tau_t[i] for i in range(0, len(x_t)))
    return V*factor - np.dot(x_t, queue_lens)


def divide_and_conquer(est_exchange_times, expected_num_of_arms, arm_indicators, queue_lens, V):

    set_dict = {}
    N_t = [i+1 for i in range(len(arm_indicators)) if arm_indicators[i] == 1]
    Z_t = [queue_lens[i] for i in range(len(arm_indicators)) if arm_indicators[i] == 1]
    A_t = [n for _, n in sorted(zip(Z_t, N_t), reverse=True)]
    k = min(expected_num_of_arms, sum(arm_indicators))
    Tau_t = est_exchange_times

    # print(f'Indicator function of arms: {arm_indicators}')
    # print(f'Estimated times for model exchange: {est_exchange_times}')
    # print(f'Length of virtual quques: {queue_lens}')
    # print("-------------------------")
    # print(f'Zt = {Z_t}')
    # print(f'Nt = {N_t}')
    # print(f'At = {A_t}')
    # print(f'k = {k}')
    # print(f'Tau_t = {Tau_t}')
    # print("-------------------------")
    # print("Possible sets:")

    for n_max in N_t:
        S_n_max = []
        for n in A_t:
            if Tau_t[n-1] <= Tau_t[n_max - 1]:
                S_n_max.append(n)
            if len(S_n_max) == k:
                F_n_max = calcP4(S_n_max, queue_lens, len(arm_indicators), V, Tau_t)
                # print(f'F_n_max = {F_n_max}, n_max = {n_max}, S_n_max = {S_n_max}')
                set_dict[n_max] = (S_n_max, F_n_max)
                break
    Fmin = np.infty

    S_chosen = 0
    n_chosen = 0
    for key in set_dict.keys():
        tmpS, tmpF = set_dict[key]
        if tmpF < Fmin:
            Fmin = tmpF
            S_chosen = tmpS
            n_chosen = key
    # print("------------------------------")
    # print("Optimal parameters:")
    # print(f'Fmin = {Fmin} \nS_chosen = {S_chosen} \nn_chosen = {n_chosen}')

    if type(S_chosen) == list:
        return [1 if i+1 in S_chosen else 0 for i in range(0, len(arm_indicators))]
    else:
        return [0 for i in range(0, len(arm_indicators))]


def RBCS_F(expected_num_of_arms, exploration_vector, client_set, ridge_lambda, beta, balance_param, total_time):

    m = expected_num_of_arms
    alpha = exploration_vector
    N = client_set
    Lambda = ridge_lambda
    Beta = beta
    V = balance_param

    H = []
    b = []
    z = []
    I = [[0] * len(N)]

    c = [[[0] * 3] * len(N)]
    Mu = [[0] * len(N)]
    s = [[0] * len(N)]
    B = [[0] * len(N)]
    M = 20

    x = [[0] * len(N)]
    Tau_avg = [[0] * len(N)]
    Tau_estimated = [[0] * len(N)]
    Tau = [[0] * len(N)]
    Theta = [[0] * len(N)]

    for j in range(len(N)):
        H.append([[Lambda, 0, 0], [0, Lambda, 0], [0, 0, Lambda]])
        b.append([[0.0], [0.0], [0.0]])
        z.append(0)

    # print("------------------------------")
    # print('Initialization:')
    # print(f'H matrix is initialized with: \n{H}')
    # print(f'b matrix is initialized with: \n{b}')
    # print(f'Original quque sizes are: \n{z}')
    # print("------------------------------")

    for t in range(1, total_time+1):
        print(f'\nFor t = {t} :\n')
        Mu.append(np.random.random(size=len(N)))
        s.append(x[-1])
        B.append(np.random.randint(2, 4, len(N)) * 1.0)

        c.append([np.transpose([1/Mu[t][i], s[t][i], M/B[t][i]]).tolist() for i in range(len(N))])
        I.append(np.random.randint(0, 2, len(N)))

        # print(f'Observing Contexts and arms availabilities:')
        # print(f'Contexs: \n{c[t]}')
        # print(f'Arm availabilities: \n{I[t]}')

        Theta_tmp = []
        Tau_est_tmp = []
        Tau_avg_tmp = []

        for n in N:
            if t == 1:
                Theta_tmp.append(np.matmul(inv(H[n-1]), np.matrix(b[n-1])).tolist())
                Tau_est_tmp.append((np.matmul(np.transpose(c[t][n-1]), Theta_tmp[n-1]))[0])
                tmp = np.sqrt(np.matmul(np.matmul(c[t][n-1], inv(H[n-1])), np.transpose(c[t][n-1])))
                Tau_avg_tmp.append(max(Tau_est_tmp[n-1] - alpha[t] * tmp, 0))

            else:
                Theta_tmp.append(np.matmul(inv(H[t-1][n-1]), np.matrix(b[t-1][n-1])).tolist())
                Tau_est_tmp.append((np.matmul(np.transpose(c[t][n-1]), Theta_tmp[n-1]))[0])
                tmp = np.sqrt(np.matmul(np.matmul(c[t][n-1], inv(H[t-1][n-1])), np.transpose(c[t][n-1])))
                Tau_avg_tmp.append(max(Tau_est_tmp[n-1] - alpha[t] * tmp, 0))

        if t == 1:
            Theta = [Theta, Theta_tmp]
            Tau_estimated = [Tau_estimated, Tau_est_tmp]
            Tau_avg = [Tau_avg, Tau_avg_tmp]
            x.append(divide_and_conquer(Tau_avg[t], m, I[t], z, V))
        else:
            Theta.append(Theta_tmp)
            Tau_estimated.append(Tau_est_tmp)
            Tau_avg.append(Tau_avg_tmp)
            x.append(divide_and_conquer(Tau_avg[t], m, I[t], z[t-1], V))

        Tau_tmp = []

        for n in N:
            if n <= 10:
                Theta_star = [1, 1, 1/np.log(1 + 1000)]
            elif n <= 20:
                Theta_star = [2, 1, 1/np.log(1 + 100)]
            elif n <= 30:
                Theta_star = [3, 1, 1/np.log(1 + 10)]
            else:
                Theta_star = [4, 1, 1/np.log(1 + 1)]

            time_dist = np.matmul(np.transpose(c[t][n-1]), Theta_star)

            Tau_tmp.append(time_dist - np.random.uniform(-time_dist, time_dist))  # MINUS BECAUSE OF uniform function

        Tau.append(Tau_tmp)

        # print("------------------------------")
        # print(f'x = {x} ')
        # print("------------------------------")
        # print(f'Distribute model and check real exchange times:')
        # print(f'Tau = {Tau}')

        z_tmp = []
        H_tmp = []
        b_tmp = []

        for n in N:
            if t == 1:
                z_tmp.append(max(0, z[n-1] + Beta - x[t][n-1]))
                H_tmp.append((np.array(H[n-1]) + x[t][n-1] * np.matmul([c[t][n - 1]], np.transpose([c[t][n - 1]]))).tolist())
                b_tmp.append((np.array(b[n-1]) + x[t][n-1] * Tau[t-1][n-1] * np.transpose([c[t][n - 1]])).tolist())

            else:
                z_tmp.append(max(0, z[t-1][n-1] + Beta - x[t][n-1]))
                H_tmp.append((np.array(H[t-1][n-1]) + x[t][n-1] * np.matmul([c[t][n - 1]], np.transpose([c[t][n - 1]]))).tolist())
                b_tmp.append((np.array(b[t-1][n-1]) + x[t][n-1] * Tau[t-1][n-1] * np.transpose([c[t][n - 1]])).tolist())

        if t == 1:
            z = [z, z_tmp]
            H = [H, H_tmp]
            b = [b, b_tmp]
        else:
            z.append(z_tmp)
            H.append(H_tmp)
            b.append(b_tmp)

        # print("------------------------------")
        # print('Update:')
        # print(f'H matrix is updated with: \n{H}')
        # print(f'b matrix is updated with: \n{b}')
        # print(f'Updated quque sizes are: \n{z}')
        # print("------------------------------")

    return z, x


if __name__ == '__main__':
    T = 500
    Z, X = RBCS_F(8, [0.1] * (T+1), [i for i in range(1, 41)], 1, 0.15, 5, T)
    Z_total = []
    for time in range(1, T+1):
        # print(f'Time = {time}:\nclients chosen are: {X[time]}\nclients chosen are: {Z[time]}')
        Z_total.append(sum(Z[time]))
    # print(Z_total)

    plt.plot(Z_total, color='blue')  # plot the data
    plt.xticks(range(0, len(Z_total) + 1, 100))  # set the tick frequency on x-axis

    plt.ylabel('Total Queue Length')  # set the label for y axis
    plt.xlabel('Federated Rounds')  # set the label for x-axis
    plt.title("SHOULD SHOW THE IMPACT OF V")  # set the title of the graph
    plt.show()  # display the graph
