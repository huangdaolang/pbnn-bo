from utils import *
import os
import argparse
from solver import solver_nn


def main(args):
    elicitation = args.elicitation
    n_train_pair = args.n_train_pair
    n_query_pair = args.n_query_pair
    n_test_pair = args.n_test_pair
    n_acq_al = args.n_acq_al
    n_acq_bo = args.n_acq_bo
    seed = args.seed
    biased_level = args.biased_level
    dataset = args.dataset
    bo_acq = args.bo_acquisition

    root_name = f"Sim/{dataset}_acc{int(biased_level*100)}"
    if not os.path.exists(root_name):
        os.makedirs(root_name)

    train_al, query_al, test_al, query_bo = get_data(dataset, n_train_pair, n_query_pair, n_test_pair, biased_level, seed)

    if elicitation:
        al_acq = args.al_acquisition
    else:
        al_acq = None

    min_nn, _ = solver_nn(train_al, query_al, test_al, query_bo, n_acq_al, n_acq_bo, al_acq=al_acq, bo_acq=bo_acq)

    print('Saving results...')
    np.save(f"{root_name}/nn_{str(seed)}.npy", min_nn)
    print(min_nn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--elicitation", type=bool, default=True, help="whether knowledge elicitation is used")
    parser.add_argument("--bo-acquisition", type=str, default="EI", help="acquisition for BO")
    parser.add_argument("--al-acquisition", type=str, default="BALD", help="acquisition for active learning")
    parser.add_argument("--n-train-pair", type=int, default=1, help="initial number of elicitation training pairs")
    parser.add_argument("--n-query-pair", type=int, default=2000, help="active learning query pool size")
    parser.add_argument("--n-test-pair", type=int, default=1000, help="active learning test data size")
    parser.add_argument("--n-acq-al", type=int, default=100, help="number of active learning acquisitions")
    parser.add_argument("--n-acq-bo", type=int, default=50, help="number of bayesian optimization acquisitions")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--biased-level", type=float, default=0.9, help="accuracy of the expert", choices=[0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--dataset", type=str, default="six_hump_camel", help="dataset name", choices=["forrester", "branin", "six_hump_camel", "levy"])

    args = parser.parse_args()
    main(args)
