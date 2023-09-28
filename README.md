# Bayesian Optimization Augmented with Actively Elicited Expert Knowledge: Code Implementation

This is an initial code implementation for the paper "Bayesian Optimization Augmented with Actively Elicited Expert Knowledge."

Note: This is not the final version of the code corresponding to the paper. The finalized version will be released once the paper is formally accepted.

## Installation
````
git clone https://github.com/huangdaolang/PBNN-BO.git

cd PBNN-BO

conda create --name PNBB-BO python=3.8

conda activate PNBB-BO

pip install -r requirements.txt
````

## Quick Start
````
python main.py --dataset="six_hump_camel" --biased-level=0.9 --seed=1
````


## License
This code is under the MIT License.