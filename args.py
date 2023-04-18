import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("-J", type=float, default=1.0, help="J")
parser.add_argument("-mu", type=float, default=1, help="mu")
parser.add_argument("-hz", type=float, default=2.0, help="hz")
parser.add_argument("-d", type=int, default=2, help="d")
parser.add_argument("-D", type=int, default=10, help="D")
parser.add_argument("-MaxIter", type=int, default=4000, help="MaxStep")
# parser.add_argument("-Threshold", type=int, default=1E-10, help="Threshold")
parser.add_argument("-VarMethod", default='Normal', choices=['Normal', 'Grassmann'], help="VarMethod")

parser.add_argument("-lr", type=float, default=2, help="learn_rate")
parser.add_argument("-max_iter", type=int, default=2, help="max_iter")

args = parser.parse_args()