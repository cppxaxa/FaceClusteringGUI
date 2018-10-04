import argparse

parser = argparse.ArgumentParser(description='Sample 1')
parser.add_argument('input', metavar='input_path')
parser.add_argument('--output', dest='output', metavar='output_path')
args = parser.parse_args()

print(str(args.input) + ' ' + str(args.output))

input("Press Enter to continue...")


