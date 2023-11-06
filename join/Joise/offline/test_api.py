import os
import argparse
from API import api


def test_api(args):
     
    cpath = args.cpath
    save_root = args.save_root

    api(cpath, save_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
#     parser.add_argument('--lake', type=str, required=True, choices=["webtable", "opendata"])
    parser.add_argument('--cpath', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    args = parser.parse_args()
    test_api(args)
