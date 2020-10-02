"""this script copy all a directory structure and resize all the images to a specific size
"""
import os
from src.argument_parser import parse_config
from src.utils import copy_and_resize


def main():
  arguments = [[str, "source", "", "directory to copy", lambda x: x != "" and os.path.isdir(x)],
               [str, "dest", "", "destination of the copy", lambda x: x != ""],
               [int, "img_size",  256, "height and width of the images after copy (the image is a square)"]]
  args = parse_config(arguments)
  copy_and_resize(args.source, args.dest, args.img_size)


if __name__ == "__main__":
  main()
