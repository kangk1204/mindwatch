"""Command-line errors must not echo private path or input arguments."""
import argparse


class PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, "Invalid command line. Use --help to check the supported arguments.\n")
