// CSE 375/475 Assignment #1
// Spring 2023
//
// Description: This file parses command line options, outputs them, and
// then runs the test function.

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>
#include "config_t.h"
#include "tests.h"

using namespace std;

// Report on how to use the command line to configure this program
void usage() {
    cout
        << "Command-Line Options:" << endl
        << "  -n <string> : a phrase to help differentiate experiments" << endl
        << "  -i <int>    : the number of total iterations" << endl
        << "  -t <int>    : the number of threads in the experiment" << endl
        << "  -h          : display this message and exit" << endl << endl;
    exit(0);
}

// Parse command line arguments using getopt()
void parseargs(int argc, char** argv, config_t& cfg) {
    // parse the command-line options
    int opt;
    while ((opt = getopt(argc, argv, "n:k:i:t:h")) != -1) {
        switch (opt) {
          case 'n': cfg.name = string(optarg); break;
          case 'i': cfg.iters = atoi(optarg); break;
          case 't': cfg.threads = atoi(optarg); break;
          case 'h': usage(); break;
        }
    }
}

// The main routine simply parses the arguments, dumps the arguments, populates the
int main(int argc, char** argv) {
    // get the configuration, print it
    config_t config;
    parseargs(argc, argv, config);
    config.dump();

    // crash if the iterations are negative
    assert(config.iters > 0);

    // launch the tests
    test_driver(config);
}
