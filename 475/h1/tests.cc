// CSE 375/475 Assignment #1
// Spring 2023
//
// Description: This file implements a function 'run_custom_tests' that should be able to use
// the configuration information to drive tests that evaluate the correctness
// and performance of the map_t object.

#include <iostream>
#include <ctime>
#include <string>
#include <thread>
#include <vector>
#include <future>
#include <chrono>
#include "config_t.h"
#include "tests.h"
#include "cashmap.h"

using namespace std;

mt19937_64 gen(random_device{}());
uniform_real_distribution<> dis(0, 1);

void run_custom_tests(config_t& cfg) {
    cashmap cm(cfg);

    cm.populate(10000, 10000000.);

    // deposit 95% of the time
    // balance 5% of the time
    auto do_work = [&](int iters, promise<double> &&promise) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            if (dis(gen) < 0.95) {
                cm.deposit();
            } else {
                cm.balance();
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        promise.set_value(elapsed / 1000.0);
    };

    // create threads and futures
    pair<thread, future<double>> threads[cfg.threads];
    int thread_iters = cfg.iters / cfg.threads;
    for (int i = 0; i < cfg.threads; i++) {
        promise<double> promise;
        threads[i].second = promise.get_future();
        threads[i].first = thread(do_work, thread_iters, std::move(promise));
    }

    // collect times from futures
    double times[cfg.threads];
    for (int i = 0; i < cfg.threads; i++) {
        times[i] = threads[i].second.get();
        threads[i].first.join();
    }

    // get longest running thread
    double longest;
    for (auto t : times) {
        longest = max(longest, t);
    }
    cout << longest << endl;
}

void test_driver(config_t &cfg) {
    run_custom_tests(cfg);
}
