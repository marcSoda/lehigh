#include <iostream>
#include <random>
#include <thread>
#include <future>
#include <chrono>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>
#include "seq_set.h"
#include "con_set.h"
#include "tran_set.h"

using namespace std;

struct stats {
    int num_adds;
    int num_removes;
    int num_contains;
    int add_calls;
    int rem_calls;

    stats(int adds, int rems, int conts, int addcalls, int remcalls) :
        num_adds(adds), num_removes(rems), num_contains(conts), add_calls(addcalls), rem_calls(remcalls) {}
    stats() : num_adds(0), num_removes(0), num_contains(0), add_calls(0), rem_calls(0) {}

    void add(const stats& s) {
        num_adds += s.num_adds;
        num_removes += s.num_removes;
        num_contains += s.num_contains;
        add_calls += s.add_calls;
        rem_calls += s.rem_calls;
    }
};

long seq_test(int num_ops, int num_elms, int max_val, int capacity) {
    seq_set<long, long> set(capacity);
    cout << "startpop" << endl;
    set.populate(num_elms, max_val);
    int initial_size = set.size();
    cout << "endpop" << endl;
    cout << set.size() << " initial size" << endl;
    cout << set._num_resize << " num_resize after pop" << endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, max_val);
    std::uniform_real_distribution<> op_dis(0, 1);
    size_t num_contains = 0, num_adds = 0, num_removes = 0, num_add_calls = 0, num_rem_calls = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_ops; i++) {
        double op = op_dis(gen);
        int item = dis(gen);

        if (op < 0.8) {
            set.contains(item);
            num_contains++;
        } else if (op < 0.9) {
            if (set.add(item, item))
                num_adds++;
            num_add_calls++;
        } else {
            if (set.remove(item))
                num_removes++;
            num_rem_calls++;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    double add_throughput = static_cast<double>(num_adds) / total_time;
    double remove_throughput = static_cast<double>(num_removes) / total_time;
    double contains_throughput = static_cast<double>(num_contains) / total_time;
    double overall_throughput = static_cast<double>((num_adds + num_removes + num_contains)) / total_time;

    cout << "Bench:\n\t"
        << set.size() << " size,\n\t"
        << initial_size + num_adds - num_removes << " expected size,\n\t"
        << set._num_resize << " num resizes,\n\t"
        << num_contains << " contains (" << contains_throughput << " ops/sec),\n\t"
        << num_adds << " adds (" << add_throughput << " ops/sec),\n\t"
        << num_removes << " removes (" << remove_throughput << " ops/sec),\n\t"
        << num_add_calls << " add calls,\n\t"
        << num_rem_calls << " rem calls,\n\t"
        << "Overall throughput: " << overall_throughput << " ops/sec" << endl;
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

long con_test(int num_ops, int num_elms, int max_val, int capacity, int num_threads) {
    cout << "start" << endl;
    con_set<long, long> set(capacity);
    cout << "startpop" << endl;
    set.populate(num_elms, max_val);
    cout << "donepop" << endl;
    int initial_size = set.size();
    cout << set.size() << " initial size" << endl;
    stats stat;
    auto do_work = [&](int iters) {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(1, max_val);
        uniform_real_distribution<> op_dis(0, 1);

        stats s;
        for (size_t i = 0; i < iters; i++) {
            double op = op_dis(gen);
            int item = dis(gen);

            if (op < 0.8) {
                set.contains(item);
                s.num_contains++;
            } else if (op < 0.9) {
                if (set.add(item, item))
                    s.num_adds++;
                s.add_calls++;
            } else {
                if (set.remove(item))
                    s.num_removes++;
                s.rem_calls++;
            }
        }
        return s;
    };

    auto start = std::chrono::high_resolution_clock::now();
    tbb::parallel_for(tbb::blocked_range<int>(0, num_threads), [&](const tbb::blocked_range<int>& range) {
        stats local_stat;
        for (int i = range.begin(); i != range.end(); i++) {
            local_stat.add(do_work(num_ops / num_threads));
        }
        stat.add(local_stat);
    });
    auto end = std::chrono::high_resolution_clock::now();

    double total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    double add_throughput = stat.num_adds / total_time;
    double remove_throughput = stat.num_removes / total_time;
    double contains_throughput = stat.num_contains / total_time;
    double overall_throughput = (stat.num_adds + stat.num_removes + stat.num_contains) / total_time;

    cout << "Bench:\n\t"
        << set.size() << " size,\n\t"
        << initial_size + stat.num_adds - stat.num_removes << " expected size,\n\t"
        << stat.num_contains << " contains (" << contains_throughput << " ops/sec),\n\t"
        << stat.num_adds << " adds (" << add_throughput << " ops/sec),\n\t"
        << stat.num_removes << " removes (" << remove_throughput << " ops/sec),\n\t"
        << stat.add_calls << " add calls,\n\t"
        << stat.rem_calls << " rem calls,\n\t"
        << "Overall throughput: " << overall_throughput << " ops/sec" << endl;
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

long tran_test(int num_ops, int num_elms, int max_val, int capacity, int num_threads) {
    cout << "start" << endl;
    tran_set<long, long> set(capacity);
    cout << "startpop" << endl;
    set.populate(num_elms, max_val);
    int initial_size = set.size();
    cout << set.size() << " initial size" << endl;
    cout << set._num_resize << " num_resize after pop" << endl;
    cout << "donepop" << endl;
    stats stat;
    auto do_work = [&](int iters) {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(1, max_val);
        uniform_real_distribution<> op_dis(0, 1);

        stats s;
        for (size_t i = 0; i < iters; i++) {
            double op = op_dis(gen);
            int item = dis(gen);

            if (op < 0.8) {
                set.contains(item);
                s.num_contains++;
            } else if (op < 0.9) {
                if (set.add(item, item))
                    s.num_adds++;
                s.add_calls++;
            } else {
                if (set.remove(item))
                    s.num_removes++;
                s.rem_calls++;
            }
        }
        return s;
    };

    auto start = std::chrono::high_resolution_clock::now();
    tbb::parallel_for(tbb::blocked_range<int>(0, num_threads), [&](const tbb::blocked_range<int>& range) {
        stats local_stat;
        for (int i = range.begin(); i != range.end(); i++) {
            local_stat.add(do_work(num_ops / num_threads));
        }
        stat.add(local_stat);
    });
    auto end = std::chrono::high_resolution_clock::now();

    double total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    double add_throughput = stat.num_adds / total_time;
    double remove_throughput = stat.num_removes / total_time;
    double contains_throughput = stat.num_contains / total_time;
    double overall_throughput = (stat.num_adds + stat.num_removes + stat.num_contains) / total_time;

    cout << "Bench:\n\t"
        << set.size() << " size,\n\t"
        << initial_size + stat.num_adds - stat.num_removes << " expected size,\n\t"
        << set._num_resize << " num resizes,\n\t"
        << stat.num_contains << " contains (" << contains_throughput << " ops/sec),\n\t"
        << stat.num_adds << " adds (" << add_throughput << " ops/sec),\n\t"
        << stat.num_removes << " removes (" << remove_throughput << " ops/sec),\n\t"
        << stat.add_calls << " add calls,\n\t"
        << stat.rem_calls << " rem calls,\n\t"
        << "Overall throughput: " << overall_throughput << " ops/sec" << endl;

    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int main() {
    int num_ops = 10000000, num_elms = 1000000, max_val = 1000000, capacity = 160000000, num_threads = 16;
    long seq_dur = seq_test(num_ops, num_elms, max_val, capacity);
    long con_dur = con_test(num_ops, num_elms, max_val, capacity, num_threads);
    long tran_dur = tran_test(num_ops, num_elms, max_val, capacity, num_threads);
    cout << "\n\n seq: " << seq_dur << "\n con: " << con_dur << "\n tran: " << tran_dur << endl;
    return 0;
}
