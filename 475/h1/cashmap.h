#include <chrono>
#include <unordered_map>
#include <mutex>
#include <random>
#include <thread>
#include <future>
#include <queue>

#include "config_t.h"

using namespace std;

struct account {
    int id;
    float bal;
    mutex amtx;
};

class cashmap {
    private:
        unordered_map<int, account> accs_;
        mutex queue_mtx_;
        queue<promise<double>> bal_promises_;
        atomic<int> bal_ctr_ = { 0 };
        atomic<int> dep_ctr_ = { 0 };
        atomic<int> prom_ctr_ = { 0 };
        config_t cfg_;
    public:
        cashmap(config_t cfg): cfg_(cfg) { }
        ~cashmap() { }

        // Insert if and only if the key is not currently present in
        // the map.  Returns true on success, false if the key was
        // already present. Not threadsafe
        bool insert(int id, double bal) {
            if (accs_.count(id) != 0) return false;
            accs_[id].bal = bal;
            return true;
        }

        // grab a promise from the queue and process it
        void process_balance() {
            bal_ctr_++;
            while (dep_ctr_.load() != 0) { }
            unique_lock<mutex> queue_lock(queue_mtx_);
            if (bal_promises_.empty()) return;
            prom_ctr_--;
            promise<double> prom = std::move(bal_promises_.front());
            bal_promises_.pop();
            queue_lock.unlock();
            int sum = 0;
            for (auto &acc : accs_) {
                sum += acc.second.bal;
            }
            prom.set_value(sum);
            // if (sum != 10000000.) cout << ">>>>>>>>>FLAG<<<<<<<<<<<<" << sum << endl;
            bal_ctr_--;
        }

        // adds a promise to the queue and returns a fututre
        // processes a balance if need be
        future<double> balance() {
            promise<double> prom;
            future<double> fut = prom.get_future();
            unique_lock<mutex> queue_lock(queue_mtx_);
            bal_promises_.push(std::move(prom));
            prom_ctr_++;
            queue_lock.unlock();
            if (prom_ctr_ >= cfg_.threads) {
                process_balance();
            }
            return fut;
        }

        // takes a random amount from a random account and places it into a different random account
        // processes multiple balances if need be
        void deposit() {
            while (prom_ctr_ >= cfg_.threads) {
                process_balance();
            }
            while (bal_ctr_.load() != 0) { }
            dep_ctr_++;

            random_device rd;
            mt19937_64 gen(rd());
            uniform_int_distribution<> acc_dis(0, accs_.size() - 1);
            int from_acc = acc_dis(gen);
            int to_acc;
            while (1) {
                to_acc = acc_dis(gen);
                if (to_acc != from_acc) break;
            }
            unique_lock<mutex> from_lock(accs_[from_acc].amtx, defer_lock);
            unique_lock<mutex> to_lock(accs_[to_acc].amtx, defer_lock);
            while (1) {
                from_lock.lock();
                if (to_lock.try_lock()) break;
                from_lock.unlock();
            }
            uniform_int_distribution<> amt_dis(0, accs_[from_acc].bal);
            int amt = amt_dis(gen);
            accs_[from_acc].bal -= amt;
            from_lock.unlock();
            accs_[to_acc].bal += amt;
            to_lock.unlock();
            dep_ctr_--;
        }

        // returns false if there is an account(s) in bals_. true otherwise
        // evenly populates num_accounts with account numbers from 0 to num_accounts - 1
        // evenly distributes total_balance to each account
        bool populate(int num_accounts, double total_balance) {
            if (!accs_.empty())
                return false;
            double split_balance = total_balance / (double)num_accounts;
            for (int i = 0; i < num_accounts; i++)
                accs_[i].bal = split_balance;
            return true;
        }
};
