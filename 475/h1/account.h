#ifndef ACCOUNT_H_
#define ACCOUNT_H_

#include <mutex>

using namespace std;

class account {
    private:
        int id_;
        float bal_;
        mutex mtx_;
    public:
        account(int id, float bal) {
            id_ = id;
            bal_ = bal;
        }
        ~account() {}
}



#endif // ACCOUNT_H_
