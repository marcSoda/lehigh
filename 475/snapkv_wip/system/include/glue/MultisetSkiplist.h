#include <SiKV.h>
#include <random>

#pragma once

namespace sikv {
namespace glue {

template<typename T, typename Comp, int MAX_HEIGHT=10>
class MultisetSkiplist {
    
    struct Node_t {
        Node_t() : next{} {}

        explicit Node_t(T&& val_) : val(val_), next() {}

        T val;
        Node_t* next[MAX_HEIGHT];
    };
public:
    MultisetSkiplist() : head(new Node_t()), tail(new Node_t()) {
        for(int i = 0; i < MAX_HEIGHT; i++) {
            head->next[i] = tail;
        } 
    }

    MultisetSkiplist(MultisetSkiplist&& other) noexcept : head(other.head), tail(other.tail) {
        other.head = nullptr;
        other.tail = nullptr;
    }

    /// Other should not be used after this
    MultisetSkiplist& operator=(MultisetSkiplist&& other) noexcept {
        head = other.head;
        other.head = nullptr;
        tail = other.tail;
        other.tail = nullptr;
        return *this;
    }

    ~MultisetSkiplist() {
        auto n = head;
        while(n != tail) {
            auto tmp = n;
            n = n->next[0];
            delete tmp;
        }
        delete n;
    }

    void insert(T&& x) {

        thread_local std::default_random_engine gen;

        auto n = head;
        Node_t* past[MAX_HEIGHT] = {};

        for(int level = MAX_HEIGHT-1; level >= 0; level--) {
            while(true) {
                if(n->next[level] == tail || Comp{}(x, n->next[level]->val)) {
                    // x < tail or x < next value on level
                    past[level] = n;
                    break;
                }
                // x >= next value on level
                n = n->next[level];
            }
        }
        auto toInsert = new Node_t(std::move(x));

        int toplevel = std::uniform_int_distribution<int>{0,MAX_HEIGHT-1}(gen);

        for(int level = toplevel; level>= 0; level--) {
            n = past[level]->next[level];
            // past[level] is in front of toInsert and n is behind to insert
            past[level]->next[level] = toInsert;
            toInsert->next[level] = n;
        } 
    }

    class Iterator {
        Node_t* n;
        explicit Iterator(Node_t* n_) : n(n_) {}
    public:
        bool operator==(const Iterator& other) {
            return n == other.n;
        }

        bool operator!=(const Iterator& other) {
            return n != other.n;
        }

        Iterator& operator++() {
            n = n->next[0];
            return *this;
        }

        T& operator*() {
            return n->val;
        }  
        
        T* operator->() {
            return &n->val;
        } 
        
    };

    Iterator begin() {
        return Iterator(head);
    }

    Iterator end() {
        return Iterator(tail);
    }

    bool empty() {
        return head->next[0] == tail;
    }

private:
    Node_t* head;
    Node_t* tail;
};

} // namespace glue
} // namespace sikv
