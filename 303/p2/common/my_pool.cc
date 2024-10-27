#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <queue>
#include <thread>
#include <unistd.h>

#include "pool.h"

using namespace std;

class my_pool : public thread_pool {
public: 

  std::function<void()> shutdown_func;
  std::function<bool(int)> handler_func;
  std::atomic<bool> cont = true;
  std::queue<int> my_queue;
  std::condition_variable cv;
  std::mutex m;
  std::vector<std::thread> thread_pool;


  /// construct a thread pool by providing a size and the function to run on
  /// each element that arrives in the queue
  ///
  /// @param size    The number of threads in the pool
  /// @param handler The code to run whenever something arrives in the pool
  my_pool(int size, function<bool(int)> handler) {
    handler_func = handler;
    auto workload = [&]() {
      while (check_active()) {
        std::unique_lock<std::mutex> ul(m);
        if (my_queue.empty()) cv.wait(ul);
        else {
          int socket_id = my_queue.front();
          my_queue.pop();
          ul.unlock();
          if(handler_func(socket_id)){
            cont = false;
            shutdown_func();
          }
          close(socket_id);
        }
      }
    };
    for (int i = 0; i < size; i++){
      thread_pool.push_back(std::thread(workload));
    }
  }


  /// destruct a thread pool
  virtual ~my_pool() = default;

  /// Allow a user of the pool to provide some code to run when the pool decides
  /// it needs to shut down.
  ///
  /// @param func The code that should be run when the pool shuts down
  virtual void set_shutdown_handler(function<void()> func) {
    shutdown_func = func;
  }

  /// Allow a user of the pool to see if the pool has been shut down
  virtual bool check_active() {
    return cont;
  }

  /// Shutting down the pool can take some time.  await_shutdown() lets a user
  /// of the pool wait until the threads are all done servicing clients.
  virtual void await_shutdown() {
    std::unique_lock<std::mutex> ul(m);
    if(!my_queue.empty()) cv.wait(ul);
    else {
      for (auto &t: thread_pool){
        t.join();
      }
    }
  }

  /// When a new connection arrives at the server, it calls this to pass the
  /// connection to the pool for processing.
  ///
  /// @param sd The socket descriptor for the new connection
  virtual void service_connection(int sd) {
    std::unique_lock<std::mutex> ul(m);
    my_queue.push(sd);
    cv.notify_one();
    ul.unlock();
  }
};

/// Create a thread_pool object.
///
/// We use a factory pattern (with private constructor) to ensure that anyone
thread_pool *pool_factory(int size, function<bool(int)> handler) {
  return new my_pool(size, handler);
}
