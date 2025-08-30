#include "in_rt.hpp"
#include <cstdio>
#include <thread>
#include <chrono>

uint32_t inrt::g_cap_mask = 0;

int inrt::fs_write(const char* path, std::string_view data){
  if(!gate_enabled(G_FILE)) return -13; // EPERM
  FILE* f = std::fopen(path, "ab");
  if(!f) return -1;
  std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
  return 0;
}

int inrt::net_send(const char*, int, std::string_view){
  if(!gate_enabled(G_NET)) return -13;
  // MVP: pretend success (or hook a stub)
  return 0;
}

int inrt::run_loop(int (*step_fn)(int64_t), dur tick_ns){
  using namespace std::chrono;
  while(true){
    auto t0 = high_resolution_clock::now();
    int rc = step_fn((int64_t)tick_ns);
    if(rc) return rc;
    auto t1 = high_resolution_clock::now();
    auto spent = duration_cast<nanoseconds>(t1 - t0).count();
    auto sleep_ns = tick_ns - spent;
    if(sleep_ns > 0) std::this_thread::sleep_for(nanoseconds(sleep_ns));
  }
}
