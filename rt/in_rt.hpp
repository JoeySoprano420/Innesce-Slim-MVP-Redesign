#pragma once
#include <cstdint>
#include <string_view>

namespace inrt {
  enum Gate : uint32_t { G_FILE=1u<<0, G_NET=1u<<1 };
  extern uint32_t g_cap_mask;

  inline bool gate_enabled(Gate g){ return (g_cap_mask & g)!=0; }

  // durations (store as ns)
  using dur = int64_t;
  inline dur ms(int64_t v){ return v*1'000'000; }
  inline dur s (int64_t v){ return v*1'000'000'000; }

  // gated I/O stubs (MVP)
  int fs_write(const char* path, std::string_view data); // checks G_FILE
  int net_send (const char* host, int port, std::string_view data); // checks G_NET

  // packetized loop
  int run_loop(int (*step_fn)(int64_t), dur tick_ns);
}
