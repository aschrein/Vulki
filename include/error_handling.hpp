#pragma once
// #include <stdio.h>
// #include <stdlib.h>
#include <filesystem>
#include <iostream>
#include <string.h>

#ifdef __GNUC__
#include <signal.h>
#endif

static void panic_impl(char const *msg, int line) {
  std::cerr << "current wd:" << std::filesystem::current_path() << "\n";
  std::cerr << "panic:" << msg << " at line " << line << "\n";
#ifdef __GNUC__
  raise(SIGTRAP);
#endif
  std::exit(1);
}

#define panic(msg) panic_impl(msg, __LINE__)
#define ASSERT_PANIC(expr)                                                     \
  if (!(expr)) {                                                               \
    panic(#expr);                                                              \
  }

static void error_callback(int error, const char *description) {
  std::cerr << "Error: " << description << "\n";
}

// Poor man's rust
using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using f32 = float;

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))

// No pointer to the object should persist
// @Cleanup: Do something better
#define RAW_MOVABLE(CLASS)                                                     \
  CLASS() = default;                                                           \
  CLASS(CLASS const &) = delete;                                               \
  CLASS(CLASS &&that) { *this = std::move(that); }                             \
  CLASS &operator=(CLASS &&that) {                                             \
    this->~CLASS();                                                            \
    memcpy(this, &that, sizeof(CLASS));                                        \
    new (&that) CLASS;                                                         \
    return *this;                                                              \
  }                                                                            \
  CLASS &operator=(CLASS const &) = delete;
