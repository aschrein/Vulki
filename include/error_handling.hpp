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

#define ito(N) for (u32 i = 0; i < N; i++)
#define jto(N) for (u32 j = 0; j < N; j++)
#define kto(N) for (u32 k = 0; k < N; k++)

template <typename T, size_t N> constexpr size_t __ARRAY_SIZE(T (&)[N]) {
  return N;
}

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

template <typename T> struct ExitScope {
  T lambda;
  ExitScope(T lambda) : lambda(lambda) {}
  ~ExitScope() { lambda(); }
  ExitScope(const ExitScope &);

private:
  ExitScope &operator=(const ExitScope &);
};

class ExitScopeHelp {
public:
  template <typename T> ExitScope<T> operator+(T t) { return t; }
};

#define CONCAT_INTERNAL(x,y) x##y
#define CONCAT(x,y) CONCAT_INTERNAL(x,y)
#define defer const auto &CONCAT(defer__, __LINE__) = ExitScopeHelp() + [&]()

template <typename T> struct Onetime {
  Onetime(T lambda) { lambda(); }
  Onetime(const Onetime &) = delete;
  Onetime &operator=(Onetime const &) = delete;

private:
};

class OnetimeHelp {
public:
  template <typename T> Onetime<T> operator+(T t) { return t; }
};

#define onetime static Onetime CONCAT(defer__, __LINE__) = OnetimeHelp() + []()
