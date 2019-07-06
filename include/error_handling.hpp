#pragma once
// #include <stdio.h>
// #include <stdlib.h>
#include <filesystem>
#include <iostream>
#include <string.h>

static void panic_impl(char const *msg, int line) {
  std::cerr << "current wd:" << std::filesystem::current_path() << "\n";
  std::cerr << "panic:" << msg << " at line " << line << "\n";
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

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))

// memset{0} - is a valid state
#define RAW_MOVABLE(CLASS)\
CLASS() = default;\
CLASS(CLASS const &) = delete;\
CLASS(CLASS &&that) {\
*this = std::move(that);\
}\
CLASS &operator=(CLASS &&that) {\
memcpy(this, &that, sizeof(CLASS));\
memset(&that, 0, sizeof(CLASS));\
return *this;\
}\
CLASS &operator=(CLASS const &) = delete;
