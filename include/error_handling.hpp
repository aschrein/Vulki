#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <filesystem>

static void panic_impl(char const *msg, int line) {
  fprintf(stderr, "current wd:%s\n", std::filesystem::current_path().c_str());
  fprintf(stderr, "panic: %s at line %i\n", msg, line);
  exit(1);  
}

#define panic(msg) panic_impl(msg, __LINE__)
#define ASSERT_PANIC(expr) if (!(expr)) { panic(#expr); }

static void error_callback(int error, const char *description) {
  fprintf(stderr, "Error: %s\n", description);
}

#define ARRAY_SIZE(x) (sizeof(x)/sizeof(x[0]))