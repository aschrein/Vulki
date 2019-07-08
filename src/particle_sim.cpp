#include "../include/particle_sim.hpp"
#include <sparsehash/dense_hash_set>

static void dummy() {
  google::dense_hash_set<uint> set;
  set.insert(0);
  set.insert(1);
}