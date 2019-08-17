#pragma once
#include "error_handling.hpp"
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <sparsehash/dense_hash_map>
using namespace glm;

struct Component_ID {
  u32 type;
  u32 index;
};

struct Entity_ID {
  u8 generation : 8;
  u64 index : 56;
};

using Component_Factory = std::function<u32()>;
using Component_Getter = std::function<void *(u32)>;
using Component_Deleter = std::function<void(u32)>;

struct Component_Mng {
  Component_Factory factory;
  Component_Getter getter;
  Component_Deleter deleter;
};

class Entity {
private:
  // ECS methods
  static u32 _allocate_type_id() {
    static u32 counter = 0;
    return counter++;
  }
  static google::dense_hash_map<u32, Component_Mng> &
  _get_component_mng_table() {
    static google::dense_hash_map<u32, Component_Mng> table;

    return table;
  }
  Component_ID get_component(u32 type) {
    for (auto cid : components) {
      if (cid.type == type) {
        return cid;
      }
    }
    return Component_ID{0u, 0u};
  }
  template <typename T>
  static T *get_component_ptr(Entity_ID owner_id, Component_ID cid) {
    return reinterpret_cast<T *>(
        _get_component_mng_table()[cid.type].getter(cid.index));
  }

  static std::vector<Entity> &get_entity_table() {
    static std::vector<Entity> table;
    return table;
  }

public:
  static void _init() {
    _get_component_mng_table().set_empty_key(UINT32_MAX);
    // create a null entity
    create_entity();
  }
  static u32 register_component(char const *name, Component_Factory factory,
                                Component_Getter getter,
                                Component_Deleter deleter) {
    auto id = _allocate_type_id();
    _get_component_mng_table()[id] = Component_Mng{factory, getter, deleter};
    // Create a null component
    factory();
    return id;
  }
  static Entity_ID create_entity() {
    get_entity_table().push_back(Entity{});
    get_entity_table()[get_entity_table().size() - 1].refcnt = 1;
    return {0u, get_entity_table().size() - 1};
  };
  static Entity *get_entity_weak(Entity_ID id) {
    return &get_entity_table()[id.index];
  }

  void acquire() { refcnt++; }
  void release() { refcnt--; }
  void check_refcnt() {
    if (refcnt == 0) {
      ASSERT_PANIC(false && "release of zero refcount entity");
    }
  }

  template <typename T> T *get_component() {
    auto cid = get_component(T::ID);
    if (cid.index) {
      return get_component_ptr<T>(id, cid);
    }
    return nullptr;
  }

  template <typename T> T *get_or_create_component() {
    auto cid = get_component(T::ID);
    if (!cid.index) {
      cid.index = _get_component_mng_table()[T::ID].factory();
      cid.type = T::ID;
      components.push_back(cid);
      get_component_ptr<T>(id, cid)->owner = id;
      get_component_ptr<T>(id, cid)->dead = false;
    }
    return get_component_ptr<T>(id, cid);
  }

public:
  std::vector<Component_ID> components;
  Entity_ID id;
  u32 refcnt;
};

template <typename T> struct Component_Base {
  static u32 ID;
  static char const *NAME;
  static std::vector<T> &table() {
    static std::vector<T> _table;
    return _table;
  }
  Entity_ID owner;
  bool dead = true;
};

#define REG_COMPONENT(CLASS)                                                   \
  template <>                                                                  \
  u32 Component_Base<CLASS>::ID = Entity::register_component(                  \
      #CLASS,                                                                  \
      [] {                                                                     \
        CLASS::table().push_back(CLASS{});                                     \
        return CLASS::table().size() - 1;                                      \
      },                                                                       \
      [](u32 id) { return &CLASS::table()[id]; },                              \
      [](u32 id) { CLASS::table()[id].dead = true; });                         \
  template <> char const * Component_Base<CLASS>::NAME = #CLASS;

struct C_Transform : public Component_Base<C_Transform> {
  vec3 scale;
  vec3 offset;
  quat rotation;
};

struct C_Name : public Component_Base<C_Name> {
  std::string name;
};