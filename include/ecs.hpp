#pragma once
#include "error_handling.hpp"
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
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

struct Component_Info {
  Entity_ID owner;
  bool dead = true;
};

template <typename T> struct Component_Base : public Component_Info {
  static u32 ID;
  static char const *NAME;
  static std::vector<T> &table() {
    static std::vector<T> _table;
    return _table;
  }
};

class Entity {
private:
  // ECS methods
  static void _init() {
    static bool initialized = false;
    if (initialized)
      return;
    initialized = true;
    _get_component_mng_table().set_empty_key(UINT32_MAX);
    // create a null entity
    create_entity();
  }
  static u32 _allocate_type_id() {
    static u32 counter = 0;
    return counter++;
  }
  static google::dense_hash_map<u32, Component_Mng> &
  _get_component_mng_table() {
    static google::dense_hash_map<u32, Component_Mng> table;
    _init();
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

  static std::vector<std::function<void()>> &get_defer_table() {
    static std::vector<std::function<void()>> table;
    return table;
  }

public:
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
    u32 index = get_entity_table().size();
    get_entity_table().push_back(Entity{});
    get_entity_table()[index].refcnt = 1;
    get_entity_table()[index].id = {0u, index};
    return {0u, index};
  };
  static Entity *get_entity_weak(Entity_ID id) {
    if (id.index == 0)
      return nullptr;
    return &get_entity_table()[id.index];
  }
  static void defer_function(std::function<void()> func) {
    get_defer_table().push_back(func);
  }
  static void flush() {
    for (auto func : get_defer_table()) {
      func();
    }
    get_defer_table().clear();
  }
  // Methods
  void acquire() { refcnt++; }
  void release() {
    refcnt--;
    if (refcnt == 0) {
      for (auto cid : components) {
        get_component_ptr<Component_Info>(id, cid)->dead = true;
      }
      components.clear();
    }
  }
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

struct Entity_StrongPtr {
  RAW_MOVABLE(Entity_StrongPtr);
  Entity_StrongPtr(Entity_ID eid) : eid(eid) {}
  Entity *operator->() { return Entity::get_entity_weak(eid); }
  ~Entity_StrongPtr() {
    if (eid.index) {
      auto e = Entity::get_entity_weak(eid);
      e->release();
    }
  }

public:
  Entity_ID eid;
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
  template <> char const *Component_Base<CLASS>::NAME = #CLASS;

struct C_Transform : public Component_Base<C_Transform> {
  vec3 scale;
  vec3 offset;
  quat rotation;
  mat4 get_matrix() {
    return glm::translate(offset) *
           rotation.operator glm::mat<4, 4, glm::f32, glm::packed_highp>() *
           glm::scale(scale);
  }
};

struct C_Name : public Component_Base<C_Name> {
  std::string name;
};