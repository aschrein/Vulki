#pragma once
#include "VulkanMemoryAllocator/src/vk_mem_alloc.h"
#include "error_handling.hpp"
#include <memory>
#include <vulkan/vulkan.hpp>

struct VmaBuffer {
  RAW_MOVABLE(VmaBuffer)
  VmaAllocator allocator;
  vk::Buffer buffer;
  VmaAllocation allocation;
  void *map() {
    void *data = nullptr;
    vmaMapMemory(allocator, allocation, &data);
    return data;
  }
  void unmap() { vmaUnmapMemory(allocator, allocation); }
  ~VmaBuffer() {
    if (buffer)
      vmaDestroyBuffer(allocator, buffer, allocation);
  }
};

struct VmaImage {
  RAW_MOVABLE(VmaImage)
  VmaAllocator allocator;
  vk::Image image;
  VmaAllocation allocation;
  ~VmaImage() {
    if (image)
      vmaDestroyImage(allocator, image, allocation);
  }
};

struct Alloc_State {
  VmaAllocator allocator;

  static std::unique_ptr<Alloc_State> create(vk::Device device,
                                             vk::PhysicalDevice phdevice) {
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = phdevice;
    allocatorInfo.device = device;
    VmaAllocator allocator;
    vmaCreateAllocator(&allocatorInfo, &allocator);

    return std::unique_ptr<Alloc_State>(new Alloc_State{allocator});
  }
  VmaBuffer allocate_buffer(vk::BufferCreateInfo create_info,
                            VmaMemoryUsage usage) {
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = usage;
    VkBuffer buffer;
    VmaAllocation allocation;
    VkBufferCreateInfo tmp_create_info = create_info;
    vmaCreateBuffer(allocator, &tmp_create_info, &allocInfo, &buffer,
                    &allocation, nullptr);
    VmaBuffer out;
    out.allocator = allocator;
    out.buffer = buffer;
    out.allocation = allocation;

    return out;
  }
  VmaImage allocate_image(vk::ImageCreateInfo create_info,
                          VmaMemoryUsage usage) {
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = usage;
    VkImage image;
    VmaAllocation allocation;
    VkImageCreateInfo tmp_create_info = create_info;
    vmaCreateImage(allocator, &tmp_create_info, &allocInfo, &image, &allocation,
                   nullptr);
    VmaImage out;
    out.allocator = allocator;
    out.image = image;
    out.allocation = allocation;

    return out;
  }
  ~Alloc_State() { vmaDestroyAllocator(allocator); }
};