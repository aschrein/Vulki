#pragma once
#include "VulkanMemoryAllocator/src/vk_mem_alloc.h"
#include "error_handling.hpp"
#include <memory>
#include <vulkan/vulkan.hpp>

struct Slot {
  bool alive;
  u32 id;
  void disable() { alive = false; }
  void set_alive() { alive = true; }
  bool is_alive() { return alive; }
  u32 get_id() { return id; }
  void set_id(u32 _id) { id = _id; }
};

struct VmaBuffer : public Slot {
  RAW_MOVABLE(VmaBuffer)
  VmaAllocator allocator;
  vk::Buffer buffer;
  VmaAllocation allocation;
  vk::BufferCreateInfo create_info;
  vk::AccessFlags access_flags;
  void barrier(vk::CommandBuffer &cmd, u32 queue_family_id,
               vk::AccessFlags new_access_flags) {
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                        vk::PipelineStageFlagBits::eAllCommands,
                        vk::DependencyFlagBits::eByRegion, {},
                        {vk::BufferMemoryBarrier()
                             .setSize(create_info.size)
                             .setBuffer(buffer)
                             .setOffset(0)
                             .setSrcAccessMask(access_flags)
                             .setDstAccessMask(new_access_flags)
                             .setDstQueueFamilyIndex(queue_family_id)
                             .setSrcQueueFamilyIndex(queue_family_id)},
                        {});
    access_flags = new_access_flags;
  }
  void *map() {
    void *data = nullptr;
    vmaMapMemory(allocator, allocation, &data);
    return data;
  }
  void unmap() { vmaUnmapMemory(allocator, allocation); }
  void destroy() {
    if (buffer) {
      vmaDestroyBuffer(allocator, buffer, allocation);
      memset(this, 0, sizeof(*this));
    }
  }
  ~VmaBuffer() { destroy(); }
};

struct VmaImage : public Slot {
  RAW_MOVABLE(VmaImage)
  VmaAllocator allocator;
  vk::Image image;
  vk::ImageCreateInfo create_info;
  VmaAllocation allocation;
  vk::ImageLayout layout;
  vk::AccessFlags access_flags;
  vk::ImageAspectFlags aspect;
  void barrier(vk::CommandBuffer &cmd, u32 queue_family_id,
               vk::ImageLayout new_layout, vk::AccessFlags new_access_flags, bool force = false) {
    if (layout == new_layout && new_access_flags == access_flags && !force)
      return;
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eAllCommands,
        vk::DependencyFlagBits::eByRegion, {}, {},
        {vk::ImageMemoryBarrier()
             .setSrcAccessMask(access_flags)
             .setDstAccessMask(new_access_flags)
             .setOldLayout(layout)
             .setNewLayout(new_layout)
             .setSrcQueueFamilyIndex(queue_family_id)
             .setDstQueueFamilyIndex(queue_family_id)
             .setImage(image)
             .setSubresourceRange(vk::ImageSubresourceRange()
                                      .setLayerCount(create_info.arrayLayers)
                                      .setLevelCount(create_info.mipLevels)
                                      .setAspectMask(aspect))});
    this->access_flags = new_access_flags;
    this->layout = new_layout;
  }
  vk::UniqueImageView create_view(vk::Device device, u32 base_level, u32 levels,
                                  u32 base_layer, u32 layers) {
    vk::ImageViewType view_type;
    switch (create_info.imageType) {
    case vk::ImageType::e2D:
      view_type = vk::ImageViewType::e2D;
      break;
    case vk::ImageType::e3D:
      view_type = vk::ImageViewType::e3D;
      break;
    default:
      ASSERT_PANIC(false && "Unsupported image type");
    }
    return device.createImageViewUnique(vk::ImageViewCreateInfo(
        vk::ImageViewCreateFlags(), image, view_type, create_info.format,
        vk::ComponentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
                             vk::ComponentSwizzle::eB,
                             vk::ComponentSwizzle::eA),
        vk::ImageSubresourceRange()
            .setBaseMipLevel(base_level)
            .setBaseArrayLayer(base_layer)
            .setLayerCount(layers)
            .setLevelCount(levels)
            .setAspectMask(aspect)));
  }
  void destroy() {
    if (image) {
      vmaDestroyImage(allocator, image, allocation);
      memset(this, 0, sizeof(*this));
    }
  }
  ~VmaImage() { destroy(); }
  void *map() {
    void *data = nullptr;
    vmaMapMemory(allocator, allocation, &data);
    return data;
  }
  void unmap() { vmaUnmapMemory(allocator, allocation); }
};

struct Alloc_State {
  VmaAllocator allocator;
  vk::Device device;
  static std::unique_ptr<Alloc_State> create(vk::Device device,
                                             vk::PhysicalDevice phdevice) {
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = phdevice;
    allocatorInfo.device = device;
    VmaAllocator allocator;
    vmaCreateAllocator(&allocatorInfo, &allocator);

    return std::unique_ptr<Alloc_State>(new Alloc_State{allocator, device});
  }
  VmaBuffer allocate_buffer(vk::BufferCreateInfo const &create_info,
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
    out.create_info = create_info;
    return out;
  }
  VmaImage allocate_image(
      vk::ImageCreateInfo const &create_info, VmaMemoryUsage usage,
      vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor) {
    ASSERT_PANIC(create_info.extent.width > 1u &&
                 create_info.extent.height > 1u);
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = usage;
    VkImage image;
    VmaAllocation allocation;
    VkImageCreateInfo tmp_create_info = create_info;
    VmaAllocationInfo AllocationInfo;
    vmaCreateImage(allocator, &tmp_create_info, &allocInfo, &image, &allocation,
                   &AllocationInfo);
    VmaImage out;
    out.allocator = allocator;
    out.image = image;
    out.allocation = allocation;
    out.layout = create_info.initialLayout;
    out.access_flags = vk::AccessFlagBits::eMemoryRead;
    out.create_info = create_info;
    out.create_info.setPNext(nullptr);
    out.create_info.setPQueueFamilyIndices(nullptr);
    out.aspect = aspect;

    return out;
  }
  ~Alloc_State() { vmaDestroyAllocator(allocator); }
};
