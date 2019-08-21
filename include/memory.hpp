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
  vk::UniqueImageView view;
  vk::ImageCreateInfo create_info;
  VmaAllocation allocation;
  vk::ImageLayout layout;
  vk::AccessFlags access_flags;
  vk::ImageAspectFlags aspect;
  void barrier(vk::CommandBuffer &cmd, u32 queue_family_id,
               vk::ImageLayout new_layout, vk::AccessFlags new_access_flags) {
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
  ~VmaImage() {
    if (image) {
      view.reset(vk::ImageView(VkImageView(0u)));
      vmaDestroyImage(allocator, image, allocation);
    }
  }
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

    return out;
  }
  VmaImage allocate_image(
      vk::ImageCreateInfo const &create_info, VmaMemoryUsage usage,
      vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor) {
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
    out.view = device.createImageViewUnique(vk::ImageViewCreateInfo(
        vk::ImageViewCreateFlags(), out.image, view_type, create_info.format,
        vk::ComponentMapping(vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
                             vk::ComponentSwizzle::eB,
                             vk::ComponentSwizzle::eA),
        vk::ImageSubresourceRange()
            .setLayerCount(create_info.arrayLayers)
            .setLevelCount(create_info.mipLevels)
            .setAspectMask(aspect)));
    return out;
  }
  ~Alloc_State() { vmaDestroyAllocator(allocator); }
};
