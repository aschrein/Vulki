#pragma once
#include <iostream>

#include "dir_monitor/include/dir_monitor/dir_monitor.hpp"
#include <boost/thread.hpp>


static void dir_event_handler(boost::asio::dir_monitor &dm,
                              std::atomic<bool> &updated,
                              const boost::system::error_code &ec,
                              const boost::asio::dir_monitor_event &ev) {
  if (ev.type == boost::asio::dir_monitor_event::event_type::modified)
    updated = true;
  dm.async_monitor([&](const boost::system::error_code &ec,
                       const boost::asio::dir_monitor_event &ev) {
    dir_event_handler(dm, updated, ec, ev);
  });
}

struct Simple_Monitor {
  boost::asio::io_service io_service;
  boost::asio::dir_monitor dm;
  boost::thread dm_thread;
  std::atomic<bool> updated = false;
  Simple_Monitor(std::string const &folder) : dm(io_service) {

    dm.add_directory(folder);
    dm.async_monitor([&](const boost::system::error_code &ec,
                         const boost::asio::dir_monitor_event &ev) {
      dir_event_handler(dm, updated, ec, ev);
    });
    dm_thread = boost::thread(
        boost::bind(&boost::asio::io_service::run, boost::ref(io_service)));
  }
  bool is_updated() {
    bool expected = true;
    return updated.compare_exchange_weak(expected, false);
  }
};
