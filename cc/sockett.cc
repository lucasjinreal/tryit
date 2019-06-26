#include <boost/asio.hpp>
#include <iostream>
#include "glog/logging.h"

using namespace google;

std::string make_daytime_string() {
  using namespace std;  // For time_t, time and ctime;
  time_t now = time(0);
  return ctime(&now);
}

int main(void) {
  try {
    std::cout << "server start." << std::endl;
    boost::asio::io_service ios;
    boost::asio::ip::tcp::endpoint endpotion(boost::asio::ip::tcp::v4(), 13695);
    boost::asio::ip::tcp::acceptor acceptor(ios, endpotion);
    std::cout << "addr: " << acceptor.local_endpoint().address() << std::endl;
    std::cout << "port: " << acceptor.local_endpoint().port() << std::endl;

    while (true) {
      boost::asio::ip::tcp::socket socket(ios);
      acceptor.accept(socket);
      LOG(INFO) << "got a connection: " << socket.remote_endpoint().address();
      LOG(INFO) << "start sending data continuely.";

      while (socket.is_open()) {
        std::string message = make_daytime_string();
        message = "a message from socket " + message;
        LOG(INFO) << "write message: " << message;
        socket.write_some(boost::asio::buffer(message));
        //   // 阻塞接收客户端发来的数据
        //   socket.read_some(boost::asio::buffer(msg));
        //   // 打印客户端发来的数据
        //   std::cout << "client reply: " << msg.c_str() << std::endl;
      }
    }

  } catch (...) {
    std::cout << "server exceptional." << std::endl;
  }
  std::cout << "server end." << std::endl;
  getchar();
  return 0;
}