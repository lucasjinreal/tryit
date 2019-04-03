#include "lib/hellogreet.h"
#include <string>

std::string get_greet(const std::string& who) {
  return "Hello " + who;
}