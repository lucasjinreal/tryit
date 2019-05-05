/**
 * 
 * 
 * Test for yaml installation
 * 
 * */
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>


using namespace std;


int main(int argc, char **argv)
{

    cout << "loading from file: " << argv[1] << endl;
    YAML::Node config = YAML::LoadFile(argv[1]);

    const std::string username = config["username"].as<std::string>();
    const std::string password = config["password"].as<std::string>();

    cout << "name: " << username << " password: " << password << endl;

    // change the value


    std::ofstream fout("config2.yaml");
    fout << config;
}
