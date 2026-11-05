#include "platform/linux/application.hpp"

// 将程序参数交给 Linux 应用入口.
int main(int argc, char** argv) {
    return voice_control::RunApplication(argc, argv);
}
