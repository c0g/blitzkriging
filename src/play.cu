#include "blitzkriging.h"

using namespace blitzkriging;

int main() {
    Dummy<HostMatrix<float>> k{3};

    std::vector<HostMatrix<float>> m, x;
    HostMatrix<float> m0{3,1}, m1{2,1}, m2{4,1}, x0{4, 1}, x1{4, 1}, x2{4,1};
    m0 = 1,
           2,
           3;
    m1 = 1,
           2;
    m2 = 1,
           2,
           3,
           4;
    x0 = 1,2,3,4;
    x1 = 1,2,3,4;
    x2 = 4,5,6,7;
    m.push_back(m0);
    m.push_back(m1);
    m.push_back(m2);
    x.push_back(x0);
    x.push_back(x1);
    x.push_back(x2);

    k.setM(m);
    k.setX(x);
    std::cout << "M\n";
    std::cout << m0 << "\n" << m1 << "\n" << m2 << "\n";
    std::cout << "X\n";
    std::cout << x0 << "\n" << x1 << "\n" << x2 << "\n";

    std::cout << "MM:\n";
    std::cout << k.sqdistMM() << std::endl;
    std::cout << "MX:\n";
    std::cout << k.sqdistMX() << std::endl;
    return 0;
}

