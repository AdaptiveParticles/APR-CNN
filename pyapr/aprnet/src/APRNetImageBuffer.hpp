//
// Created by joel on 02.08.21.
//

#ifndef APRNET_IMAGEBUFFER_HPP
#define APRNET_IMAGEBUFFER_HPP

template<typename T>
class APRNetImageBuffer {
public:
    int y_num;
    int x_num;
    int z_num;
    std::vector<T> mesh;

    APRNetImageBuffer() = default;

    APRNetImageBuffer(int aSizeOfY, int aSizeOfX, int aSizeOfZ) {
        resize(aSizeOfY, aSizeOfX, aSizeOfZ);
    }

    APRNetImageBuffer(int aSizeOfY, int aSizeOfX, int aSizeOfZ, T val) {
        resize(aSizeOfY, aSizeOfX, aSizeOfZ, val);
    }

    void resize(int aSizeOfY, int aSizeOfX, int aSizeOfZ) {
        y_num = aSizeOfY;
        x_num = aSizeOfX;
        z_num = aSizeOfZ;
        mesh.resize((size_t) y_num * x_num * z_num);
    }

    void resize(int aSizeOfY, int aSizeOfX, int aSizeOfZ, T val) {
        y_num = aSizeOfY;
        x_num = aSizeOfX;
        z_num = aSizeOfZ;
        mesh.resize((size_t) y_num * x_num * z_num, val);
    }

    void fill(const T val) {
        std::fill(mesh.begin(), mesh.end(), val);
    }

    inline size_t offset_z(const int z) {
        return (size_t) z * x_num * y_num;
    }

    inline size_t offset(const int z, const int x) {
        return (size_t) z * x_num * y_num + x * y_num;
    }

    inline T& at(int y, int x, int z) {
        return mesh[offset(z, x) + y];
    }

    inline const T& at(int y, int x, int z) const {
        return mesh[offset(z, x) + y];
    }
};

#endif //APRNET_IMAGEBUFFER_HPP
