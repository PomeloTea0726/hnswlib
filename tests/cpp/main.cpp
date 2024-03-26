#include <iostream>

int ext_M, ext_ef_construction, ext_ef, ext_k, ext_interq_multithread, ext_batch_size;
std::string ext_dataset_path;

void sift_test1M();
int main(int argc, char** argv) {

    ext_M = 16;
    ext_ef_construction = 200;
    ext_ef = 200;
    ext_k = 10;
    ext_interq_multithread = 1;
    ext_batch_size = 100;
    ext_dataset_path = argv[1];

    sift_test1M();

    return 0;
}
