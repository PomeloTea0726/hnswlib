#include <iostream>

int ext_M, ext_ef_construction, ext_ef, ext_k, ext_omp, ext_interq_multithread, ext_batch_size;
std::string ext_dataset_path;

void sift_test1M_parallel();
void sift_test1M();

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cout << "Usage: ./main <M> <ef_construction> <ef> <k> <omp> <interq_multithread> <batch_size> <dataset_path>" << std::endl;
        return 1;
    }

    ext_M = atoi(argv[1]);
    ext_ef_construction = atoi(argv[2]);
    ext_ef = atoi(argv[3]);
    ext_k = atoi(argv[4]);
    ext_omp = atoi(argv[5]);
    ext_interq_multithread = atoi(argv[6]);
    ext_batch_size = atoi(argv[7]);
    ext_dataset_path = argv[8];

    sift_test1M();
    
    return 0;
}
