#include <iostream>

int ext_ef, ext_omp, ext_interq_multithread, ext_batch_size;
char* ext_dataset;
std::string ext_hnsw_index_file;

void hnsw_search();

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cout << "Usage: ./main <dataset> <hnsw_index_file> <ef> <omp> <interq_multithread> <batch_size>" << std::endl;
        return 1;
    }

    ext_dataset = argv[1];
    ext_hnsw_index_file = argv[2];
    ext_ef = atoi(argv[3]);
    ext_omp = atoi(argv[4]);
    ext_interq_multithread = atoi(argv[5]);
    ext_batch_size = atoi(argv[6]);

    hnsw_search();
    
    return 0;
}
