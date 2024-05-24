#include <iostream>

int ext_ef, ext_omp, ext_interq_multithread, ext_batch_size, ext_sub_graph_num;
char* ext_dataset;
std::string ext_hnsw_index_prefix;
std::string ext_res_path;

void hnsw_search();

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cout << "Usage: ./main <dataset> <hnsw_index_prefix> <sub_graph_num> <ef> <omp> <interq_multithread> <batch_size> <res_path>" << std::endl;
        return 1;
    }

    ext_dataset = argv[1];
    ext_hnsw_index_prefix = argv[2];
    ext_sub_graph_num = atoi(argv[3]);
    ext_ef = atoi(argv[4]);
    ext_omp = atoi(argv[5]);
    ext_interq_multithread = atoi(argv[6]);
    ext_batch_size = atoi(argv[7]);
    ext_res_path = argv[8];

    hnsw_search();
    
    return 0;
}
