#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/load_data.h"


#include <unordered_set>

using namespace std;
using namespace hnswlib;

extern int ext_ef, ext_omp, ext_interq_multithread, ext_batch_size;
extern char* ext_dataset;
extern std::string ext_hnsw_index_file;

double calculate_recall(std::vector<std::vector<unsigned>>& I, std::vector<std::vector<unsigned>>& gt, int k) {
    assert(I[0].size() >= k);
    assert(gt[0].size() >= k);
    int nq = I.size();
    int total_intersect = 0;
    for (int i = 0; i < nq; ++i) {
        std::vector<int> intersection;
        std::set_intersection(I[i].begin(), I[i].begin() + k, gt[i].begin(), gt[i].begin() + k, std::back_inserter(intersection));
        int n_intersect = intersection.size();
        total_intersect += n_intersect;
    }
    return static_cast<double>(total_intersect) / (nq * k);
}


class StopW {
    std::chrono::steady_clock::time_point time_begin;
 public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};



/*
* Author:  David Robert Nadeau
* Site:    http://NadeauSoftware.com/
* License: Creative Commons Attribution 3.0 Unported License
*          http://creativecommons.org/licenses/by/3.0/deed.en_US
*/

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif


/**
* Returns the peak (maximum so far) resident set size (physical
* memory use) measured in bytes, or zero if the value cannot be
* determined on this OS.
*/
static size_t getPeakRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L;      /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
        close(fd);
        return (size_t)0L;      /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t) (rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}


/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
static size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}


static void
test_approx(
    float *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<float> &appr_alg,
    size_t vecdim,
    std::vector<std::vector<unsigned>> &answers) {

    if (qsize % ext_batch_size != 0) {
        cout << "qsize must be divisible by batch size\n";
        exit(1);
    }
    
    cout << "qsize divided into " << qsize / ext_batch_size << " batches\n";
    cout << "time for each batch:\n";

    StopW stopw_batch = StopW();
    std::vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> res(qsize);
    float total_time = 0.0;
    if (ext_omp)
        for (int j = 0; j < qsize; j+=ext_batch_size) {
            stopw_batch.reset();
            #pragma omp parallel for num_threads(ext_interq_multithread) schedule(dynamic)
            for (int i = j; i < j + ext_batch_size; i++) {
                res[i] = appr_alg.searchKnn(massQ + vecdim * i, ext_ef);
            }
            float time_us_batch = stopw_batch.getElapsedTimeMicro();
            total_time += time_us_batch;
            cout << time_us_batch << " us\n";
        }

    else
        for (int j = 0; j < qsize; j+=ext_batch_size) {
            stopw_batch.reset();
            for (int i = j; i < j + ext_batch_size; i++) {
                res[i] = appr_alg.searchKnn(massQ + vecdim * i, ext_ef);
            }
            float time_us_batch = stopw_batch.getElapsedTimeMicro();
            total_time += time_us_batch;
            cout << time_us_batch << " us\n";
        }
    
    // reverse the priority queue and store the result in I
    std::vector<std::vector<unsigned>> I(qsize, std::vector<unsigned>(ext_ef));
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < ext_ef; j++) {
            I[i][ext_ef - 1 - j] = res[i].top().second;
            res[i].pop();
        }
    }

    // // print I[0]
    // for (int i = 0; i < I[0].size(); i++) {
    //     std::cout << I[0][i] << " ";
    // }
    // // print answers[0]
    // std::cout << "\n\n\n";
    // for (int i = 0; i < answers[0].size(); i++) {
    //     std::cout << answers[0][i] << " ";
    // }

    double recall_1, recall_10;
    if (ext_ef < 10) {
        recall_1 = calculate_recall(I, answers, 1);
        std::cout << "recall_1: " << recall_1 << std::endl;
    }
    else {
        recall_1 = calculate_recall(I, answers, 1);
        recall_10 = calculate_recall(I, answers, 10);
        std::cout << "recall_1: " << recall_1 << std::endl;
        std::cout << "recall_10: " << recall_10 << std::endl;
    }

    double qps = 1e6 * (float)qsize / total_time;
    std::cout << "qps: " << qps << std::endl;

    return;
}

static void
test_vs_recall(
    float *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<float> &appr_alg,
    size_t vecdim,
    std::vector<std::vector<unsigned>> &answers) {

    appr_alg.setEf(ext_ef);

    test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers);
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}


void hnsw_search() {
    float* data_load = NULL;
    unsigned points_num, dim;
    float* query_load = NULL;
    unsigned query_num, query_dim;
    std::vector<std::vector<unsigned>> gt;
    if (strcmp(ext_dataset, "SIFT1M") == 0) {
        std::cout << "load base vectors..." << std::endl;
        points_num = 1e6;
        load_data_bvecs("/mnt/scratch/wenqi/Faiss_experiments/bigann/bigann_base.bvecs", data_load, dim, points_num);
        std::cout << "load query vectors..." << std::endl;
        query_num = 1e4;
        load_data_bvecs("/mnt/scratch/wenqi/Faiss_experiments/bigann/bigann_query.bvecs", query_load, query_dim, query_num);
        load_data_ivecs("/mnt/scratch/wenqi/Faiss_experiments/bigann/gnd/idx_1M.ivecs", gt, query_num);
    }
    else if (strcmp(ext_dataset, "SIFT10M") == 0) {
        points_num = 1e7;
        load_data_bvecs("/mnt/scratch/wenqi/Faiss_experiments/bigann/bigann_base.bvecs", data_load, dim, points_num);
        query_num = 1e4;
        load_data_bvecs("/mnt/scratch/wenqi/Faiss_experiments/bigann/bigann_query.bvecs", query_load, query_dim, query_num);
        load_data_ivecs("/mnt/scratch/wenqi/Faiss_experiments/bigann/gnd/idx_10M.ivecs", gt, query_num);
    }
    else if (strcmp(ext_dataset, "SBERT1M") == 0) {
        points_num = 1e6;
        load_data_SBERT("/mnt/scratch/wenqi/Faiss_experiments/sbert/sbert1M.fvecs", data_load, points_num);
        dim = 384;
        query_num = 1e4;
        load_data_SBERT("/mnt/scratch/wenqi/Faiss_experiments/sbert/query_10K.fvecs", query_load, query_num);
        query_dim = 384;
        load_data_deep_ibin("/mnt/scratch/wenqi/Faiss_experiments/sbert/gt_idx_1M.ibin", gt, query_num);
    }
    else if (strcmp(ext_dataset, "Deep1M") == 0) {
        points_num = 1e6;
        load_data_deep_fbin("/mnt/scratch/wenqi/Faiss_experiments/deep1b/base.1B.fbin", data_load, points_num);
        dim = 96;
        query_num = 1e4;
        load_data_deep_fbin("/mnt/scratch/wenqi/Faiss_experiments/deep1b/query.public.10K.fbin", query_load, query_num);
        query_dim = 96;
        load_data_deep_ibin("/mnt/scratch/wenqi/Faiss_experiments/deep1b/gt_idx_1M.ibin", gt, query_num);
    }
    else if (strcmp(ext_dataset, "Deep10M") == 0) {
        points_num = 1e7;
        load_data_deep_fbin("/mnt/scratch/wenqi/Faiss_experiments/deep1b/base.1B.fbin", data_load, points_num);
        dim = 96;
        query_num = 1e4;
        load_data_deep_fbin("/mnt/scratch/wenqi/Faiss_experiments/deep1b/query.public.10K.fbin", query_load, query_num);
        query_dim = 96;
        load_data_deep_ibin("/mnt/scratch/wenqi/Faiss_experiments/deep1b/gt_idx_10M.ibin", gt, query_num);
    }
    else {
        std::cout << "Unknown dataset" << std::endl;
        exit(-1);
    }
    assert(dim == query_dim);
    
    size_t vecsize = (size_t)points_num;
    size_t qsize = (size_t)query_num;
    size_t vecdim = (size_t)dim;

    // int in = 0;
    L2Space l2space(vecdim);

    HierarchicalNSW<float> *appr_alg;
    if (exists_test(ext_hnsw_index_file)) {
        cout << "Loading index from " << ext_hnsw_index_file << ":\n";
        appr_alg = new HierarchicalNSW<float>(&l2space, ext_hnsw_index_file, false);
        cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    } else {
        cout << "Error: index file not found\n";
        return;
//         cout << "Building index:\n";
//         appr_alg = new HierarchicalNSW<int>(&l2space, vecsize, M, efConstruction);

//         input.read((char *) &in, 4);
//         if (in != 128) {
//             cout << "file error";
//             exit(1);
//         }
//         input.read((char *) massb, in * 4);

//         for (int j = 0; j < vecdim; j++) {
//             mass[j] = static_cast<unsigned char>(massb[j]);
//         }

//         appr_alg->addPoint((void *) (mass), (size_t) 0);
//         int j1 = 0;
//         StopW stopw = StopW();
//         StopW stopw_full = StopW();
//         size_t report_every = 100000;
// #pragma omp parallel for
//         for (int i = 1; i < vecsize; i++) {
//             unsigned char mass[vecdim];
//             int j2 = 0;
// #pragma omp critical
//             {
//                 input.read((char *) &in, 4);
//                 if (in != 128) {
//                     cout << "file error";
//                     exit(1);
//                 }
//                 input.read((char *) massb, in * 4);
//                 for (int j = 0; j < vecdim; j++) {
//                     mass[j] = static_cast<unsigned char>(massb[j]);
//                 }
//                 j1++;
//                 j2 = j1;
//                 if (j1 % report_every == 0) {
//                     cout << j1 / (0.01 * vecsize) << " %, "
//                          << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips " << " Mem: "
//                          << getCurrentRSS() / 1000000 << " Mb \n";
//                     stopw.reset();
//                 }
//             }
//             appr_alg->addPoint((void *) (mass), (size_t) j2);
//         }
//         input.close();
//         cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";
//         appr_alg->saveIndex(path_index);
    }

    test_vs_recall(query_load, vecsize, qsize, *appr_alg, vecdim, gt);
    cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    return;
}
