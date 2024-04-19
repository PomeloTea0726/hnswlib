#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "../../hnswlib/hnswlib.h"


#include <unordered_set>

using namespace std;
using namespace hnswlib;

extern int ext_M, ext_ef_construction, ext_ef, ext_k, ext_omp, ext_interq_multithread, ext_batch_size;
extern std::string ext_dataset_path;
long long node_counter;

extern double paralleltime;
extern std::vector<std::vector<double>> timevec;
int multicand, multithread;
double whilecounter = 0;

#define NUM_ANSWERS 100


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
get_gt(
    unsigned int *massQA,
    unsigned char *massQ,
    unsigned char *mass,
    size_t vecsize,
    size_t qsize,
    L2SpaceI &l2space,
    size_t vecdim,
    vector<std::priority_queue<std::pair<int, labeltype>>> &answers,
    size_t k) {
    (vector<std::priority_queue<std::pair<int, labeltype >>>(qsize)).swap(answers);
    DISTFUNC<int> fstdistfunc_ = l2space.get_dist_func();
    cout << qsize << "\n";
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[NUM_ANSWERS * i + j]);
        }
    }
}

static float
test_approx(
    unsigned char *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<int> &appr_alg,
    size_t vecdim,
    vector<std::priority_queue<std::pair<int, labeltype>>> &answers,
    size_t k) {
    size_t correct = 0;
    size_t total = 0;

    if (qsize % ext_batch_size != 0) {
        cout << "qsize must be divisible by batch size\n";
        exit(1);
    }
    
    cout << "qsize divided into " << qsize / ext_batch_size << " batches\n";
    cout << "node counter and time for each batch:\n";

    // float temp = appr_alg.readgraph();
    // cout << temp << "\n";

    StopW stopw_batch = StopW();
    if (ext_omp)
        for (int j = 0; j < qsize; j+=ext_batch_size) {
            node_counter = 0;
            stopw_batch.reset();
            #pragma omp parallel for reduction(+ : correct, total, node_counter) num_threads(ext_interq_multithread) schedule(dynamic)
            for (int i = j; i < j + ext_batch_size; i++) {
                std::priority_queue<std::pair<int, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
                std::priority_queue<std::pair<int, labeltype >> gt(answers[i]);
                unordered_set<labeltype> g;
                total += gt.size();

                while (gt.size()) {
                    g.insert(gt.top().second);
                    gt.pop();
                }

                while (result.size()) {
                    if (g.find(result.top().second) != g.end()) {
                        correct++;
                    } else {
                    }
                    result.pop();
                }
            }
            float time_us_batch = stopw_batch.getElapsedTimeMicro();
            cout << node_counter << " ";
            cout << time_us_batch << " us\n";
        }

    else
        for (int j = 0; j < qsize; j+=ext_batch_size) {
            node_counter = 0;
            stopw_batch.reset();
            for (int i = j; i < j + ext_batch_size; i++) {
                std::priority_queue<std::pair<int, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
                std::priority_queue<std::pair<int, labeltype >> gt(answers[i]);
                unordered_set<labeltype> g;
                total += gt.size();

                while (gt.size()) {
                    g.insert(gt.top().second);
                    gt.pop();
                }

                while (result.size()) {
                    if (g.find(result.top().second) != g.end()) {
                        correct++;
                    } else {
                    }
                    result.pop();
                }
            }
            float time_us_batch = stopw_batch.getElapsedTimeMicro();
            cout << node_counter << " ";
            cout << time_us_batch << " us\n";
        }

    return 1.0f * correct / total;
}

static void
test_vs_recall(
    unsigned char *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<int> &appr_alg,
    size_t vecdim,
    vector<std::priority_queue<std::pair<int, labeltype>>> &answers,
    size_t k) {
    vector<size_t> efs;  // = { 10,10,10,10,10 };
    // for (int i = k; i < 30; i++) {
    //     efs.push_back(i);
    // }
    // for (int i = 30; i < 100; i += 10) {
    //     efs.push_back(i);
    // }
    // for (int i = 200; i < 600; i += 100) {
    //     efs.push_back(i);
    //     // efs.push_back(i);
    // }
    efs.push_back(ext_ef);
    // efs.push_back(ext_ef);
    // efs.push_back(ext_ef);
    // efs.push_back(ext_ef);
    // ofstream outfile("log_16_40_340_100.txt", std::ios::app);
    for (size_t ef : efs) {
        // whilecounter = 0;
        // paralleltime = 0;
        // timevec[0][0] = 0;
        // timevec[1][0] = 0;
        // for (int i = 0; i <= multithread; i++) {
        //     timevec[3][i] = 0;
        //     timevec[5][i] = 0;
        // }
        // timevec[4][0] = 0;


        appr_alg.setEf(ef);
        StopW stopw = StopW();

        float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
        float query_per_second = 1e6 * qsize / stopw.getElapsedTimeMicro();

        cout << "ef: " << ef << "\n";
        cout << "qps: " << query_per_second << "\n";
        cout << "recall: " << recall << "\n";

        // cout << "multicand: " << multicand << " multithread: " << multithread << " M: " << ext_M << " ef: " << ef << "\n";

        // whilecounter /= qsize;
        // cout << "whilecounter: " << whilecounter << "\n";
        
        // // cout << "ef: " << ef << "\n";
        // cout << query_per_second << " ";
        // cout << recall << "\n";

        // // cout << "avg_cost: " << 1.0 * 1e6 / query_per_second << " us\n";
        // cout << "0: " << timevec[0][0] / qsize << "\n";
        // cout << "1: " << timevec[1][0] / qsize << "\n";
        // double avg_time = 0;
        // for (int i = 0; i < multithread; i++) {
        //     avg_time += timevec[3][i] / qsize / multithread;
        //     // cout << (timevec[3][i] / qsize) / whilecounter << "\n";
        //     cout << (timevec[3][i] / qsize) << "\n";
        // }
        // double maxpath = (timevec[3][multithread] / qsize);
        // cout << "avg time: " << avg_time << ", idle time: " << maxpath - avg_time << endl;

        // // double paratime = (paralleltime / qsize) / whilecounter;
        // double paratime = (paralleltime / qsize);
        // // cout << "parallel time = " << paratime << "\n";
        // cout << "maxpath: " << maxpath << ", paratime: " << paratime << ", diff: " << paratime - maxpath << "\n";
        // cout << "4: " << timevec[4][0] / qsize << "\n";
    }
    // outfile.close();
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}


void sift_test1M() {
    int subset_size_milllions = 1;
    int efConstruction = ext_ef_construction; // original value = 40
    int M = ext_M;

    size_t vecsize = subset_size_milllions * 1000000;

    size_t qsize = 10000;
    size_t vecdim = 128;
    char path_index[1024];
    char path_gt[1024];

    const string path_q = ext_dataset_path + "sift_query.fvecs";
    const string path_data = ext_dataset_path + "sift_base.fvecs";
    snprintf(path_index, sizeof(path_index), "sift1m_ef_%d_M_%d.bin", efConstruction, M);

    snprintf(path_gt, sizeof(path_gt), "%s/sift_groundtruth.ivecs", ext_dataset_path.c_str());

    float *massb = new float[vecdim];

    cout << "Loading GT:\n";
    ifstream inputGT(path_gt, ios::binary);
    unsigned int *massQA = new unsigned int[qsize * NUM_ANSWERS];
    
    for (int i = 0; i < qsize; i++) {
        int t;
        inputGT.read((char *) &t, 4);
        inputGT.read((char *) (massQA + NUM_ANSWERS * i), t * 4);
        if (t != NUM_ANSWERS) {
            cout << "err";
            return;
        }
    }
    inputGT.close();

    cout << "Loading queries:\n";
    unsigned char *massQ = new unsigned char[qsize * vecdim];
    ifstream inputQ(path_q, ios::binary);

    for (int i = 0; i < qsize; i++) {
        int in = 0;
        inputQ.read((char *) &in, 4);
        if (in != 128) {
            cout << "file error";
            exit(1);
        }
        inputQ.read((char *) massb, in * 4);
        for (int j = 0; j < vecdim; j++) {
            massQ[i * vecdim + j] = static_cast<unsigned char>(massb[j]);
        }
    }
    inputQ.close();


    unsigned char *mass = new unsigned char[vecdim];
    ifstream input(path_data, ios::binary);
    int in = 0;
    L2SpaceI l2space(vecdim);

    HierarchicalNSW<int> *appr_alg;
    if (exists_test(path_index)) {
        cout << "Loading index from " << path_index << ":\n";
        appr_alg = new HierarchicalNSW<int>(&l2space, path_index, false);
        cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    } else {
        cout << "Building index:\n";
        appr_alg = new HierarchicalNSW<int>(&l2space, vecsize, M, efConstruction);

        input.read((char *) &in, 4);
        if (in != 128) {
            cout << "file error";
            exit(1);
        }
        input.read((char *) massb, in * 4);

        for (int j = 0; j < vecdim; j++) {
            mass[j] = static_cast<unsigned char>(massb[j]);
        }

        appr_alg->addPoint((void *) (mass), (size_t) 0);
        int j1 = 0;
        StopW stopw = StopW();
        StopW stopw_full = StopW();
        size_t report_every = 100000;
#pragma omp parallel for
        for (int i = 1; i < vecsize; i++) {
            unsigned char mass[vecdim];
            int j2 = 0;
#pragma omp critical
            {
                input.read((char *) &in, 4);
                if (in != 128) {
                    cout << "file error";
                    exit(1);
                }
                input.read((char *) massb, in * 4);
                for (int j = 0; j < vecdim; j++) {
                    mass[j] = static_cast<unsigned char>(massb[j]);
                }
                j1++;
                j2 = j1;
                if (j1 % report_every == 0) {
                    cout << j1 / (0.01 * vecsize) << " %, "
                         << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips " << " Mem: "
                         << getCurrentRSS() / 1000000 << " Mb \n";
                    stopw.reset();
                }
            }
            appr_alg->addPoint((void *) (mass), (size_t) j2);
        }
        input.close();
        cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";
        appr_alg->saveIndex(path_index);
    }


    vector<std::priority_queue<std::pair<int, labeltype >>> answers;
    size_t k = ext_k;
    cout << "Parsing gt:\n";
    get_gt(massQA, massQ, mass, vecsize, qsize, l2space, vecdim, answers, k);
    cout << "Loaded gt\n";
    
    multicand = 4, multithread = 1;
    test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);

    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);

    // multicand = 8, multithread = 4;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);

    // multicand = 16, multithread = 4;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);

    // multicand = 32, multithread = 4;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);

    // multicand = 64, multithread = 4;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);

    // multicand = 2, multithread = 1;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);

    // multicand = 4, multithread = 1;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);

    // multicand = 8, multithread = 1;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);

    // multicand = 16, multithread = 1;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);

    // multicand = 32, multithread = 1;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);


    // multicand = 4, multithread = 4;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);

    // multicand = 8, multithread = 8;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    
    // multicand = 16, multithread = 16;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);

    // multicand = 32, multithread = 1;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);

    // multicand = 32, multithread = 2;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);

    // multicand = 32, multithread = 4;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);

    // multicand = 32, multithread = 8;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);

    // multicand = 32, multithread = 32;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);

    // multicand = 64, multithread = 32;
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    // test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);



    // int multicand_list[] = {1,2,4,8,16,32};
    // int multithread_list[] = {1,2,4,8,16,32};
    // for (int i = 0; i < 6; i++) {
    //     for (int j = 0; j <= i; j++) {
    //         multicand = multicand_list[i];
    //         multithread = multithread_list[j];
    //         test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    //     }
    // }

    // int multithread_list[] = {1,2,4};
    // int multicand_list[] = {1,2,4,8,16,32};
    // for (int i = 0; i < 3; i++) {
    //     for (int j = i; j < 6; j++) {
    //         multithread = multithread_list[i];
    //         multicand = multicand_list[j];
    //         test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    //     }
    // }
    
    cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    return;
}
