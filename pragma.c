#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <stdint.h>

//PERF
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define NUM_THREADS 5
#define PERF_EVENTS 2

struct read_format {
    uint64_t nr;
    struct {
        long int value;
    } values[PERF_EVENTS]; 
};

int main(void) 
{
    struct perf_event_attr base = {
        .type = PERF_TYPE_HW_CACHE,
        .size = sizeof(struct perf_event_attr),
        .read_format = PERF_FORMAT_GROUP,
        .exclude_kernel = 1
    };
    struct perf_event_attr perfs[PERF_EVENTS];
    for(int i=0;i<PERF_EVENTS;i++)
        memcpy(&perfs[i], &base, sizeof(struct perf_event_attr)); 

    perfs[0].config = PERF_COUNT_HW_CACHE_RESULT_MISS  << 16 | PERF_COUNT_HW_CACHE_OP_READ << 8 | PERF_COUNT_HW_CACHE_L1D;
    perfs[1].config = PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16 | PERF_COUNT_HW_CACHE_OP_READ << 8 | PERF_COUNT_HW_CACHE_L1D;

    int fd = syscall(__NR_perf_event_open, &perfs[0], 0, -1, -1, 0);
    syscall(__NR_perf_event_open, &perfs[1], 0, -1, fd, 0);
    ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    double st = omp_get_wtime();
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);

    size_t steps = 100000000;
    double step = 1.0/steps;
    double sum = 0;
    omp_set_num_threads(NUM_THREADS);   
    #pragma omp parallel
    {
        int idx = omp_get_thread_num();
        int allth = omp_get_num_threads();
        double sumc = 0;
        for(int i=idx;i<steps;i+=allth) 
        {
           double x = (i+0.5)*step;
           sumc+=4.0/(1.0+x*x);
        }
        #pragma omp atomic
            sum += sumc;
    }

    float pi = sum*step;
    printf("Pi %lf\n", pi);


    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    struct read_format rf;
    read(fd, &rf, sizeof(struct read_format));
    printf("Exec Time %.3lf\n", omp_get_wtime()-st); 
    printf("Cache Misses %ld/%ld\n", rf.values[0].value, rf.values[1].value);

}
