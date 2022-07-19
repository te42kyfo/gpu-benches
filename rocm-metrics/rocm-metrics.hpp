
/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. and Dominik Ernst
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

#ifndef ROCM_METRICS_H_
#define ROCM_METRICS_H_

#include <vector>
#include <iostream>
#include "rocprofiler.h"

#include <hsa.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <atomic>

#include <stdatomic.h>

typedef struct {
#define MAX_DEVICE_COUNT 16
    rocprofiler_pool_t *pools[MAX_DEVICE_COUNT];
} callback_arg_t;

typedef struct {
    rocprofiler_feature_t *features;
    unsigned feature_count;
} handler_arg_t;

typedef struct {
    bool valid;
    hsa_agent_t agent;
    rocprofiler_group_t group;
    rocprofiler_callback_data_t data;
} context_entry_t;

typedef struct {
    hsa_agent_t agent;
} agent_info_t;

/* Data structures handled by init and shutdown */
handler_arg_t *handler_arg;
callback_arg_t callback_arg[1];
agent_info_t agent_info_arr[16];
unsigned agent_info_arr_len;
rocprofiler_feature_t *features;
pthread_mutex_t feature_lock = PTHREAD_MUTEX_INITIALIZER;

static unsigned _get_gpu_id(hsa_agent_t agent)
{
    unsigned gpu_id;
    for (gpu_id = 0; gpu_id < agent_info_arr_len; ++gpu_id) {
        if (memcmp(&agent_info_arr[gpu_id].agent, &agent, sizeof(hsa_agent_t)) == 0) {
            return gpu_id;
        }
    }
    return -1;
}

static void _atomic_store(volatile bool *flag, bool value)
{
#ifndef __STDC_NO_ATOMICS__
    atomic_store((atomic_bool *) flag, value);
#else
#error "not atomics. brace yourselves!"
    *flag = value;
#endif
}

static bool _atomic_load(const volatile bool *flag)
{
#ifndef __STDC_NO_ATOMICS__
    return atomic_load((atomic_bool *) flag);
#else
#error "not atomics. brace yourselves!"
    return *flag;
#endif
}

static hsa_status_t _rocp_dispatch_callback(const rocprofiler_callback_data_t *callback_data,
                                            void *arg, rocprofiler_group_t *group)
{
  std::cout << "dispatch\n";
    hsa_agent_t agent = callback_data->agent;
    hsa_status_t status = HSA_STATUS_ERROR;

    unsigned gpu_id = _get_gpu_id(agent);
    assert(gpu_id >= 0);

    callback_arg_t *callback_arg = (callback_arg_t *) arg;
    rocprofiler_pool_t *pool = callback_arg->pools[gpu_id];
    rocprofiler_pool_entry_t pool_entry;
    status = rocprofiler_pool_fetch(pool, &pool_entry);
    assert(status == HSA_STATUS_SUCCESS);

    rocprofiler_t *context = pool_entry.context;
    context_entry_t *entry = (context_entry_t *) pool_entry.payload;

    status = rocprofiler_get_group(context, 0, group);
    assert(status == HSA_STATUS_SUCCESS);

    entry->agent = agent;
    entry->group = *group;
    entry->data = *callback_data;
    entry->data.kernel_name = strdup(callback_data->kernel_name);
    _atomic_store(&entry->valid, true);

    //fprintf(stdout, "%s tid(%ld)\n", __func__, syscall(SYS_gettid));

    return status;
}

static void _dump_context_entry(context_entry_t *entry,
                                rocprofiler_feature_t *features,
                                unsigned feature_count)
{
    std::cout << "dumo_context_entry\n";

    while(_atomic_load(&entry->valid) == false) sched_yield();

    fflush(stdout);
    fprintf(stdout, "kernel symbol(0x%lx) name(\"%s\") tid(%u) queue-id(%u) "
                    "gpu-id(%u) ",
                    entry->data.kernel_object,
                    entry->data.kernel_name,
                    entry->data.thread_id,
                    entry->data.queue_id,
                    _get_gpu_id(entry->agent));
    if (entry->data.record)
        fprintf(stdout, "time(%lu,%lu,%lu,%lu)\n",
                        entry->data.record->dispatch,
                        entry->data.record->begin,
                        entry->data.record->end,
                        entry->data.record->complete);
    fflush(stdout);

    assert(entry->group.context != NULL);

    if (feature_count > 0) {
        hsa_status_t status = rocprofiler_group_get_data(&entry->group);
        assert(status == HSA_STATUS_SUCCESS);
        status = rocprofiler_get_metrics(entry->group.context);
        assert(status == HSA_STATUS_SUCCESS);
    }

    for (unsigned i = 0; i < feature_count; ++i) {
        const rocprofiler_feature_t *p = &features[i];
        fprintf(stdout, "> %s ", p->name);
        switch(p->data.kind) {
            case ROCPROFILER_DATA_KIND_INT64:
                fprintf(stdout, "= (%lu)\n", p->data.result_int64);
                break;
            default:
                fprintf(stdout, "Undefined data kind(%u)\n", p->data.kind);
                assert(0);
        }
    }
}

static bool _rocp_context_handler(const rocprofiler_pool_entry_t *entry,
                                  void *arg)
{
    context_entry_t *ctx_entry = (context_entry_t *) entry->payload;
    handler_arg_t *handler_arg = (handler_arg_t *) arg;

    pthread_mutex_lock(&feature_lock);

    //fprintf(stdout, "%s tid(%ld)\n", __func__, syscall(SYS_gettid));
    _dump_context_entry(ctx_entry, handler_arg->features,
                        handler_arg->feature_count);

    pthread_mutex_unlock(&feature_lock);

    return false;
}

static hsa_status_t _count_devices(hsa_agent_t agent, void *data)
{
    unsigned *count = (unsigned *) data;
    hsa_device_type_t type;
    hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    assert(status == HSA_STATUS_SUCCESS);
    if (type == HSA_DEVICE_TYPE_GPU) {
        agent_info_arr[(*count)++].agent = agent;
    }
    return status;
}

static unsigned _get_device_count(void)
{
    unsigned count = 0;
    hsa_status_t status = hsa_iterate_agents(&_count_devices, &count);
    assert(status == HSA_STATUS_SUCCESS);
    return count;
}

static hsa_agent_t _get_agent(unsigned gpu_id)
{
    return agent_info_arr[gpu_id].agent;
}

static int init_intercept(rocprofiler_feature_t *features, unsigned feature_count)
{
    std::cout << "init intercept\n";
    handler_arg = (handler_arg_t *)calloc(1, sizeof(*handler_arg));
    assert(handler_arg != NULL);
    handler_arg->features = features;
    handler_arg->feature_count = feature_count;

    rocprofiler_pool_properties_t properties;
    properties.num_entries = 100;
    properties.payload_bytes = sizeof(context_entry_t);
    properties.handler = _rocp_context_handler;
    properties.handler_arg = handler_arg;

    unsigned gpu_count = _get_device_count();
    agent_info_arr_len = gpu_count;

    for (unsigned gpu_id = 0; gpu_id < gpu_count; ++gpu_id) {
        hsa_agent_t agent = _get_agent(gpu_id);

        rocprofiler_pool_t *pool = NULL;
        hsa_status_t status = rocprofiler_pool_open(agent, features,
                                                    feature_count, &pool,
                                                    0, &properties);
        assert(status == HSA_STATUS_SUCCESS);

        callback_arg->pools[gpu_id] = pool;
    }

    rocprofiler_queue_callbacks_t callback_ptr;
    callback_ptr.dispatch = _rocp_dispatch_callback;

    hsa_status_t status = rocprofiler_set_queue_callbacks(callback_ptr,
                                                          callback_arg);
    assert(status == HSA_STATUS_SUCCESS);

    return 0;
}

static int start_intercept(void)
{
    std::cout << "start intercept\n";
    hsa_status_t status = rocprofiler_start_queue_callbacks();
    assert(status == HSA_STATUS_SUCCESS);
    return 0;
}

static int stop_intercept(void)
{
    std::cout << "stop intercept\n";
    hsa_status_t status = rocprofiler_stop_queue_callbacks();
    assert(status == HSA_STATUS_SUCCESS);
    return 0;
}

static int shutdown_intercept(void)
{
    std::cout << "shutdown intercept\n";
    hsa_status_t status = rocprofiler_remove_queue_callbacks();
    assert(status == HSA_STATUS_SUCCESS);

    int i;
    //for (i = 0; i < agent_info_arr_len; ++i) {
    //    status = rocprofiler_pool_flush(callback_arg->pools[i]);
    //    assert(status == HSA_STATUS_SUCCESS);
    //    status = rocprofiler_pool_close(callback_arg->pools[i]);
    //    assert(status == HSA_STATUS_SUCCESS);
    //}
    ////free(handler_arg);
    return 0;
}
void measureBandwidthStart() {
    start_intercept();

}

std::vector<double> measureMetricStop() {
    hipDeviceSynchronize();
    stop_intercept();
    return std::vector<double>(6, 0);

}

void initMeasureMetric() {
    char*rocm_root = "/opt/rocm/";

    char metrics_path[128];
  sprintf(metrics_path, "%s/%s", rocm_root, "rocprofiler/lib/metrics.xml");
  //setenv("ROCP_TOOL_LIB", "/opt/rocm/rocprofiler/tool/libtool.so", 1);
  setenv("ROCP_METRICS", metrics_path, 1);
  setenv("ROCP_HSA_INTERCEPT", "1", 1);
  setenv("ROCPROFILER_LOG", "1", 1);
  setenv("HSA_TOOLS_LIB", "librocprofiler64.so", 1);
  setenv("HSA_TOOLS_LIB", "librocprofiler64.so", 1);


  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  std::cout << " System minor " << devProp.minor << std::endl;
  std::cout << " System major " << devProp.major << std::endl;
  std::cout << " agent prop name " << devProp.name << std::endl;

  std::cout << "hip Device prop succeeded " << std::endl ;
  rocprofiler_feature_t features[4];
  features[0].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[0].name = "SQ_WAVES";
  features[1].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[1].name = "SQ_INSTS_VALU";

  unsigned feature_count = 2;
  init_intercept(features, feature_count);
}


#endif // ROCM-METRICS_H_
