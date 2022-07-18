
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
#include <atomic>
#include "rocprofiler.h"


// Check returned HSA API status
void check_status(hsa_status_t status) {
  if (status != HSA_STATUS_SUCCESS) {
    const char* error_string = NULL;
    rocprofiler_error_string(&error_string);
    std::cout << "rocprof ERROR: %s\n";
    abort();
  }
}

pthread_mutex_t mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;

// Context stored entry type
struct context_entry_t {
  bool valid;
  hsa_agent_t agent;
  rocprofiler_group_t group;
  rocprofiler_callback_data_t data;
};

// Dump stored context entry
void dump_context_entry(context_entry_t* entry) {
  volatile std::atomic<bool>* valid = reinterpret_cast<std::atomic<bool>*>(&entry->valid);
  while (valid->load() == false) sched_yield();

  const std::string kernel_name = entry->data.kernel_name;
  const rocprofiler_dispatch_record_t* record = entry->data.record;


  std::cout << "kernel-object name" <<  kernel_name.c_str() << "\n";

  rocprofiler_group_t& group = entry->group;
  if (group.context == NULL) {
    std::cout << "tool error: context is NULL\n";
  }

  rocprofiler_close(group.context);
}

// Profiling completion handler
// Dump and delete the context entry
// Return true if the context was dumped successfully
bool context_handler(rocprofiler_group_t group, void* arg) {
  context_entry_t* entry = reinterpret_cast<context_entry_t*>(arg);

  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }

  dump_context_entry(entry);
  delete entry;

  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }

  return false;
}

// Kernel disoatch callback
hsa_status_t dispatch_callback(const rocprofiler_callback_data_t* callback_data, void* /*user_data*/,
                               rocprofiler_group_t* group) {
    std::cout << "dispatch callback\n";
  // HSA status
  hsa_status_t status = HSA_STATUS_ERROR;

  // Profiling context
  rocprofiler_t* context = NULL;

  // Context entry
  context_entry_t* entry = new context_entry_t();

  // context properties
  rocprofiler_properties_t properties{};
  properties.handler = context_handler;
  properties.handler_arg = (void*)entry;

  // Open profiling context
  status = rocprofiler_open(callback_data->agent, NULL, 0,
                            &context, ROCPROFILER_MODE_SINGLEGROUP, &properties);
  check_status(status);

  // Get group[0]
  status = rocprofiler_get_group(context, 0, group);
  check_status(status);

  // Fill profiling context entry
  entry->agent = callback_data->agent;
  entry->group = *group;
  entry->data = *callback_data;
  entry->data.kernel_name = strdup(callback_data->kernel_name);
  reinterpret_cast<std::atomic<bool>*>(&entry->valid)->store(true);

  return HSA_STATUS_SUCCESS;
}


/*hsa_status_t dispatch_callback(const rocprofiler_callback_data_t *callback_data,
                               void *user_data, rocprofiler_group_t *group) {

  hsa_status_t status = HSA_STATUS_SUCCESS;


  // Profiling info objects
  rocprofiler_feature_t features * = new rocprofiler_feature_t[2];
  // Tracing parameters
  rocprofiler_feature_parameter_t *parameters =
      new rocprofiler_feature_parameter_t[2];

  // Setting profiling features
  features[0].type = ROCPROFILER_METRIC;
  features[0].name = "L1_MISS_RATIO";
  features[1].type = ROCPROFILER_METRIC;
  features[1].name = "DRAM_BANDWIDTH";

  // Creating profiling context
  status = rocprofiler_open(callback_data->dispatch.agent, features, 2,
                            &context, ROCPROFILER_MODE_SINGLEGROUP, NULL);

  return status;
}*/




void measureMetricInit(){};
void measureBandwidthStart() {
    std::cout << "measureBandwidthStart\n";
    rocprofiler_queue_callbacks_t callbacks_ptrs{};
    callbacks_ptrs.dispatch = dispatch_callback;
    check_status( rocprofiler_set_queue_callbacks(callbacks_ptrs, NULL) );
    check_status( rocprofiler_start_queue_callbacks() );

};

std::vector<double> measureMetricStop() {

    std::cout << "measureBandwidthStop\n";
    rocprofiler_stop_queue_callbacks();



    return std::vector<double>(6, 0.0);
}

#endif // ROCM-METRICS_H_
