
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

#include "hip/hip_runtime.h"
#include <hsa/hsa.h>
#include <iostream>
#include <rocprofiler.h>
#include <vector>
#include <unistd.h>

#define HSA_ASSERT(x) (assert((x) == HSA_STATUS_SUCCESS))

#define ROCP_CALL_CK(call)                                                     \
  do {                                                                         \
    hsa_status_t _status = call;                                               \
    if (_status != HSA_STATUS_SUCCESS && _status != HSA_STATUS_INFO_BREAK) {   \
      const char *profErr;                                                     \
      rocprofiler_error_string(&profErr);                                      \
      std::cout << "ERROR: function call \n \"" << #call                       \
                << "\" at " __FILE__ ":" << __LINE__                           \
                << " \n failed with status " << _status << ": \" " << profErr  \
                << "\"\n";                                                     \
    }                                                                          \
  } while (0);

hsa_agent_t agent_info_arr[16];
unsigned agent_info_arr_len;

static hsa_status_t _count_devices(hsa_agent_t agent, void *data) {
  unsigned *count = (unsigned *)data;
  hsa_device_type_t type;
  hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  assert(status == HSA_STATUS_SUCCESS);
  if (type == HSA_DEVICE_TYPE_GPU) {
    agent_info_arr[(*count)++] = agent;
  }
  return status;
}

static unsigned _get_device_count(void) {
  unsigned count = 0;
  hsa_status_t status = hsa_iterate_agents(&_count_devices, &count);
  assert(status == HSA_STATUS_SUCCESS);
  return count;
}

static hsa_agent_t _get_agent(unsigned gpu_id) {
  return agent_info_arr[gpu_id];
}

hsa_status_t info_data_callback(const rocprofiler_info_data_t info,
                                void *data) {

    std::cout << "info data callback\n";
  switch (info.kind) {
  case ROCPROFILER_INFO_KIND_METRIC: {
    if (info.metric.expr != NULL) {
      std::cout << "Derived counter:  gpu-agent" << info.agent_index << " "
                << info.metric.name << ": " << info.metric.description;

      std::cout << info.metric.name << " = " << info.metric.expr << "\n";
    } else {
      std::cout << "Basic counter:  gpu-agent" << info.agent_index << ": "
                << info.metric.name << "\n";
      if (info.metric.instances > 1) {
        std::cout << "[0-" << info.metric.instances - 1 << "]\n";
      }
      std::cout << " : " << info.metric.description;
      std::cout << "      block " << info.metric.block_name << " has "
                << info.metric.block_counters << " counters\n";
    }
    break;
  }
  default:
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}
void printMetrics(hsa_agent_t agent) {
  ROCP_CALL_CK( rocprofiler_iterate_info(
      &agent, ROCPROFILER_INFO_KIND_METRIC, info_data_callback, NULL));

}

hsa_agent_t agent;
// Profiling context
rocprofiler_t *context = NULL;

const unsigned feature_count = 2;
rocprofiler_feature_t feature[feature_count];
double prevValues[feature_count];


void measureBandwidthStart() {
    hipDeviceSynchronize();
  // Start counters and sample them in the loop with the sampling rate
}

std::vector<double> measureMetricStop() {
  hipDeviceSynchronize();


  std::vector<double> results(6,0);

  ROCP_CALL_CK( rocprofiler_read(context, 0));
  ROCP_CALL_CK( rocprofiler_get_data(context, 0));
  ROCP_CALL_CK( rocprofiler_get_metrics(context));
  // print_results(feature, feature_count);


  double hits = (feature[0].data.result_double - prevValues[0]);
  double misses = (feature[1].data.result_double - prevValues[1]);


  results[0] = misses*32;
  results[2] = hits+misses;

  

  for (unsigned i = 0; i < feature_count; ++i) {
      const rocprofiler_feature_t *p = &feature[i];
      //std::cout << p->name << ": ";

      double val = 0;
      switch(p->data.kind) {
          case ROCPROFILER_DATA_KIND_INT64:
              val = p->data.result_int64;
              break;
          case ROCPROFILER_DATA_KIND_DOUBLE:
              val = p->data.result_double;
              break;
          default:
              std::cout << "Undefined data kind: " << p->data.kind << "\n";
              assert(0);
      }
      //std::cout << "= " << val << ", Delta: " << val - prevValues[i] << "\n";
      prevValues[i] = val;
  }



  // Stop counters
  //ROCP_CALL_CK( rocprofiler_stop(context, 0));

  return results;
}

void initMeasureMetric() {
  setenv("HSA_TOOLS_LIB", "/opt/rocm/rocprofiler/lib/librocprofiler64.so", 1);
  setenv("ROCP_METRICS", "/opt/rocm/lib/rocprofiler/metrics.xml", 1);


  HSA_ASSERT(hsa_init());
  hsa_status_t status = HSA_STATUS_ERROR;
  // HSA agent

  unsigned gpu_count = _get_device_count();
  agent_info_arr_len = gpu_count;

  for (unsigned gpu_id = 0; gpu_id < gpu_count; ++gpu_id) {
    hsa_agent_t agent = _get_agent(gpu_id);
    std::cout << "Agent " << gpu_id << "\n";
  }

  agent = _get_agent(0);

  //printMetrics(agent);

  // Profiling feature objects

  // Counters and metrics
  feature[0].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  feature[0].name = "TCC_HIT_sum";
  feature[1].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  feature[1].name = "TCC_MISS_sum";

  //feature[2].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  //feature[2].name = "FETCH_SIZE";
  //feature[3].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  //feature[3].name = "WRITE_SIZE";

  // Creating profiling context with standalone queue
  rocprofiler_properties_t properties = {};
  properties.queue_depth = 128;
  uint32_t mode = ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_CREATEQUEUE |
                  ROCPROFILER_MODE_SINGLEGROUP;

  properties.queue_depth = 128;

  ROCP_CALL_CK(rocprofiler_open(agent, feature, feature_count, &context, mode,
                                &properties));

  ROCP_CALL_CK(rocprofiler_start(context, 0));
}

#endif // ROCM-METRICS_H_
