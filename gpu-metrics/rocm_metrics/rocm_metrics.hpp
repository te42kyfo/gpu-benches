
#include "hip/hip_runtime.h"
#include <hsa/hsa.h>
#include <iostream>
#include <rocprofiler.h>
#include <unistd.h>
#include <vector>

#include <dlfcn.h>
#include <hsa/hsa.h>
#include <string.h>
#include <unistd.h>

#include <atomic>
#include <iostream>
#include <sstream>
#include <vector>

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

// Dispatch callbacks and context handlers synchronization
pthread_mutex_t mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
// Tool is unloaded
volatile bool is_loaded = false;
// Profiling features
// rocprofiler_feature_t* features = NULL;
// unsigned feature_count = 0;

// Error handler
void fatal(const std::string msg) {
  fflush(stdout);
  fprintf(stderr, "%s\n\n", msg.c_str());
  fflush(stderr);
  abort();
}

// Check returned HSA API status
void check_status(hsa_status_t status) {
  if (status != HSA_STATUS_SUCCESS) {
    const char *error_string = NULL;
    rocprofiler_error_string(&error_string);
    fprintf(stderr, "ERROR: %s\n", error_string);
    abort();
  }
}

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

// Context stored entry type
struct context_entry_t {
  bool completed;
  rocprofiler_group_t group;
  rocprofiler_callback_data_t data;
  std::vector<rocprofiler_feature_t> features;
};

std::vector<context_entry_t> context_queue;
std::vector<rocprofiler_feature_t> current_features;

// Dump stored context entry
bool context_handler(rocprofiler_group_t group, void *arg) {
  context_entry_t *entry = reinterpret_cast<context_entry_t *>(arg);
  // std::cout << "context handler\n";

  // dump_context_entry(entry, entry->features.data(), entry->features.size());

  check_status(rocprofiler_group_get_data(&entry->group));
  check_status(rocprofiler_get_metrics(entry->group.context));

  entry->completed = true;
  return true;
}

// Kernel dispatch callback
hsa_status_t dispatch_callback(const rocprofiler_callback_data_t *callback_data,
                               void *arg, rocprofiler_group_t *group) {
  // std::cout << "dispatch callback\n";
  hsa_agent_t agent = _get_agent(0);

  context_queue.push_back(context_entry_t());
  auto entry = &(context_queue.back());
  entry->features = current_features;

  // context properties
  rocprofiler_properties_t properties{};
  properties.handler = context_handler;
  properties.handler_arg = (void *)entry;

  // Open profiling context
  rocprofiler_t *context = NULL;
  check_status(rocprofiler_open(agent, entry->features.data(),
                                entry->features.size(), &context, 0,
                                &properties));

  // Check that we have only one profiling group
  uint32_t group_count = 0;
  check_status(rocprofiler_group_count(context, &group_count));
  assert(group_count == 1);

  // Get group[0]
  check_status(rocprofiler_get_group(context, 0, group));

  // Fill profiling context entry
  entry->group = *group;
  entry->completed = false;
  entry->data = *callback_data;
  entry->data.kernel_name = strdup(callback_data->kernel_name);

  return HSA_STATUS_SUCCESS;
}

void initMeasureMetric() {

  setenv("HSA_TOOLS_LIB", "/opt/rocm/rocprofiler/lib/librocprofiler64.so", 1);
  setenv("ROCP_METRICS", "/opt/rocm/lib/rocprofiler/metrics.xml", 1);
  setenv("ROCP_HSA_INTERCEPT", "1", 1);
  hsa_init();
  unsigned gpu_count = _get_device_count();
  agent_info_arr_len = gpu_count;

  for (unsigned gpu_id = 0; gpu_id < gpu_count; ++gpu_id) {
    hsa_agent_t agent = _get_agent(gpu_id);
    // std::cout << "Agent " << gpu_id << "\n";

    char agent_name[64];
    hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, &agent_name);
    std::cout << agent_name << "\n";
  }

  rocprofiler_queue_callbacks_t callbacks_ptrs{};
  callbacks_ptrs.dispatch = dispatch_callback;
  // std::cout << "set queue callbacks\n";
  rocprofiler_set_queue_callbacks(callbacks_ptrs, NULL);

  rocprofiler_stop_queue_callbacks();
}

void cleanup() {
  // Unregister dispatch callback
  rocprofiler_remove_queue_callbacks();
}

void measureMetricsStart(std::vector<const char *> metricNames) {

  static bool initialized = false;
  if (!initialized) {
    initMeasureMetric();
    initialized = true;
  }

  hsa_agent_t agent = _get_agent(0);
  // Available GPU agents

  current_features.clear();
  for (auto &metricName : metricNames) {
    current_features.push_back({ROCPROFILER_FEATURE_KIND_METRIC, metricName});
  }

  rocprofiler_start_queue_callbacks();
}

std::vector<double> measureMetricsStop() {
  if (context_queue.size() == 0) {
    std::cout << "measureMetricStop: no kernel kaunch was intercepted\n";
  }

  bool all_completed = false;
  while (!all_completed) {
    all_completed = true;
    for (auto &entry : context_queue) {
      all_completed &= entry.completed;
    }
    sched_yield();
  }

  rocprofiler_stop_queue_callbacks();

  std::vector<double> values;

  for (auto &entry : context_queue) {
    const std::string kernel_name = entry.data.kernel_name;
    const rocprofiler_dispatch_record_t *record = entry.data.record;

    // std::cout << kernel_name << "\n";

    for (auto &p : entry.features) {
      // std::cout << p.name << ": ";
      switch (p.data.kind) {
      // Output metrics results
      case ROCPROFILER_DATA_KIND_INT64:
        values.push_back((double)p.data.result_int64);
        // std::cout << values.back();
        break;
      case ROCPROFILER_DATA_KIND_DOUBLE:
        values.push_back((double)p.data.result_double);
        // std::cout << values.back();
        break;
      default:
        fprintf(stderr, "Undefined data kind(%u)\n", p.data.kind);
        abort();
      }
      // std::cout << "\n";
    }
  }
  context_queue.clear();
  return values;
}

void measureDRAMBytesStart() {
  char agent_name[64];
  hsa_agent_get_info(_get_agent(0), HSA_AGENT_INFO_NAME, &agent_name);
  if (agent_name[3] == '9') {
    measureMetricsStart({"FETCH_SIZE", "WRITE_SIZE"});
  } else {
    measureMetricsStart({"FETCH_SIZE"});
  }
}

std::vector<double> measureDRAMBytesStop() {
  auto values = measureMetricsStop();
  for (auto &v : values) {
    v *= 1024;
  }
  return values;
}

void measureL2BytesStart() {
  char agent_name[64];
  hsa_agent_get_info(_get_agent(0), HSA_AGENT_INFO_NAME, &agent_name);
  if (agent_name[3] == '9') {
    measureMetricsStart({"TCP_TCC_READ_REQ_sum", "TCP_TCC_WRITE_REQ_sum"});
  } else {
    measureMetricsStart({"GL2C_HIT_sum", "GL2C_MISS_sum"});
  }
}

std::vector<double> measureL2BytesStop() {
  auto values = measureMetricsStop();
  values[0] *= 64;
  values[1] *= 64;
  return values;
}
