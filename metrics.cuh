/*
 * Modified 2017 by Dominik Ernst (dominik.ernst@fau.de)
 *
 * Derived from the
 * sample app to demonstrate use of CUPTI library to obtain metric values
 * using callbacks for CUDA runtime APIs
 * This software contains source code provided by NVIDIA Corporation.
 */

#include <cuda.h>
#include <cupti.h>
#include <cstdio>
#include <functional>
#include <iostream>

namespace {

bool abortMeasureMetric;

#define VERBOSE false

#define DRIVER_API_CALL(apiFuncCall)                                       \
  do {                                                                     \
    CUresult _status = apiFuncCall;                                        \
    if (_status != CUDA_SUCCESS) {                                         \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n", \
              __FILE__, __LINE__, #apiFuncCall, _status);                  \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                         \
  do {                                                                        \
    cudaError_t _status = apiFuncCall;                                        \
    if (_status != cudaSuccess) {                                             \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",    \
              __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status)); \
      exit(-1);                                                               \
    }                                                                         \
  } while (0)

#define CUPTI_CALL(call)                                                   \
  do {                                                                     \
    CUptiResult _status = call;                                            \
    if (_status != CUPTI_SUCCESS) {                                        \
      const char *errstr;                                                  \
      cuptiGetResultString(_status, &errstr);                              \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                          \
    }                                                                      \
  } while (0)

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                 \
  (((uintptr_t)(buffer) & ((align)-1))                              \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) \
       : (buffer))

// User data for event collection callback
typedef struct MetricData_st {
  // the device where metric is being collected
  CUdevice device;
  // the set of event groups to collect for a pass
  CUpti_EventGroupSet *eventGroups;
  // the current number of events collected in eventIdArray and
  // eventValueArray
  uint32_t eventIdx;
  // the number of entries in eventIdArray and eventValueArray
  uint32_t numEvents;
  // array of event ids
  CUpti_EventID *eventIdArray;
  // array of event values
  uint64_t *eventValueArray;
} MetricData_t;

static uint64_t kernelDuration;

void CUPTIAPI getMetricValueCallback(void *userdata,
                                     CUpti_CallbackDomain domain,
                                     CUpti_CallbackId cbid,
                                     const CUpti_CallbackData *cbInfo) {
  if (abortMeasureMetric) return;
  MetricData_t *metricData = (MetricData_t *)userdata;
  unsigned int i, j, k;

  // This callback is enabled only for launch so we shouldn't see
  // anything else.
  if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {
    printf("%s:%d: unexpected cbid %d\n", __FILE__, __LINE__, cbid);
    exit(-1);
  }

  // on entry, enable all the event groups being collected this pass,
  // for metrics we collect for all instances of the event
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cudaDeviceSynchronize();

    CUPTI_CALL(cuptiSetEventCollectionMode(cbInfo->context,
                                           CUPTI_EVENT_COLLECTION_MODE_KERNEL));

    for (i = 0; i < metricData->eventGroups->numEventGroups; i++) {
      uint32_t all = 1;
      CUPTI_CALL(cuptiEventGroupSetAttribute(
          metricData->eventGroups->eventGroups[i],
          CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(all),
          &all));
      CUPTI_CALL(
          cuptiEventGroupEnable(metricData->eventGroups->eventGroups[i]));
    }
  }

  // on exit, read and record event values
  if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    cudaDeviceSynchronize();

    // for each group, read the event values from the group and record
    // in metricData
    for (i = 0; i < metricData->eventGroups->numEventGroups; i++) {
      CUpti_EventGroup group = metricData->eventGroups->eventGroups[i];
      CUpti_EventDomainID groupDomain;
      uint32_t numEvents, numInstances, numTotalInstances;
      CUpti_EventID *eventIds;
      size_t groupDomainSize = sizeof(groupDomain);
      size_t numEventsSize = sizeof(numEvents);
      size_t numInstancesSize = sizeof(numInstances);
      size_t numTotalInstancesSize = sizeof(numTotalInstances);
      uint64_t *values, normalized, sum;
      size_t valuesSize, eventIdsSize;

      CUPTI_CALL(cuptiEventGroupGetAttribute(
          group, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, &groupDomainSize,
          &groupDomain));
      CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(
          metricData->device, groupDomain,
          CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT, &numTotalInstancesSize,
          &numTotalInstances));
      CUPTI_CALL(cuptiEventGroupGetAttribute(
          group, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &numInstancesSize,
          &numInstances));
      CUPTI_CALL(cuptiEventGroupGetAttribute(group,
                                             CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                                             &numEventsSize, &numEvents));
      eventIdsSize = numEvents * sizeof(CUpti_EventID);
      eventIds = (CUpti_EventID *)malloc(eventIdsSize);
      CUPTI_CALL(cuptiEventGroupGetAttribute(
          group, CUPTI_EVENT_GROUP_ATTR_EVENTS, &eventIdsSize, eventIds));

      valuesSize = sizeof(uint64_t) * numInstances;
      values = (uint64_t *)malloc(valuesSize);

      for (j = 0; j < numEvents; j++) {
        CUPTI_CALL(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE,
                                            eventIds[j], &valuesSize, values));

        if (metricData->eventIdx >= metricData->numEvents) {
          fprintf(stderr,
                  "error: too many events collected, metric expects only %d "
                  "instead of %d\n",
                  (int)metricData->numEvents, metricData->eventIdx);
          abortMeasureMetric = true;
          return;
        }

        // sum collect event values from all instances
        sum = 0;
        for (k = 0; k < numInstances; k++) sum += values[k];

        // normalize the event value to represent the total number of
        // domain instances on the device
        normalized = (sum * numTotalInstances) / numInstances;

        metricData->eventIdArray[metricData->eventIdx] = eventIds[j];
        metricData->eventValueArray[metricData->eventIdx] = normalized;
        metricData->eventIdx++;

        // print collected value
        {
          char eventName[128];
          size_t eventNameSize = sizeof(eventName) - 1;
          CUPTI_CALL(cuptiEventGetAttribute(eventIds[j], CUPTI_EVENT_ATTR_NAME,
                                            &eventNameSize, eventName));
          eventName[127] = '\0';
          if (VERBOSE) {
            printf("\t%s = %llu (", eventName, (unsigned long long)sum);
            if (numInstances > 1) {
              for (k = 0; k < numInstances; k++) {
                if (k != 0) printf(", ");
                printf("%llu", (unsigned long long)values[k]);
              }
            }

            printf(")\n");
            printf("\t%s (normalized) (%llu * %u) / %u = %llu\n", eventName,
                   (unsigned long long)sum, numTotalInstances, numInstances,
                   (unsigned long long)normalized);
          }
        }
      }

      free(values);
    }

    for (i = 0; i < metricData->eventGroups->numEventGroups; i++)
      CUPTI_CALL(
          cuptiEventGroupDisable(metricData->eventGroups->eventGroups[i]));
  }
}

static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size,
                                     size_t *maxNumRecords) {
  uint8_t *rawBuffer;

  *size = 16 * 1024;
  rawBuffer = (uint8_t *)malloc(*size + ALIGN_SIZE);

  *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
  *maxNumRecords = 0;

  if (*buffer == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }
}

static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId,
                                     uint8_t *buffer, size_t size,
                                     size_t validSize) {
  CUpti_Activity *record = NULL;
  CUpti_ActivityKernel3 *kernel;

  // since we launched only 1 kernel, we should have only 1 kernel record
  CUPTI_CALL(cuptiActivityGetNextRecord(buffer, validSize, &record));

  kernel = (CUpti_ActivityKernel3 *)record;
  if (kernel->kind != CUPTI_ACTIVITY_KIND_KERNEL) {
    fprintf(stderr, "Error: expected kernel activity record, got %d\n",
            (int)kernel->kind);
    exit(-1);
  }

  kernelDuration = kernel->end - kernel->start;
  if (VERBOSE) std::cout << "(Kernel Duration: " << kernelDuration << ")\n";
  free(buffer);
}

static CUcontext context = 0;

void measureMetricInit() {
  if (context == 0) {
    CUdevice device = 0;
    cuInit(0);
    DRIVER_API_CALL(cuDeviceGet(&device, 0));
    DRIVER_API_CALL(cuCtxCreate(&context, 0, device));
  }
}

double measureMetric(std::function<double()> runPass, const char *metricName) {
  abortMeasureMetric = false;

  CUpti_SubscriberHandle subscriber;
  CUdevice device = 0;
  DRIVER_API_CALL(cuDeviceGet(&device, 0));

  CUpti_MetricID metricId;
  CUpti_EventGroupSets *passData;
  MetricData_t metricData;
  unsigned int pass;
  CUpti_MetricValue metricValue;

  CUptiResult res = cuptiMetricGetIdFromName(device, metricName, &metricId);
  if (res != CUPTI_SUCCESS) return 0.0;

  // runPass();
  cudaDeviceSynchronize();
  // make sure activity is enabled before any CUDA API
  CUPTI_CALL(cuptiActivityEnableContext(context, CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(cuptiActivityFlushAll(0));
  // need to collect duration of kernel execution without any event
  // collection enabled (some metrics need kernel duration as part of
  // calculation). The only accurate way to do this is by using the
  // activity API.
  {
    CUPTI_CALL(
        cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    runPass();
    cudaDeviceSynchronize();
    CUPTI_CALL(cuptiActivityFlushAll(0));
  }
  CUPTI_CALL(cuptiActivityDisableContext(context, CUPTI_ACTIVITY_KIND_KERNEL));

  // setup launch callback for event collection
  CUPTI_CALL(cuptiSubscribe(
      &subscriber, (CUpti_CallbackFunc)getMetricValueCallback, &metricData));
  CUPTI_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                                 CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));

  // allocate space to hold all the events needed for the metric
  cuptiMetricGetIdFromName(device, metricName, &metricId);

  CUPTI_CALL(cuptiMetricGetNumEvents(metricId, &metricData.numEvents));
  metricData.device = device;
  metricData.eventIdArray =
      (CUpti_EventID *)malloc(metricData.numEvents * sizeof(CUpti_EventID));
  metricData.eventValueArray =
      (uint64_t *)malloc(metricData.numEvents * sizeof(uint64_t));
  metricData.eventIdx = 0;

  // get the number of passes required to collect all the events
  // needed for the metric and the event groups for each pass
  CUPTI_CALL(cuptiMetricCreateEventGroupSets(context, sizeof(metricId),
                                             &metricId, &passData));
  for (pass = 0; pass < passData->numSets; pass++) {
    if (VERBOSE) printf("Pass %u\n", pass);
    metricData.eventGroups = passData->sets + pass;
    runPass();
  }

  if (metricData.eventIdx != metricData.numEvents) {
    fprintf(stderr, "error: expected %u metric events, got %u\n",
            metricData.numEvents, metricData.eventIdx);
    return 0;
  }

  // use all the collected events to calculate the metric value
  if (!abortMeasureMetric)
    CUPTI_CALL(cuptiMetricGetValue(
        device, metricId, metricData.numEvents * sizeof(CUpti_EventID),
        metricData.eventIdArray, metricData.numEvents * sizeof(uint64_t),
        metricData.eventValueArray, kernelDuration, &metricValue));

  double val = 0.0;
  // print metric value, we format based on the value kind
  if (!abortMeasureMetric) {
    CUpti_MetricValueKind valueKind;
    size_t valueKindSize = sizeof(valueKind);
    CUPTI_CALL(cuptiMetricGetAttribute(metricId, CUPTI_METRIC_ATTR_VALUE_KIND,
                                       &valueKindSize, &valueKind));
    switch (valueKind) {
      case CUPTI_METRIC_VALUE_KIND_DOUBLE:
        if (VERBOSE)
          printf("Metric %s = %f\n", metricName, metricValue.metricValueDouble);
        val = metricValue.metricValueDouble;
        break;
      case CUPTI_METRIC_VALUE_KIND_UINT64:
        if (VERBOSE)
          printf("Metric %s = %llu\n", metricName,
                 (unsigned long long)metricValue.metricValueUint64);
        val = metricValue.metricValueUint64;
        break;
      case CUPTI_METRIC_VALUE_KIND_INT64:
        if (VERBOSE)
          printf("Metric %s = %lld\n", metricName,
                 (long long)metricValue.metricValueInt64);
        val = metricValue.metricValueInt64;
        break;
      case CUPTI_METRIC_VALUE_KIND_PERCENT:
        if (VERBOSE)
          printf("Metric %s = %f%%\n", metricName,
                 metricValue.metricValuePercent);
        val = metricValue.metricValuePercent;
        break;
      case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
        if (VERBOSE)
          printf("Metric %s = %llu bytes/sec\n", metricName,
                 (unsigned long long)metricValue.metricValueThroughput);
        val = metricValue.metricValueThroughput;
        break;
      case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
        if (VERBOSE)
          printf("Metric %s = utilization level %u\n", metricName,
                 (unsigned int)metricValue.metricValueUtilizationLevel);
        val = (unsigned int)metricValue.metricValueUtilizationLevel;
        break;
      default:
        fprintf(stderr, "error: unknown value kind\n");
        return 0;
    }
  }

  CUPTI_CALL(cuptiUnsubscribe(subscriber));
  // cuCtxDestroy(context);
  return val;
}
}
