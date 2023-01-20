// $ cat makefile
// CUPTI = /opt/cuda/extras/CUPTI
//
// all: cupti
//
// cupti: cupti.cu
//	nvcc -I$(CUPTI)/include -arch=sm_30 $< -o $@ -L$(CUPTI)/lib64 -lcupti
//-Xlinker -rpath=$(CUPTI)/lib64
//
// clean:
//	rm -rf cupti

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cupti.h>

#define BUF_SIZE (32 * 1024)
#define ALIGNMENT 8

#define CUPTI_CHECKED_CALL(x)                                                  \
  do {                                                                         \
    CUptiResult err = x;                                                       \
    if ((err) != CUPTI_SUCCESS) {                                              \
      const char *errstr;                                                      \
      cuptiGetResultString(err, &errstr);                                      \
      printf("Error \"%s\" at %s:%d\n", errstr, __FILE__, __LINE__);           \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

void CUPTIAPI allocBuffer(uint8_t **buffer, size_t *size,
                          size_t *maxNumRecords) {
  if (posix_memalign((void **)buffer, ALIGNMENT, BUF_SIZE) != 0) {
    fprintf(stderr, "Error: out of memory\n");
    exit(-1);
  }

  *size = BUF_SIZE;
  *maxNumRecords = 0;
}

void CUPTIAPI freeBuffer(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                         size_t size, size_t validSize) {
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_ENVIRONMENT) {
          CUpti_ActivityEnvironment *env = (CUpti_ActivityEnvironment *)record;

          switch (env->environmentKind) {
          case CUPTI_ACTIVITY_ENVIRONMENT_SPEED:
            printf("SPEED\n");
            printf("\tsmClock = %d\n", env->data.speed.smClock);
            printf("\tmemoryClock = %d\n", env->data.speed.memoryClock);
            break;
          case CUPTI_ACTIVITY_ENVIRONMENT_TEMPERATURE:
            printf("TEMPERATURE = %d C\n",
                   env->data.temperature.gpuTemperature);
            break;
          case CUPTI_ACTIVITY_ENVIRONMENT_POWER:
            printf("POWER\n");
            break;
          case CUPTI_ACTIVITY_ENVIRONMENT_COOLING:
            printf("COOLING\n");
            break;
          default:
            printf("<unknown>\n");
            break;
          }
        }
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CHECKED_CALL(status);
      }
    } while (1);

    // Report any records dropped from the queue
    size_t dropped;
    CUPTI_CHECKED_CALL(
        cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      printf("Dropped %u activity records\n", (unsigned int)dropped);
    }
  }

  free(buffer);
}

double get_clock() {
  CUPTI_CHECKED_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_ENVIRONMENT));
  CUPTI_CHECKED_CALL(cuptiActivityRegisterCallbacks(allocBuffer, freeBuffer));

  cudaDeviceSynchronize();
  cuptiActivityFlushAll(0);
  CUPTI_CHECKED_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_ENVIRONMENT));
  return 1380;
}
