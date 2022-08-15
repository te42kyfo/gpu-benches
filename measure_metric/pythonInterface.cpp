#include "measureMetricPW.hpp"

#include <Python.h>

extern "C" PyObject *measureMetricStop() {

  runTestEnd();

  /*CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {
      CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
*/
  auto values = NV::Metric::Eval::GetMetricValues(chipName, counterDataImage,
                                                  metricNames);

  PyGILState_STATE gstate = PyGILState_Ensure();


  PyObject *result = PyList_New(0);
  for (auto value : values) {
    PyList_Append(result, PyFloat_FromDouble(value));
  }

  PyGILState_Release(gstate);

  return result;
}
