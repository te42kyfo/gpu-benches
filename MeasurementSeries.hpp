#pragma once
#include <algorithm>
#include <numeric>
#include <vector>

class MeasurementSeries {
public:
  void add(double v) { data.push_back(v); }
  double value() {
    if (data.size() == 0)
      return 0.0;
    if (data.size() == 1)
      return data[0];
    if (data.size() == 2)
      return (data[0] + data[1]) / 2.0;
    std::sort(begin(data), end(data));
    return std::accumulate(begin(data) + 1, end(data) - 1, 0.0) /
           (data.size() - 2);
  }
  double median() {
    if (data.size() == 0)
      return 0.0;
    if (data.size() == 1)
      return data[0];
    if (data.size() == 2)
      return (data[0] + data[1]) / 2.0;

    std::sort(begin(data), end(data));
    if (data.size() % 2 == 0) {
      return (data[data.size() / 2] + data[data.size() / 2 + 1]) / 2;
    }
    return data[data.size() / 2];
  }

  double minValue() {
    if (data.size() == 0)
      return 0.0;
    std::sort(begin(data), end(data));
    return *begin(data);
  }

  double maxValue() {
    if (data.size() == 0)
      return 0.0;
    std::sort(begin(data), end(data));
    return data.back();
  }
  double spread() {
    if (data.size() <= 1)
      return 0.0;
    if (data.size() == 2)
      return abs(data[0] - data[1]) / value();
    std::sort(begin(data), end(data));
    return abs(*(begin(data)) - *(end(data) - 1)) / value();
  }

private:
  std::vector<double> data;
};
