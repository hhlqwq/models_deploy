#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// Minimal protocol inputs keep this geometry/protocol test independent of the
// inference SDK and ROS runtime.
struct TestBBox { float x1, y1, x2, y2; };
struct TestInstance { TestBBox bbox; int type_id; float confidence; };
struct YoloxResult { std::vector<TestInstance> result; };
struct RuntimeMetrics {
  unsigned long long frameCount = 0;
  double endToEndFps = 0.0;
  double inferenceFps = 0.0;
};
inline const char* garbageClassName(int) { return "debris"; }

#include "shared/reporting/perception_result_protocol.hpp"

void expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

void expectNear(double actual, double expected, double tolerance,
                const char* message) {
  expect(std::abs(actual - expected) <= tolerance, message);
}

int main() {
  IpmConfig cfg;
  cfg.camera_height_m = 1.05f;
  cfg.pitch_deg = 45.0f;
  IpmCalculator ipm;
  expect(ipm.init(cfg, 608.8793945f, 608.7294922f, 631.3320923f,
                  356.6316833f, 1280, 720), "IPM initialization");

  const auto empty = measureGroundCoverage(ipm, 1280, 720, {});
  expect(empty.valid, "empty detections must have valid IPM coverage");
  expectNear(empty.frameAreaM2, 17.8257, 0.01, "full-frame ground area");
  expectNear(empty.coveragePercent, 0.0, 1e-9, "empty detections are clean");

  const auto full = measureGroundCoverage(ipm, 1280, 720,
                                          {{0, 0, 1280, 720}});
  expect(full.valid, "full-frame detection");
  expectNear(full.coveragePercent, 100.0, 1e-5, "full-frame coverage");

  const auto overlap = measureGroundCoverage(
      ipm, 1280, 720, {{100, 100, 300, 300}, {200, 100, 400, 300}});
  const auto outer = measureGroundCoverage(ipm, 1280, 720,
                                            {{100, 100, 400, 300}});
  expect(overlap.valid && outer.valid, "overlapping boxes project");
  expectNear(overlap.unionAreaM2, outer.unionAreaM2, 1e-5,
             "overlap counted only once");

  const auto diagonalOverlap = measureGroundCoverage(
      ipm, 1280, 720, {{100, 100, 300, 300}, {200, 200, 400, 400}});
  const auto firstBox = measureGroundCoverage(ipm, 1280, 720,
                                               {{100, 100, 300, 300}});
  const auto secondBox = measureGroundCoverage(ipm, 1280, 720,
                                                {{200, 200, 400, 400}});
  const auto intersection = measureGroundCoverage(ipm, 1280, 720,
                                                   {{200, 200, 300, 300}});
  expectNear(diagonalOverlap.unionAreaM2,
             firstBox.unionAreaM2 + secondBox.unionAreaM2 -
                 intersection.unionAreaM2,
             1e-5, "two-dimensional overlap counted only once");

  const auto clipped = measureGroundCoverage(ipm, 1280, 720,
                                              {{-100, -100, 200, 200}});
  const auto inside = measureGroundCoverage(ipm, 1280, 720,
                                             {{0, 0, 200, 200}});
  expectNear(clipped.unionAreaM2, inside.unionAreaM2, 1e-5,
             "boxes clipped to the image");

  const auto farBox = measureGroundCoverage(ipm, 1280, 720,
                                             {{500, 50, 600, 150}});
  const auto nearBox = measureGroundCoverage(ipm, 1280, 720,
                                              {{500, 550, 600, 650}});
  expect(farBox.unionAreaM2 > nearBox.unionAreaM2,
         "equal pixel boxes have different physical areas by depth");

  IpmCalculator missingIpm;
  expect(!measureGroundCoverage(missingIpm, 1280, 720, {}).valid,
         "missing IPM must not report zero coverage");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  expect(!measureGroundCoverage(ipm, 1280, 720,
                                {{nan, 0, 100, 100}}).valid,
         "non-finite detection coordinates invalidate coverage");

  CoverageSmoother smoother(5);
  for (double value : {0.0, 10.0, 20.0, 30.0, 40.0}) smoother.add(value);
  expectNear(smoother.add(50.0), 30.0, 1e-9,
             "five-frame moving average drops the oldest frame");
  smoother.reset();
  expectNear(smoother.add(5.0), 5.0, 1e-9,
             "calibration reset clears smoothing history");

  CoverageSmoother scoreSmoother(5);
  for (double score : {18.0, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0})
    scoreSmoother.add(score);
  expect(scoreSmoother.add(0.0) == 0.0,
         "zero-valued score window has no negative rounding residue");

  const YoloxResult detections{};
  const RuntimeMetrics metrics{};
  const perception_protocol::ResultSource source{};
  const auto validJson = perception_protocol::encodeResult(
      1, 1, 1280, 720, 0, detections,
      perception_protocol::GroundClassification{0, 0.93f},
      metrics, &ipm, source, {true, 2.5, 2});
  expect(validJson.find("\"coverage_percent\":2.5") != std::string::npos,
         "coverage percent appears in JSON");
  expect(validJson.find("\"dirt_level\":2") != std::string::npos,
         "dirt level appears in JSON");
  expect(validJson.find("\"ground_classification\":{\"category_code\":\"industrial_floor\",\"confidence\":0.93}") != std::string::npos,
         "industrial floor classification appears in JSON");
  const auto stoneJson = perception_protocol::encodeResult(
      3, 3, 1280, 720, 0, detections,
      perception_protocol::GroundClassification{1, 0.94f},
      metrics, &ipm, source, {true, 2.5, 2});
  expect(stoneJson.find("\"category_code\":\"stone_floor\"") != std::string::npos,
         "stone floor classification appears in JSON");
  const auto missingJson = perception_protocol::encodeResult(
      2, 2, 1280, 720, 0, detections, std::nullopt,
      metrics, nullptr, source, {});
  expect(missingJson.find("\"ground_classification\":null") != std::string::npos,
         "classification failure is null in JSON");
  expect(missingJson.find("\"coverage_percent\":null") != std::string::npos &&
         missingJson.find("\"dirt_level\":null") != std::string::npos &&
         missingJson.find("IPM info missing") != std::string::npos,
         "missing IPM is explicit in JSON");

  std::vector<ImageRect> manyBoxes;
  for (int i = 0; i < 50; ++i) {
    const double x = (i * 23) % 1200;
    const double y = (i * 31) % 630;
    manyBoxes.push_back({x, y, x + 80, y + 60});
  }
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; ++i)
    expect(measureGroundCoverage(ipm, 1280, 720, manyBoxes).valid,
           "50-box coverage remains valid");
  const auto elapsed = std::chrono::steady_clock::now() - start;
  const double meanMs =
      std::chrono::duration<double, std::milli>(elapsed).count() / 100.0;

  std::cout << "ground_coverage_test passed; 50-box mean " << meanMs
            << " ms/frame\n";
  return 0;
}
