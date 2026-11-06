// Ground-plane coverage of the union of image-space detection boxes.
#pragma once

#include "geometry/ipm.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <utility>
#include <vector>

struct ImageRect {
    double x1 = 0.0;
    double y1 = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
};

struct GroundCoverageMeasurement {
    bool valid = false;
    double frameAreaM2 = 0.0;
    double unionAreaM2 = 0.0;
    double coveragePercent = 0.0;
};

struct DirtAssessment {
    bool valid = false;
    double coveragePercent = 0.0;  // Smoothed, 0..100.
    int dirtLevel = 0;             // 1..5 when valid.
};

// Project all four corners of an image rectangle. Perspective maps its edges
// to straight ground-plane edges, so the shoelace formula gives its exact area.
inline bool projectedRectAreaM2(const IpmCalculator& ipm, const ImageRect& rect,
                                double* areaM2) {
    if (!areaM2 || !(rect.x1 < rect.x2) || !(rect.y1 < rect.y2)) return false;
    const std::pair<double, double> pixels[4] = {{rect.x1, rect.y1},
                                                 {rect.x2, rect.y1},
                                                 {rect.x2, rect.y2},
                                                 {rect.x1, rect.y2}};
    double groundX[4], groundY[4];
    for (int i = 0; i < 4; ++i) {
        float x = 0.0f, y = 0.0f;
        if (!ipm.imageToGround(static_cast<float>(pixels[i].first),
                               static_cast<float>(pixels[i].second), &x, &y) ||
            !std::isfinite(x) || !std::isfinite(y)) {
            return false;
        }
        groundX[i] = x;
        groundY[i] = y;
    }
    double twiceArea = 0.0;
    for (int i = 0; i < 4; ++i) {
        const int j = (i + 1) % 4;
        twiceArea += groundX[i] * groundY[j] - groundX[j] * groundY[i];
    }
    *areaM2 = std::abs(twiceArea) / 2.0;
    return std::isfinite(*areaM2);
}

inline GroundCoverageMeasurement measureGroundCoverage(
    const IpmCalculator& ipm, int frameWidth, int frameHeight,
    const std::vector<ImageRect>& boxes) {
    GroundCoverageMeasurement result;
    if (!ipm.isInitialized() || frameWidth <= 0 || frameHeight <= 0)
        return result;

    const ImageRect fullFrame{0.0, 0.0, static_cast<double>(frameWidth),
                              static_cast<double>(frameHeight)};
    if (!projectedRectAreaM2(ipm, fullFrame, &result.frameAreaM2) ||
        result.frameAreaM2 <= 0.0)
        return result;

    std::vector<ImageRect> clipped;
    std::vector<double> xEdges;
    clipped.reserve(boxes.size());
    xEdges.reserve(boxes.size() * 2);
    for (const auto& box : boxes) {
        if (!std::isfinite(box.x1) || !std::isfinite(box.y1) ||
            !std::isfinite(box.x2) || !std::isfinite(box.y2))
            return result;
        ImageRect rect{std::clamp(box.x1, 0.0, fullFrame.x2),
                       std::clamp(box.y1, 0.0, fullFrame.y2),
                       std::clamp(box.x2, 0.0, fullFrame.x2),
                       std::clamp(box.y2, 0.0, fullFrame.y2)};
        if (rect.x1 >= rect.x2 || rect.y1 >= rect.y2) continue;
        xEdges.push_back(rect.x1);
        xEdges.push_back(rect.x2);
        clipped.push_back(rect);
    }

    // Partition the image-space union into disjoint vertical strips. Because
    // the ground projection is one-to-one inside a valid full frame, their
    // projected areas sum to the exact ground-plane union area.
    std::sort(xEdges.begin(), xEdges.end());
    xEdges.erase(std::unique(xEdges.begin(), xEdges.end()), xEdges.end());
    for (size_t i = 1; i < xEdges.size(); ++i) {
        const double left = xEdges[i - 1], right = xEdges[i];
        std::vector<std::pair<double, double>> intervals;
        for (const auto& rect : clipped) {
            if (rect.x1 <= left && rect.x2 >= right)
                intervals.emplace_back(rect.y1, rect.y2);
        }
        std::sort(intervals.begin(), intervals.end());
        if (intervals.empty()) continue;

        double top = intervals.front().first;
        double bottom = intervals.front().second;
        auto addProjectedStrip = [&](double y1, double y2) {
            double area = 0.0;
            if (!projectedRectAreaM2(ipm, {left, y1, right, y2}, &area))
                return false;
            result.unionAreaM2 += area;
            return std::isfinite(result.unionAreaM2);
        };
        for (size_t j = 1; j < intervals.size(); ++j) {
            if (intervals[j].first <= bottom) {
                bottom = std::max(bottom, intervals[j].second);
            } else {
                if (!addProjectedStrip(top, bottom)) return result;
                top = intervals[j].first;
                bottom = intervals[j].second;
            }
        }
        if (!addProjectedStrip(top, bottom)) return result;
    }

    result.coveragePercent =
        std::clamp(100.0 * result.unionAreaM2 / result.frameAreaM2, 0.0, 100.0);
    result.valid = true;
    return result;
}

class CoverageSmoother {
   public:
    explicit CoverageSmoother(size_t windowSize = 5)
        : windowSize_(std::max<size_t>(windowSize, 1)) {}

    void reset() { samples_.clear(); }

    double add(double coveragePercent) {
        samples_.push_back(coveragePercent);
        if (samples_.size() > windowSize_) {
            samples_.pop_front();
        }
        // The window is small; recomputing avoids a negative rounding residue
        // after positive values leave a window of zero-valued samples.
        double sum = 0.0;
        for (const double sample : samples_) sum += sample;
        return sum / static_cast<double>(samples_.size());
    }

   private:
    size_t windowSize_;
    std::deque<double> samples_;
};
