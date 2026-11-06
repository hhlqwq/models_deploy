#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>

// YOLOX category IDs: liquid, debris, paper, cable, bottle.
constexpr std::size_t kDirtClassCount = 5;
using DirtClassCounts = std::array<std::size_t, kDirtClassCount>;

struct DirtScoreConfig {
    // Coverage is expressed in percentage points; the score is not a percent.
    double areaWeight = 2.0;
    std::array<double, kDirtClassCount> classWeights = {4.2, 4.5, 3.4, 3.0,
                                                        3.8};
    std::array<double, 4> levelMinimumScores = {3.5, 18.0, 35.0, 50.0};
};

inline DirtScoreConfig loadDirtScoreConfig(
    const std::map<std::string, std::string>& values) {
    DirtScoreConfig config;
    auto readNonnegative = [&](const std::string& key, double fallback) {
        const auto it = values.find(key);
        if (it == values.end()) return fallback;
        std::size_t parsed = 0;
        double value = 0.0;
        try {
            value = std::stod(it->second, &parsed);
        } catch (const std::exception&) {
            throw std::runtime_error(key +
                                     " must be a finite nonnegative number");
        }
        if (parsed != it->second.size() || !std::isfinite(value) ||
            value < 0.0) {
            throw std::runtime_error(key +
                                     " must be a finite nonnegative number");
        }
        return value;
    };

    config.areaWeight =
        readNonnegative("dirt_assessment.area_weight", config.areaWeight);
    constexpr const char* classKeys[kDirtClassCount] = {
        "weight_liquid", "weight_debris", "weight_paper", "weight_cable",
        "weight_bottle"};
    for (std::size_t i = 0; i < kDirtClassCount; ++i) {
        config.classWeights[i] =
            readNonnegative(std::string("dirt_assessment.") + classKeys[i],
                            config.classWeights[i]);
    }
    constexpr const char* levelKeys[4] = {
        "level_2_min_score", "level_3_min_score", "level_4_min_score",
        "level_5_min_score"};
    for (std::size_t i = 0; i < config.levelMinimumScores.size(); ++i) {
        config.levelMinimumScores[i] =
            readNonnegative(std::string("dirt_assessment.") + levelKeys[i],
                            config.levelMinimumScores[i]);
        if (config.levelMinimumScores[i] <=
            (i == 0 ? 0.0 : config.levelMinimumScores[i - 1])) {
            throw std::runtime_error(
                "dirt_assessment level minimum scores must be strictly increasing "
                "and level_2_min_score must be positive");
        }
    }
    return config;
}

inline double dirtScoreForFrame(const DirtScoreConfig& config,
                                double coveragePercent,
                                const DirtClassCounts& counts) {
    if (!std::isfinite(coveragePercent) || coveragePercent < 0.0 ||
        coveragePercent > 100.0) {
        throw std::runtime_error(
            "coverage percent must be finite and in [0, 100]");
    }
    double score = config.areaWeight * coveragePercent;
    for (std::size_t i = 0; i < kDirtClassCount; ++i) {
        score += config.classWeights[i] * static_cast<double>(counts[i]);
    }
    if (!std::isfinite(score)) throw std::runtime_error("dirt score overflow");
    return score;
}

inline int dirtLevelForScore(const DirtScoreConfig& config, double score) {
    if (!std::isfinite(score) || score < 0.0) {
        throw std::runtime_error("dirt score must be finite and nonnegative");
    }
    int level = 1;
    for (const double minimum : config.levelMinimumScores) {
        if (score < minimum) break;
        ++level;
    }
    return level;
}
