// Linux 应用入口: Orbbec/ROS 2 图像、双模型感知与结果输出.
#include "assessment/dirt_score.hpp"
#ifndef ASYNC_PERCEPTION_MTK_G720
#include "platform/linux/ros2/ros2_image_subscriber.hpp"
#endif
#ifndef ASYNC_PERCEPTION_NO_ROS2
#include "platform/linux/ros2/ros2_json_publisher.hpp"
#else
#include <csignal>
#endif
#include "platform/linux/web_preview_server.hpp"
#include "shared/core/perception_foundation.hpp"
#include "shared/session/perception_session.hpp"
#include <cstdlib>
#include <map>
#include <optional>

namespace {

#ifdef ASYNC_PERCEPTION_NO_ROS2
volatile std::sig_atomic_t signalStopRequested = 0;

// 仅设置退出标志, 资源回收由主线程完成.
void requestSignalStop(int) { signalStopRequested = 1; }
#endif

std::map<std::string, std::string> loadIni(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open configuration: " + path.string());
    std::map<std::string, std::string> values;
    std::string section;
    for (std::string line; std::getline(input, line);) {
        if (line.empty() || line.front() == '#' || line.front() == ';')
            continue;
        if (line.front() == '[' && line.back() == ']') {
            section = line.substr(1, line.size() - 2);
            continue;
        }
        const auto sep = line.find('=');
        if (sep != std::string::npos)
            values[section + "." + line.substr(0, sep)] = line.substr(sep + 1);
    }
    return values;
}

bool readBool(const std::map<std::string, std::string>& v, const std::string& k,
              bool def) {
    auto it = v.find(k);
    return it == v.end() ? def : (it->second == "true" || it->second == "1");
}

int readInt(const std::map<std::string, std::string>& values,
            const std::string& key, int defaultValue, int minimum,
            int maximum) {
    const auto iterator = values.find(key);
    if (iterator == values.end() || iterator->second.empty())
        return defaultValue;
    size_t parsed = 0;
    const int value = std::stoi(iterator->second, &parsed);
    if (parsed != iterator->second.size() || value < minimum ||
        value > maximum) {
        throw std::runtime_error(key + " must be an integer in [" +
                                 std::to_string(minimum) + ", " +
                                 std::to_string(maximum) + "]");
    }
    return value;
}

std::string requireValue(const std::map<std::string, std::string>& v,
                         const std::string& k) {
    auto it = v.find(k);
    if (it == v.end() || it->second.empty())
        throw std::runtime_error("missing: " + k);
    return it->second;
}

std::string optionalValue(const std::map<std::string, std::string>& v,
                          const std::string& k) {
    auto it = v.find(k);
    return it == v.end() ? "" : it->second;
}

}  // namespace

int runLinuxApplication(int argc, char** argv) {
    const std::filesystem::path executable = std::filesystem::absolute(argv[0]);
    const std::filesystem::path config =
        argc > 2 && std::string(argv[1]) == "--config"
            ? std::filesystem::path(argv[2])
            : executable.parent_path() / "config/async_perception.ini";
    try {
        const auto values = loadIni(config);
        const std::string mode = requireValue(values, "application.mode");
        if (mode != "playback" && mode != "live" && mode != "ros")
            throw std::runtime_error(
                "application.mode must be playback, live or ros");
#ifdef ASYNC_PERCEPTION_MTK_G720
        if (mode == "ros") {
            throw std::runtime_error(
                "G720 requires application.mode=live or playback");
        }
#endif

        const std::string sourcePrefix = mode + ".";
        RunOptions options;
        options.outputDirectory =
            requireValue(values, sourcePrefix + "output_directory");
        options.liveDeviceUid =
            mode == "live" ? optionalValue(values, "live.device_uid") : "";
        if (mode == "playback")
            options.bagPath = requireValue(values, "playback.bag_path");

#ifndef ASYNC_PERCEPTION_MTK_G720
        const std::string asyncInferConfig =
            (std::filesystem::path(executable.parent_path()) / "config" /
             "0_root_config.json")
                .string();
#endif

#ifndef ASYNC_PERCEPTION_NO_ROS2
        std::atomic_bool stopRequested{false};
#endif
        std::atomic_bool livePaused{false};
#ifdef ASYNC_PERCEPTION_NO_ROS2
        const bool emitJsonLines = true;
#else
        const bool emitJsonLines =
            readBool(values, "reporting.enable_json_lines", false);
#endif
        perception_protocol::ResultSource resultSource;
        const std::string sensorId =
            optionalValue(values, "reporting.sensor_id");
        const std::string pipelineId =
            optionalValue(values, "reporting.pipeline_id");
        if (!sensorId.empty()) resultSource.sensorId = sensorId;
        if (!pipelineId.empty()) resultSource.pipelineId = pipelineId;
        const bool webPreviewEnabled =
            readBool(values, "web_preview.enabled", true);
        bool webPreviewActive = false;
        WebPreviewServer webPreview;
        if (webPreviewEnabled) {
            const std::string bindAddress =
                optionalValue(values, "web_preview.bind_address").empty()
                    ? "0.0.0.0"
                    : optionalValue(values, "web_preview.bind_address");
            const int port =
                readInt(values, "web_preview.port", 8080, 1, 65535);
            const int jpegQuality =
                readInt(values, "web_preview.jpeg_quality", 80, 10, 100);
            const int maxFps =
                readInt(values, "web_preview.max_fps", 10, 1, 30);
            std::string previewError;
            if (!webPreview.start(bindAddress, static_cast<uint16_t>(port),
                                  jpegQuality, maxFps, &previewError)) {
                std::cerr << "web preview disabled: " << previewError << '\n';
            } else {
                webPreviewActive = true;
                std::cout << "web preview: http://" << bindAddress << ':'
                          << port << "/\n";
            }
        }

        // IPM 配置
        IpmConfig ipmCfg = loadIpmConfig(values);
        IpmCalculator ipm;
        const DirtScoreConfig dirtScoreCfg = loadDirtScoreConfig(values);
        CoverageSmoother coverageSmoother(5);
        CoverageSmoother dirtScoreSmoother(5);

#ifndef ASYNC_PERCEPTION_NO_ROS2
#ifndef ASYNC_PERCEPTION_MTK_G720
        // 容器 IPC 隔离时, ROS 图像输入默认采用 UDP.
        if (mode == "ros" && !std::getenv("FASTDDS_BUILTIN_TRANSPORTS")) {
            ::setenv("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4", 1);
        }
#endif
        RosPublisher rosPublisher(argc, argv);
        rclcpp::on_shutdown([&stopRequested] { stopRequested.store(true); });
#else
        std::signal(SIGINT, requestSignalStop);
        std::signal(SIGTERM, requestSignalStop);
#endif

        std::cout << "async_perception " << ASYNC_PERCEPTION_VERSION
                  << " Linux console mode\n";
#ifdef ASYNC_PERCEPTION_MTK_G720
        initializeNeuronInference(executable.parent_path() / "models");
        std::cout << "backend: MediaTek Neuron Runtime (MDLA)\n";
#else
        // 初始化 Intel SDK.
        const auto initStatus = async_infer_init(asyncInferConfig.c_str());
        if (initStatus != AIS_OK) {
            std::cerr << "async_infer_init failed: "
                      << async_infer_err_name(initStatus) << " - "
                      << async_infer_err_desc(initStatus) << std::endl;
            return 1;
        }
        const auto sdkInfo = async_infer_get_info();
        std::cout << "backend: Intel OpenVINO (async_infer "
                  << (sdkInfo.version ? sdkInfo.version : "?") << ")\n";
#endif
        std::cout << "source mode: " << mode << "\n";
        std::cout << "output directory: " << options.outputDirectory << "\n";
#ifndef ASYNC_PERCEPTION_NO_ROS2
        std::cout
            << "ROS2: node=/async_perception_node topic=/perception/result\n";
#else
        std::cout << "reporting: JSON Lines (G720_ENABLE_ROS2=OFF)\n";
#endif

        std::shared_ptr<ob::Context> context;
        std::shared_ptr<ob::Pipeline> pipeline;
        std::shared_ptr<ob::FormatConvertFilter> converter;
        bool ipmReady = false;

#ifndef ASYNC_PERCEPTION_MTK_G720
        std::unique_ptr<RosImageSubscriber> rosSubscriber;
        if (mode == "ros") {
            const auto imageTopic = requireValue(values, "ros.image_topic");
            const auto cameraInfoTopic =
                requireValue(values, "ros.camera_info_topic");
            const auto queueCapacity =
                readInt(values, "ros.queue_capacity", 2, 1, 8);
            rosSubscriber = std::make_unique<RosImageSubscriber>(
                imageTopic, cameraInfoTopic,
                static_cast<size_t>(queueCapacity));
        } else
#endif
        {
            // Orbbec pipeline setup
#ifndef ASYNC_PERCEPTION_MTK_G720
            if (mode == "playback") {
#endif
                const std::string extensionsDirectory =
                    (executable.parent_path() / "extensions").string();
                ob::Context::setExtensionsDirectory(
                    extensionsDirectory.c_str());
#ifndef ASYNC_PERCEPTION_MTK_G720
            }
#endif
            context = std::make_shared<ob::Context>();

            if (mode == "playback") {
                auto playback =
                    std::make_shared<ob::PlaybackDevice>(options.bagPath);
                pipeline = std::make_shared<ob::Pipeline>(playback);
            } else {
                auto deviceList = context->queryDeviceList();
                std::shared_ptr<ob::Device> device;
                if (deviceList && deviceList->getCount() > 0) {
                    for (uint32_t i = 0; i < deviceList->getCount(); ++i) {
                        auto dev = deviceList->getDevice(i);
                        if (!options.liveDeviceUid.empty()) {
                            if (dev->getDeviceInfo()->uid() ==
                                options.liveDeviceUid) {
                                device = dev;
                                break;
                            }
                        } else {
                            device = dev;
                            break;
                        }
                    }
                }
                if (!device) {
                    std::cerr << "No Orbbec device found." << std::endl;
                    releasePerceptionInference();
                    return 1;
                }
                pipeline = std::make_shared<ob::Pipeline>(device);
            }

            if (mode == "playback") {
                auto config =
                    buildPerceptionStreamConfig(pipeline->getDevice());
                try {
                    pipeline->enableFrameSync();
                } catch (...) {
                }
                pipeline->start(config);
            } else {
#ifdef ASYNC_PERCEPTION_MTK_G720
                auto streamConfig = std::make_shared<ob::Config>();
                streamConfig->enableVideoStream(OB_SENSOR_COLOR, 640, 360, 30,
                                                OB_FORMAT_MJPG);
                std::cout << "Orbbec color: 640x360 MJPEG @ 30 fps\n";
                pipeline->start(streamConfig);
#else
                auto streamConfig =
                    buildPerceptionStreamConfig(pipeline->getDevice());
                pipeline->start(streamConfig);
#endif
            }

            // 从 Orbbec SDK 获取相机内参并初始化 IPM
            try {
                auto camParam = pipeline->getCameraParam();
                const auto& rgb = camParam.rgbIntrinsic;
                if (
#ifdef ASYNC_PERCEPTION_MTK_G720
                    (mode != "live" ||
                     (rgb.width == 640 && rgb.height == 360)) &&
#endif
                    ipm.init(ipmCfg, rgb.fx, rgb.fy, rgb.cx, rgb.cy, rgb.width,
                             rgb.height)) {
                    std::cout << "IPM initialized (SDK): fx=" << rgb.fx
                              << " fy=" << rgb.fy << " cx=" << rgb.cx
                              << " cy=" << rgb.cy << " img=" << rgb.width << "x"
                              << rgb.height
                              << " height=" << ipmCfg.camera_height_m << "m"
                              << " pitch=" << ipmCfg.pitch_deg << "deg"
                              << std::endl;
                    ipmReady = true;
                }
            } catch (const ob::Error& e) {
                std::cerr << "getCameraParam failed: " << e.what() << std::endl;
            }
            // SDK 失败时，尝试从配置文件手动内参
            if (!ipmReady &&
#ifdef ASYNC_PERCEPTION_MTK_G720
                (mode != "live" ||
                 (ipmCfg.manual_img_w == 640 && ipmCfg.manual_img_h == 360)) &&
#endif
                ipmCfg.manual_fx > 0 && ipmCfg.manual_fy > 0 &&
                ipmCfg.manual_img_w > 0 && ipmCfg.manual_img_h > 0) {
                if (ipm.init(ipmCfg, ipmCfg.manual_fx, ipmCfg.manual_fy,
                             ipmCfg.manual_cx, ipmCfg.manual_cy,
                             ipmCfg.manual_img_w, ipmCfg.manual_img_h)) {
                    std::cout
                        << "IPM initialized (manual): fx=" << ipmCfg.manual_fx
                        << " fy=" << ipmCfg.manual_fy
                        << " cx=" << ipmCfg.manual_cx
                        << " cy=" << ipmCfg.manual_cy
                        << " img=" << ipmCfg.manual_img_w << "x"
                        << ipmCfg.manual_img_h
                        << " height=" << ipmCfg.camera_height_m << "m"
                        << " pitch=" << ipmCfg.pitch_deg << "deg" << std::endl;
                    ipmReady = true;
                }
            }
            if (!ipmReady) {
                std::cerr
                    << "IPM disabled - set ipm.fx/fy/cx/cy/img_width/img_height in INI"
                    << std::endl;
            }

            // Both live and playback may yield compressed or non-RGB color frames.
            // ensureRgbFrame() requires a converter even when the frame is already RGB.
            converter = std::make_shared<ob::FormatConvertFilter>();
        }
        uint64_t processedFrames = 0;
        uint64_t sequenceId = 0;
        RateCounter inferenceRate, endToEndRate;
        CpuMeter cpuMeter;
        RuntimeMetrics metrics;
#ifndef ASYNC_PERCEPTION_MTK_G720
        RosCameraCalibration lastCalibration;
#endif

        std::cout << "Streaming started. Press Ctrl+C to stop.\n";

#ifdef ASYNC_PERCEPTION_NO_ROS2
        while (!signalStopRequested) {
#else
        while (rclcpp::ok() && !stopRequested.load()) {
#endif
            cv::Mat original;
            uint64_t tsMs = 0;
            bool frameIpmReady = ipmReady;
#ifndef ASYNC_PERCEPTION_MTK_G720
            if (rosSubscriber) {
                RosImageFrame frame;
                if (!rosSubscriber->waitForFrame(
                        &frame, std::chrono::milliseconds(500)))
                    continue;
                RosCameraCalibration calibration;
                frameIpmReady =
                    rosSubscriber->calibrationFor(frame, &calibration);
                original = std::move(frame.bgr);
                tsMs = frame.captureTimestampMs;
                if (frameIpmReady &&
                    (!ipmReady || calibration.width != lastCalibration.width ||
                     calibration.height != lastCalibration.height ||
                     calibration.fx != lastCalibration.fx ||
                     calibration.fy != lastCalibration.fy ||
                     calibration.cx != lastCalibration.cx ||
                     calibration.cy != lastCalibration.cy)) {
                    ipmReady = ipm.init(ipmCfg, calibration.fx, calibration.fy,
                                        calibration.cx, calibration.cy,
                                        calibration.width, calibration.height);
                    lastCalibration = calibration;
                    coverageSmoother.reset();
                    dirtScoreSmoother.reset();
                    std::cout << "IPM initialized (ROS CameraInfo): "
                              << calibration.width << 'x' << calibration.height
                              << std::endl;
                }
            } else
#endif
            {
                std::shared_ptr<ob::FrameSet> frames;
                try {
                    frames = pipeline->waitForFrames(2000);
                } catch (const ob::Error& e) {
                    std::cerr << "[DEBUG] waitForFrames error: " << e.what()
                              << std::endl;
                    continue;
                }
                if (!frames) continue;
                auto colorFrame = frames->colorFrame();
                if (!colorFrame) continue;
                auto rgbFrame = ensureRgbFrame(colorFrame, converter);
                if (!rgbFrame) continue;
                auto videoFrame = rgbFrame->as<ob::VideoFrame>();
                const int width = videoFrame->getWidth();
                const int height = videoFrame->getHeight();
                cv::Mat rgb(height, width, CV_8UC3,
                            const_cast<uint8_t*>(videoFrame->getData()));
                cv::cvtColor(rgb, original, cv::COLOR_RGB2BGR);
                tsMs = colorFrame->getTimeStampUs() / 1000;
            }

            const int w = original.cols;
            const int h = original.rows;

            // 同一帧提交检测和地面分类.
            auto frameStart = std::chrono::steady_clock::now();
            std::string inferError;
            auto pending = submitYolox(original, &inferError);
            if (!pending) {
                gLogQueue.push("submit failed: " + inferError);
                continue;
            }

            // Submit both tasks for the same BGR frame before waiting for either one.
            std::string classificationError;
            auto classificationPending =
                submitMobilenetV2(original, &classificationError);
            if (!classificationPending) {
                gLogQueue.push(
                    "MobileNetV2 submit failed: " + classificationError,
                    kLogWarn);
            }

            if (!waitInference(pending, &inferError)) {
                gLogQueue.push("YOLOX inference wait failed: " + inferError,
                               kLogWarn);
                continue;
            }

            std::optional<perception_protocol::GroundClassification>
                groundClassification;
            if (classificationPending) {
                if (waitInference(classificationPending,
                                  &classificationError)) {
                    const auto& classification =
                        classificationPending->result.mobilenetv2;
                    if (classificationPending->result.task_type ==
                            TID_MOBILENETV2 &&
                        (classification.class_id == 0 ||
                         classification.class_id == 1) &&
                        std::isfinite(classification.confidence) &&
                        classification.confidence >= 0.0f &&
                        classification.confidence <= 1.0f) {
                        groundClassification =
                            perception_protocol::GroundClassification{
                                classification.class_id,
                                classification.confidence};
                    } else {
                        classificationError =
                            "invalid MobileNetV2 class or confidence";
                    }
                }
                if (!groundClassification) {
                    gLogQueue.push(
                        "MobileNetV2 inference failed: " + classificationError,
                        kLogWarn);
                }
            }
            inferenceRate.add(std::chrono::steady_clock::now() - frameStart);

            // Build visual overlay (optional)
            cv::Mat visual = original.clone();
            drawYoloxDetections(&visual, pending->result.yolox);

            ++processedFrames;
            ++sequenceId;
            endToEndRate.add(std::chrono::steady_clock::now() - frameStart);
            metrics.inferenceFps = inferenceRate.fps();
            metrics.inferenceMs = inferenceRate.milliseconds();
            metrics.endToEndFps = endToEndRate.fps();
            metrics.endToEndMs = endToEndRate.milliseconds();
            metrics.frameCount = processedFrames;
            metrics.cpuPercent = cpuMeter.sample();

            if (webPreviewActive) {
                cv::Mat previewFrame;
                cv::hconcat(original, visual, previewFrame);
                constexpr int kMaxPreviewWidth = 1600;
                if (previewFrame.cols > kMaxPreviewWidth) {
                    const double scale = static_cast<double>(kMaxPreviewWidth) /
                                         previewFrame.cols;
                    cv::resize(previewFrame, previewFrame, cv::Size(), scale,
                               scale, cv::INTER_AREA);
                }
                WebPreviewMetrics previewMetrics;
                previewMetrics.inferenceFps = metrics.inferenceFps;
                previewMetrics.inferenceMs = metrics.inferenceMs;
                previewMetrics.endToEndFps = metrics.endToEndFps;
                previewMetrics.cpuPercent = metrics.cpuPercent;
                previewMetrics.frameCount = metrics.frameCount;
                webPreview.publish(previewFrame, previewMetrics);
            }

            // Encode and publish JSON
            const IpmCalculator* frameIpm =
                frameIpmReady && ipm.isInitialized() ? &ipm : nullptr;
            DirtAssessment assessment;
            if (frameIpm) {
                std::vector<ImageRect> boxes;
                boxes.reserve(pending->result.yolox.result.size());
                DirtClassCounts classCounts{};
                for (const auto& instance : pending->result.yolox.result) {
                    boxes.push_back({instance.bbox.x1, instance.bbox.y1,
                                     instance.bbox.x2, instance.bbox.y2});
                    if (instance.type_id >= 0 &&
                        instance.type_id < static_cast<int>(kDirtClassCount)) {
                        ++classCounts[static_cast<std::size_t>(
                            instance.type_id)];
                    }
                }
                const auto coverage =
                    measureGroundCoverage(*frameIpm, w, h, boxes);
                if (coverage.valid) {
                    assessment.valid = true;
                    assessment.coveragePercent =
                        coverageSmoother.add(coverage.coveragePercent);
                    const double frameScore = dirtScoreForFrame(
                        dirtScoreCfg, coverage.coveragePercent, classCounts);
                    assessment.dirtLevel = dirtLevelForScore(
                        dirtScoreCfg, dirtScoreSmoother.add(frameScore));
                }
            }
            const std::string json = perception_protocol::encodeResult(
                sequenceId, static_cast<int>(processedFrames), w, h, tsMs,
                pending->result.yolox, groundClassification, metrics, frameIpm,
                resultSource, assessment);
            if (emitJsonLines) std::cout << json << '\n';
#ifndef ASYNC_PERCEPTION_NO_ROS2
            rosPublisher.publish(json);
            rosPublisher.spinSome();
#endif

            if (processedFrames % 30 == 0) {
                std::cout << "frame=" << processedFrames
                          << " e2e_fps=" << metrics.endToEndFps
                          << " infer_fps=" << metrics.inferenceFps << '\n';
            }
        }

#ifndef ASYNC_PERCEPTION_MTK_G720
        rosSubscriber.reset();
#endif
        if (pipeline) pipeline->stop();
        webPreview.stop();
#ifndef ASYNC_PERCEPTION_NO_ROS2
        rosPublisher.shutdown();
#endif
        releasePerceptionInference();
        flushPendingLogsToFile();
        std::cout << "Finished. Processed RGB frames: " << processedFrames
                  << std::endl;
        return 0;
    } catch (const ob::Error& error) {
        std::cerr << "Orbbec error: " << error.what() << std::endl;
        gLogQueue.push(std::string("Orbbec error: ") + error.what());
        releasePerceptionInference();
        flushPendingLogsToFile();
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Runtime error: " << error.what() << std::endl;
        gLogQueue.push(std::string("Runtime error: ") + error.what());
        releasePerceptionInference();
        flushPendingLogsToFile();
        return 1;
    }
}
