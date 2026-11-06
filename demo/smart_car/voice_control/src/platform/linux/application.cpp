#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "platform/linux/application.hpp"
#include "platform/linux/audio.hpp"
#ifdef VOICE_INPUT_MIC
#include "platform/linux/mpv_player.hpp"
#endif
#include "platform/s100/deepseek.hpp"
#include "platform/s100/whisper.hpp"
#include "shared/command.hpp"
#include "shared/speech_segmenter.hpp"

namespace {

struct CommandInput {
    std::string wake_word;
    bool require_wake_word;
#ifdef VOICE_INPUT_MIC
    const voice_control::MpvPlayer& player;
#endif
};

// 对每条完整语音只生成一个最终 JSON,模型异常不会变为有效动作.
bool ProcessAudio(const std::vector<float>& audio, voice_control::Whisper& whisper,
                  voice_control::DeepSeek& model, std::uint64_t sequence,
                  std::chrono::steady_clock::time_point ended,
                  const voice_control::SpeechSegmenter::Timing& vad_timing,
                  const CommandInput& input) {
    const auto started = std::chrono::steady_clock::now();
    voice_control::Whisper::Timing asr_timing;
    voice_control::DeepSeek::Timing llm_timing;
    voice_control::Command command;
    std::string transcript;
    const char* decision_path = "not_classified";
    bool success = false;
    std::string error;
    try {
        transcript = voice_control::Trim(whisper.Transcribe(audio, asr_timing));
        std::cerr << "[转录] " << transcript << std::endl;
        if (transcript.empty() || transcript.size() > 2048) {
            error = "empty_or_excessive_transcript";
        } else {
            const auto text = voice_control::NormalizeCommand(
                transcript, input.wake_word, input.require_wake_word);
            if (text.empty()) {
                success = true;
            } else if (voice_control::IsEmergencyStop(transcript, input.wake_word)) {
                decision_path = "emergency_stop";
                command = voice_control::MotionCommands().at("停止");
                success = true;
            } else {
                decision_path = "deepseek";
                // 每条控制指令独立分析,防止前一条动作污染当前结果.
                model.Reset();
                success = model.Ask(voice_control::CommandPrompt(text), llm_timing);
                if (success) {
                    command = voice_control::ModelCommand(model.LastText(), text);
                }
            }
        }
    } catch (const std::exception& exception) {
        error = exception.what();
        std::cerr << "[处理错误] " << error << std::endl;
    }
    auto result = voice_control::CommandResult(sequence, transcript, command, success);
    result["decision_path"] = decision_path;
    if (!error.empty()) {
        result["error"] = error;
        if (error == "empty_or_excessive_transcript") {
            result["status"] = "ignored";
        }
    }
#ifdef VOICE_INPUT_MIC
    result["source"]["sensor_id"] = "usb_microphone";
    if (success && command.media) {
        result["playback"] = input.player.Execute(command);
    }
#else
    result["source"]["sensor_id"] = "audio_file";
    if (success && command.media) {
        result["playback"] = {{"status", "intent_only"}};
    }
#endif
    // 缺失阶段输出 null,避免把未执行或未完成的计时写为零.
    const auto measured = [](double value) -> nlohmann::json {
        return value < 0 ? nlohmann::json(nullptr) : nlohmann::json(value);
    };
    result["timing_ms"] = {
        {"vad_compute", vad_timing.compute_ms},
        {"vad_tail_audio", vad_timing.tail_audio_ms},
        {"queue", std::chrono::duration<double, std::milli>(started - ended).count()},
        {"whisper_mel", measured(asr_timing.mel_ms)},
        {"whisper_encoder", measured(asr_timing.encoder_ms)},
        {"whisper_decoder", measured(asr_timing.decoder_ms)},
        {"deepseek_first_text", measured(llm_timing.first_text_ms)},
        {"deepseek_total", measured(llm_timing.total_ms)},
    };
    result["since_audio_end_ms"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - ended).count();
    const auto json = result.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    std::cout << json << '\n' << std::flush;
    return success || result["status"] == "ignored";
}

#ifdef VOICE_INPUT_MIC
// 持续采集,通过有界队列将分句和模型推理解耦.
bool RunMicrophone(voice_control::Microphone& microphone,
                   voice_control::SpeechSegmenter& segmenter,
                   voice_control::Whisper& whisper, voice_control::DeepSeek& model,
                   int capacity, const CommandInput& input) {
    std::uint64_t sequence = 0;
    bool all_processed = true;
    struct Utterance {
        std::vector<float> audio;
        std::chrono::steady_clock::time_point ended;
        voice_control::SpeechSegmenter::Timing vad_timing;
    };
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<Utterance> queue;
    std::exception_ptr capture_error;
    bool finished = false;
    if (capacity < 1 || capacity > 8) {
        throw std::runtime_error("queue_capacity 必须在 1 到 8 之间.");
    }
    // 采集线程不调用模型,有界队列满时丢弃新语句并提示.
    std::thread capture([&]() {
        try {
            std::vector<float> frame;
            std::cerr << "[监听] USB 麦克风已就绪;Ctrl+C 后等待在途推理完成." << std::endl;
            while (!voice_control::stop_requested && microphone.Read(frame)) {
                std::vector<float> utterance;
                if (segmenter.Feed(frame, utterance)) {
                    const auto ended = std::chrono::steady_clock::now();
                    const auto timing = segmenter.TakeTiming();
                    std::lock_guard<std::mutex> lock(mutex);
                    if (queue.size() >= static_cast<std::size_t>(capacity)) {
                        std::cerr << "[丢弃] 模型繁忙,语句队列已满." << std::endl;
                    } else {
                        queue.push_back({std::move(utterance), ended, timing});
                        ready.notify_one();
                    }
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            capture_error = std::current_exception();
        }
        std::lock_guard<std::mutex> lock(mutex);
        finished = true;
        ready.notify_all();
    });
    try {
        while (!voice_control::stop_requested) {
            Utterance utterance;
            {
                std::unique_lock<std::mutex> lock(mutex);
                ready.wait_for(lock, std::chrono::milliseconds(100), [&]() {
                    return finished || !queue.empty();
                });
                if (capture_error) {
                    std::rethrow_exception(capture_error);
                }
                if (queue.empty()) {
                    if (finished) {
                        break;
                    }
                    continue;
                }
                utterance = std::move(queue.front());
                queue.pop_front();
            }
            all_processed = ProcessAudio(utterance.audio, whisper, model, ++sequence,
                utterance.ended, utterance.vad_timing, input) && all_processed;
        }
    } catch (...) {
        voice_control::stop_requested = 1;
        capture.join();
        throw;
    }
    voice_control::stop_requested = 1;
    capture.join();
    return all_processed;
}
#else
// 按文件顺序解码和分句,处理完末尾语音后退出.
bool RunFiles(int argc, char** argv, voice_control::SpeechSegmenter& segmenter,
              voice_control::Whisper& whisper, voice_control::DeepSeek& model,
              const CommandInput& input) {
    std::uint64_t sequence = 0;
    bool all_processed = true;
    std::cerr << "[就绪] FILE 模式,本地音频 -> Silero -> Whisper tiny -> DeepSeek -> 控制 JSON." << std::endl;
    for (int i = 2; i < argc && !voice_control::stop_requested; ++i) {
        std::cerr << "[音频] " << i - 1 << "/" << argc - 2 << ": " << argv[i] << std::endl;
        segmenter.Reset();
        const auto before = sequence;
        std::vector<float> utterance;
        // 离线模式逐段同步处理,解码等候推理,保持顺序且不丢有效输入.
        const auto process = [&]() {
            const auto ended = std::chrono::steady_clock::now();
            all_processed = ProcessAudio(utterance, whisper, model, ++sequence,
                ended, segmenter.TakeTiming(), input) && all_processed;
            utterance.clear();
        };
        voice_control::ReadAudioFile(argv[i], [&](const std::vector<float>& frame) {
            if (segmenter.Feed(frame, utterance)) {
                process();
            }
        });
        if (voice_control::stop_requested) {
            break;
        }
        if (segmenter.Finish(utterance)) {
            process();
        }
        if (segmenter.Dropped()) {
            all_processed = false;
            std::cerr << "[文件] 超长语音段: " << segmenter.Dropped() << std::endl;
        }
        if (sequence == before && segmenter.Dropped() == 0) {
            std::cerr << "[文件] 未检测到有效语音." << std::endl;
        }
    }
    return all_processed;
}
#endif

}  // 匿名命名空间

// 编译时选择文件或麦克风输入,两种方式复用同一模型和 JSON 协议.
int voice_control::RunApplication(int argc, char** argv) {
    try {
#ifdef VOICE_INPUT_MIC
        if (argc != 2) {
            std::cerr << "用法(MIC): voice_control_mic <config.json>\n";
            return 2;
        }
#else
        if (argc < 3) {
            std::cerr << "用法(FILE): voice_control_file <config.json> <audio_file>...\n";
            return 2;
        }
#endif
        std::signal(SIGINT, voice_control::HandleSignal);
        std::signal(SIGTERM, voice_control::HandleSignal);
        std::ifstream stream(argv[1]);
        if (!stream) {
            throw std::runtime_error("无法读取本地配置.");
        }
        const auto config = nlohmann::json::parse(stream);
        const auto audio_options = config.value("audio", nlohmann::json::object());
        const auto speech = config.value("speech", nlohmann::json::object());
        const auto wake_word = voice_control::Trim(speech.value("wake_word", std::string("闪电")));
        const auto config_root = std::filesystem::absolute(argv[1]).parent_path();
        // 本地资源路径相对配置文件解析,从任意工作目录启动均一致.
        const auto resource_path = [&](const nlohmann::json& value) {
            auto path = std::filesystem::path(value.get<std::string>());
            return (path.is_absolute() ? path : config_root / path).lexically_normal().string();
        };
#ifdef VOICE_INPUT_MIC
        const voice_control::MpvPlayer player(config.value("media", nlohmann::json::object()), config_root);
        const CommandInput input{wake_word, true, player};
        if (input.wake_word.empty()) {
            throw std::runtime_error("MIC 模式需要配置非空的 speech.wake_word.");
        }
        voice_control::Microphone microphone(audio_options.at("device"));
#else
        const CommandInput input{wake_word, false};
#endif
        const auto& vad = config.at("vad");
        voice_control::SpeechSegmenter segmenter(resource_path(vad.at("model")),
            vad.value("threshold", 0.5F), audio_options.value("end_silence_ms", 700),
            audio_options.value("min_speech_ms", 160), audio_options.value("max_utterance_ms", 30000));
        const auto& llm = config.at("llm");
        voice_control::DeepSeek model(resource_path(llm.at("model")), resource_path(llm.at("tokenizer")),
                                      resource_path(llm.at("chat_template")), llm.value("print_raw", false),
                                      llm.value("thinking", false));
        const auto& asr = config.at("whisper");
        voice_control::Whisper whisper(resource_path(asr.at("models")), asr.value("language", "zh"));
#ifdef VOICE_INPUT_MIC
        const bool all_processed = RunMicrophone(
            microphone, segmenter, whisper, model, audio_options.value("queue_capacity", 2), input);
#else
        const bool all_processed = RunFiles(argc, argv, segmenter, whisper, model, input);
#endif
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            std::cerr << "[内存] process_peak_rss_kib=" << usage.ru_maxrss << std::endl;
        }
        std::cerr << "[退出] 处理结束." << std::endl;
        return all_processed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[错误] " << error.what() << std::endl;
        return 1;
    }
}
