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
#include <optional>
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
#include "shared/command_plan.hpp"
#include "shared/speech_segmenter.hpp"

namespace {

struct CommandInput {
    std::string wake_word;
    bool require_wake_word;
    bool enable_command_plan;
#ifdef VOICE_INPUT_MIC
    const voice_control::MpvPlayer* player;
#endif
};

// 对每条完整语音只生成一个最终 JSON,模型异常不会变为有效动作.
bool ProcessAudio(const std::vector<float>& audio, voice_control::Whisper& whisper,
                  voice_control::DeepSeek& model, std::uint64_t sequence,
                  std::chrono::steady_clock::time_point ended,
                  const voice_control::SpeechSegmenter::Timing& vad_timing,
                  const CommandInput& input, const std::string& source_file) {
    const auto started = std::chrono::steady_clock::now();
    voice_control::Whisper::Timing asr_timing;
    voice_control::DeepSeek::Timing llm_timing;
    voice_control::CommandPlan plan;
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
                plan = {{voice_control::MotionCommands().at("停止")}};
                success = true;
            } else {
                decision_path = "deepseek";
                // 每条控制指令独立分析,防止前一条动作污染当前结果.
                model.Reset();
                const auto prompt = input.enable_command_plan ? voice_control::CommandPrompt(text) :
                    voice_control::SingleCommandPrompt(text);
                success = model.Ask(prompt, llm_timing);
                if (success && input.enable_command_plan) {
                    auto parsed = voice_control::ParseCommandPlan(model.LastText(), text);
                    if (parsed) {
                        plan = std::move(*parsed);
                    } else {
                        error = "invalid_command_plan";
                    }
                } else if (success) {
                    plan = voice_control::ParseSingleCommand(model.LastText(), text);
                }
            }
        }
    } catch (const std::exception& exception) {
        error = exception.what();
        std::cerr << "[处理错误] " << error << std::endl;
    }
    auto result = voice_control::PlanResult(sequence, transcript, plan, success);
    result["decision_path"] = decision_path;
    if (!error.empty()) {
        result["error"] = error;
        if (error == "empty_or_excessive_transcript") {
            result["status"] = "ignored";
        }
    }
#ifdef VOICE_INPUT_MIC
    result["source"]["sensor_id"] = source_file.empty() ? "usb_microphone" : "audio_replay";
#else
    result["source"]["sensor_id"] = "audio_file";
#endif
    if (!source_file.empty()) {
        result["source"]["file"] = source_file;
    }
    if (success && result["message_type"] == "voice_media_intent") {
        result["playback"] = {{"status", "intent_only"}};
#ifdef VOICE_INPUT_MIC
        if (input.player) {
            result["playback"] = input.player->Execute(plan.front().front());
        }
#endif
    }
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
// 两种实时音频来源共用有界队列.文件回放在正常结束时补交尾段并排空队列.
template <typename CaptureFrames>
bool RunLive(CaptureFrames capture_frames, voice_control::SpeechSegmenter& segmenter,
             voice_control::Whisper& whisper, voice_control::DeepSeek& model,
             int capacity, const CommandInput& input, const std::string& source_file) {
    std::uint64_t sequence = 0;
    std::size_t queue_drops = 0;
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
    // 采集线程不调用模型.正常输入队列满时丢弃新语句,回放 EOF 尾段等待空位.
    std::thread capture([&]() {
        try {
            const auto submit = [&](std::vector<float> utterance, bool final_frame) {
                const auto ended = std::chrono::steady_clock::now();
                const auto timing = segmenter.TakeTiming();
                std::unique_lock<std::mutex> lock(mutex);
                const auto full = [&]() { return queue.size() >= static_cast<std::size_t>(capacity); };
                while (final_frame && full() && !voice_control::stop_requested) {
                    ready.wait_for(lock, std::chrono::milliseconds(100));
                }
                if (voice_control::stop_requested) {
                    return;
                }
                if (full()) {
                    ++queue_drops;
                    std::cerr << "[丢弃] 模型繁忙,语句队列已满." << std::endl;
                    return;
                }
                queue.push_back({std::move(utterance), ended, timing});
                ready.notify_one();
            };
            capture_frames([&](const std::vector<float>& frame) {
                std::vector<float> utterance;
                if (segmenter.Feed(frame, utterance)) {
                    submit(std::move(utterance), false);
                }
            });
            std::vector<float> tail;
            if (!source_file.empty() && !voice_control::stop_requested && segmenter.Finish(tail)) {
                submit(std::move(tail), true);
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
                ready.notify_all();
            }
            all_processed = ProcessAudio(utterance.audio, whisper, model, ++sequence,
                utterance.ended, utterance.vad_timing, input, source_file) && all_processed;
        }
    } catch (...) {
        voice_control::stop_requested = 1;
        capture.join();
        throw;
    }
    voice_control::stop_requested = 1;
    capture.join();
    std::cerr << "[实时输入汇总] 输出=" << sequence << ",队列丢段=" << queue_drops
              << ",超长丢段=" << segmenter.Dropped() << std::endl;
    return all_processed && (source_file.empty() || (queue_drops == 0 && segmenter.Dropped() == 0));
}
#else
// 按文件顺序解码和分句,处理完末尾语音后退出.
bool RunFiles(const std::vector<std::string>& files, voice_control::SpeechSegmenter& segmenter,
              voice_control::Whisper& whisper, voice_control::DeepSeek& model,
              const CommandInput& input) {
    std::uint64_t sequence = 0;
    std::size_t processed = 0;
    std::size_t failed = 0;
    std::cerr << "[就绪] FILE 模式,本地音频 -> Silero -> Whisper tiny -> DeepSeek -> 控制 JSON." << std::endl;
    for (const auto& file : files) {
        if (voice_control::stop_requested) {
            break;
        }
        ++processed;
        std::cerr << "[音频] " << processed << "/" << files.size() << ": " << file << std::endl;
        segmenter.Reset();
        const auto before = sequence;
        const auto source_file = std::filesystem::path(file).filename().string();
        bool file_ok = true;
        std::vector<float> utterance;
        // 离线模式逐段同步处理,解码等候推理,保持顺序且不丢有效输入.
        const auto process = [&]() {
            const auto ended = std::chrono::steady_clock::now();
            file_ok = ProcessAudio(utterance, whisper, model, ++sequence,
                ended, segmenter.TakeTiming(), input, source_file) && file_ok;
            utterance.clear();
        };
        try {
            voice_control::ReadAudioFile(file, [&](const std::vector<float>& frame) {
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
        } catch (const std::exception& error) {
            file_ok = false;
            std::cerr << "[文件错误] " << file << ": " << error.what() << std::endl;
        }
        if (segmenter.Dropped()) {
            file_ok = false;
            std::cerr << "[文件] 超长语音段: " << segmenter.Dropped() << std::endl;
        }
        if (file_ok && sequence == before) {
            std::cerr << "[文件] 未检测到有效语音." << std::endl;
        }
        if (!file_ok) {
            ++failed;
        }
    }
    std::cerr << "[文件汇总] 已处理=" << processed << "/" << files.size()
              << ",失败=" << failed << ",输出=" << sequence << std::endl;
    return failed == 0;
}
#endif

}  // 匿名命名空间

// 编译时选择文件或麦克风输入,两种方式复用同一模型和 JSON 协议.
int voice_control::RunApplication(int argc, char** argv) {
    try {
        const bool show_help = argc == 2 && std::string(argv[1]) == "--help";
        const bool force_plan = argc > 2 && std::string(argv[2]) == "--plan";
        const int first_input = force_plan ? 3 : 2;
#ifdef VOICE_INPUT_MIC
        const bool replay = argc == first_input + 2 && std::string(argv[first_input]) == "--replay";
        if (show_help || (argc != first_input && !replay)) {
            std::cerr << "用法(MIC): voice_control_mic <config.json> [--plan] [--replay <audio_file>]\n";
            return show_help ? 0 : 2;
        }
        const std::string replay_file = replay ? argv[first_input + 1] : "";
#else
        if (show_help || argc <= first_input) {
            std::cerr << "用法(FILE): voice_control_file <config.json> [--plan] <audio_file>...\n";
            return show_help ? 0 : 2;
        }
        const std::vector<std::string> files(argv + first_input, argv + argc);
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
        const auto command_mode = speech.value("command_mode", std::string("single"));
        if (command_mode != "single" && command_mode != "plan") {
            throw std::runtime_error("speech.command_mode 必须为 single 或 plan.");
        }
        const auto wake_word = voice_control::Trim(speech.value("wake_word", std::string("闪电")));
        const auto config_root = std::filesystem::absolute(argv[1]).parent_path();
        // 本地资源路径相对配置文件解析,从任意工作目录启动均一致.
        const auto resource_path = [&](const nlohmann::json& value) {
            auto path = std::filesystem::path(value.get<std::string>());
            return (path.is_absolute() ? path : config_root / path).lexically_normal().string();
        };
#ifdef VOICE_INPUT_MIC
        std::optional<voice_control::MpvPlayer> player;
        if (!replay) {
            player.emplace(config.value("media", nlohmann::json::object()), config_root);
        }
        const CommandInput input{wake_word, true, force_plan || command_mode == "plan",
                                 player ? &*player : nullptr};
        if (input.wake_word.empty()) {
            throw std::runtime_error("MIC 模式需要配置非空的 speech.wake_word.");
        }
#else
        const CommandInput input{wake_word, false, force_plan || command_mode == "plan"};
#endif
        std::cerr << "[模式] 指令=" << (input.enable_command_plan ? "plan" : "single") << std::endl;
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
        const auto capacity = audio_options.value("queue_capacity", 2);
        const auto run_live = [&](auto capture, const std::string& source_file) {
            return RunLive(capture, segmenter, whisper, model, capacity, input, source_file);
        };
        bool all_processed;
        if (replay) {
            std::cerr << "[回放] 按实时采样节奏输入: " << replay_file << std::endl;
            all_processed = run_live([&](const auto& consume) {
                voice_control::ReplayAudioFile(replay_file, consume);
            }, std::filesystem::path(replay_file).filename().string());
        } else {
            voice_control::Microphone microphone(audio_options.at("device"));
            std::cerr << "[监听] USB 麦克风已就绪;Ctrl+C 后等待在途推理完成." << std::endl;
            all_processed = run_live([&](const auto& consume) {
                std::vector<float> frame;
                while (!voice_control::stop_requested && microphone.Read(frame)) {
                    consume(frame);
                }
            }, "");
        }
#else
        const bool all_processed = RunFiles(files, segmenter, whisper, model, input);
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
