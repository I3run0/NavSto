#pragma once
// =============================================================================
//  Logger.hpp  —  Lightweight structured logger.
//
//  Writes timestamped, severity-tagged lines to both stdout and an optional
//  log file.  Levels: DEBUG < INFO < WARN < ERROR.
//  Call Logger::setLevel(Logger::Level::DEBUG) to enable verbose output.
// =============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <mutex>

class Logger {
public:
    enum class Level { DEBUG = 0, INFO = 1, WARN = 2, ERR = 3 };

    static Logger& instance() {
        static Logger inst;
        return inst;
    }

    void setLevel(Level l)    { minLevel_ = l; }
    void setFile(const std::string& path) {
        file_.open(path, std::ios::out | std::ios::trunc);
    }

    template <typename... Args>
    void log(Level level, Args&&... args) {
        if (level < minLevel_) return;
        std::lock_guard<std::mutex> lock(mu_);
        std::ostringstream oss;
        oss << timestamp() << " [" << tag(level) << "] ";
        (oss << ... << std::forward<Args>(args));
        const std::string msg = oss.str();
        std::cout << msg << '\n';
        if (file_.is_open()) file_ << msg << '\n';
    }

    // Convenience helpers
    template <typename... A> void debug(A&&... a) { log(Level::DEBUG, std::forward<A>(a)...); }
    template <typename... A> void info (A&&... a) { log(Level::INFO,  std::forward<A>(a)...); }
    template <typename... A> void warn (A&&... a) { log(Level::WARN,  std::forward<A>(a)...); }
    template <typename... A> void error(A&&... a) { log(Level::ERR,   std::forward<A>(a)...); }

private:
    Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    Level         minLevel_ = Level::INFO;
    std::ofstream file_;
    std::mutex    mu_;

    static std::string timestamp() {
        using namespace std::chrono;
        auto now  = system_clock::now();
        auto t    = system_clock::to_time_t(now);
        auto ms   = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&t), "%H:%M:%S")
            << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }

    static const char* tag(Level l) {
        switch (l) {
            case Level::DEBUG: return "DEBUG";
            case Level::INFO:  return " INFO";
            case Level::WARN:  return " WARN";
            case Level::ERR:   return "ERROR";
        }
        return "?????";
    }
};

// Global convenience macros — zero overhead when the level is below minLevel_.
#define LOG_DEBUG(...) Logger::instance().debug(__VA_ARGS__)
#define LOG_INFO(...)  Logger::instance().info (__VA_ARGS__)
#define LOG_WARN(...)  Logger::instance().warn (__VA_ARGS__)
#define LOG_ERROR(...) Logger::instance().error(__VA_ARGS__)
