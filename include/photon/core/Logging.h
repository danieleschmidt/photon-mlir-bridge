//===- Logging.h - Comprehensive logging infrastructure -------*- C++ -*-===//
//
// This file defines logging utilities for the photonic compiler.
//
//===----------------------------------------------------------------------===//

#ifndef PHOTONIC_LOGGING_H
#define PHOTONIC_LOGGING_H

#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>

namespace mlir {
namespace photonic {

/// Log levels
enum class LogLevel {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  FATAL = 5
};

/// Convert log level to string
std::string logLevelToString(LogLevel level);

/// Log level from string
LogLevel logLevelFromString(const std::string& level);

/// Log entry with metadata
struct LogEntry {
  LogLevel level;
  std::string message;
  std::string context;
  std::string file;
  int line;
  std::chrono::system_clock::time_point timestamp;
  std::thread::id thread_id;
  
  std::string format() const;
  std::string formatJSON() const;
};

/// Abstract log sink interface
class LogSink {
public:
  virtual ~LogSink() = default;
  virtual void write(const LogEntry& entry) = 0;
  virtual void flush() = 0;
};

/// Console log sink
class ConsoleSink : public LogSink {
public:
  ConsoleSink(bool use_colors = true);
  void write(const LogEntry& entry) override;
  void flush() override;

private:
  bool use_colors_;
  std::mutex mutex_;
};

/// File log sink with rotation
class FileSink : public LogSink {
public:
  FileSink(const std::string& filename, 
           size_t max_size_mb = 100,
           int max_files = 5);
  ~FileSink();
  
  void write(const LogEntry& entry) override;
  void flush() override;

private:
  void rotateIfNeeded();
  void rotate();
  
  std::string base_filename_;
  std::ofstream file_;
  size_t max_size_bytes_;
  int max_files_;
  size_t current_size_;
  std::mutex mutex_;
};

/// Structured JSON log sink
class JSONSink : public LogSink {
public:
  JSONSink(const std::string& filename);
  ~JSONSink();
  
  void write(const LogEntry& entry) override;
  void flush() override;

private:
  std::ofstream file_;
  std::mutex mutex_;
  bool first_entry_;
};

/// Main logger class
class Logger {
public:
  static Logger& getInstance();
  
  void setLevel(LogLevel level) { level_ = level; }
  LogLevel getLevel() const { return level_; }
  
  void addSink(std::unique_ptr<LogSink> sink);
  void clearSinks();
  
  void log(LogLevel level, const std::string& message, 
           const std::string& context = "",
           const std::string& file = "",
           int line = 0);
  
  void trace(const std::string& message, const std::string& context = "");
  void debug(const std::string& message, const std::string& context = "");
  void info(const std::string& message, const std::string& context = "");
  void warn(const std::string& message, const std::string& context = "");
  void error(const std::string& message, const std::string& context = "");
  void fatal(const std::string& message, const std::string& context = "");
  
  void flush();
  
  // Performance logging
  void logCompilationMetrics(const std::string& phase, 
                           std::chrono::milliseconds duration,
                           size_t memory_usage_mb = 0);
  
  void logOptimizationStep(const std::string& pass_name,
                          const std::string& operation,
                          bool success,
                          const std::string& details = "");

private:
  Logger() : level_(LogLevel::INFO) {}
  
  LogLevel level_;
  std::vector<std::unique_ptr<LogSink>> sinks_;
  std::mutex mutex_;
};

/// RAII logging context
class LogContext {
public:
  LogContext(const std::string& context);
  ~LogContext();
  
  static std::string getCurrentContext();

private:
  std::string previous_context_;
  static thread_local std::string current_context_;
};

/// Performance timer with automatic logging
class PerformanceTimer {
public:
  PerformanceTimer(const std::string& operation, LogLevel level = LogLevel::DEBUG);
  ~PerformanceTimer();
  
  void checkpoint(const std::string& checkpoint_name);
  std::chrono::milliseconds elapsed() const;

private:
  std::string operation_;
  LogLevel level_;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point last_checkpoint_;
};

/// Structured logging helper
class StructuredLog {
public:
  StructuredLog(LogLevel level);
  ~StructuredLog();
  
  StructuredLog& field(const std::string& key, const std::string& value);
  StructuredLog& field(const std::string& key, int value);
  StructuredLog& field(const std::string& key, double value);
  StructuredLog& field(const std::string& key, bool value);
  
  StructuredLog& message(const std::string& msg);
  StructuredLog& context(const std::string& ctx);

private:
  LogLevel level_;
  std::ostringstream fields_;
  std::string message_;
  std::string context_;
  bool has_fields_;
};

// Convenient macros for logging
#define PHOTONIC_LOG(level, message) \
  Logger::getInstance().log(level, message, LogContext::getCurrentContext(), __FILE__, __LINE__)

#define PHOTONIC_TRACE(message) \
  Logger::getInstance().trace(message, LogContext::getCurrentContext())

#define PHOTONIC_DEBUG(message) \
  Logger::getInstance().debug(message, LogContext::getCurrentContext())

#define PHOTONIC_INFO(message) \
  Logger::getInstance().info(message, LogContext::getCurrentContext())

#define PHOTONIC_WARN(message) \
  Logger::getInstance().warn(message, LogContext::getCurrentContext())

#define PHOTONIC_ERROR(message) \
  Logger::getInstance().error(message, LogContext::getCurrentContext())

#define PHOTONIC_FATAL(message) \
  Logger::getInstance().fatal(message, LogContext::getCurrentContext())

#define PHOTONIC_TIMER(operation) \
  PerformanceTimer __timer(operation)

#define PHOTONIC_CONTEXT(context) \
  LogContext __context(context)

#define PHOTONIC_STRUCTURED(level) \
  StructuredLog(level)

} // namespace photonic
} // namespace mlir

#endif // PHOTONIC_LOGGING_H