//===- Logging.cpp - Logging implementation -----------------------------===//
//
// This file implements logging utilities.
//
//===----------------------------------------------------------------------===//

#include "photon/core/Logging.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <iomanip>
#include <algorithm>
#include <sstream>

using namespace mlir;
using namespace mlir::photonic;

thread_local std::string LogContext::current_context_ = "";

std::string logLevelToString(LogLevel level) {
  switch (level) {
    case LogLevel::TRACE: return "TRACE";
    case LogLevel::DEBUG: return "DEBUG";
    case LogLevel::INFO:  return "INFO";
    case LogLevel::WARN:  return "WARN";
    case LogLevel::ERROR: return "ERROR";
    case LogLevel::FATAL: return "FATAL";
    default: return "UNKNOWN";
  }
}

LogLevel logLevelFromString(const std::string& level) {
  std::string upper = level;
  std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
  
  if (upper == "TRACE") return LogLevel::TRACE;
  if (upper == "DEBUG") return LogLevel::DEBUG;
  if (upper == "INFO")  return LogLevel::INFO;
  if (upper == "WARN")  return LogLevel::WARN;
  if (upper == "ERROR") return LogLevel::ERROR;
  if (upper == "FATAL") return LogLevel::FATAL;
  
  return LogLevel::INFO; // Default
}

std::string LogEntry::format() const {
  std::ostringstream oss;
  
  // Timestamp
  auto time_t = std::chrono::system_clock::to_time_t(timestamp);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      timestamp.time_since_epoch()) % 1000;
  
  oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
  oss << "." << std::setfill('0') << std::setw(3) << ms.count();
  
  // Level
  oss << " [" << std::setw(5) << logLevelToString(level) << "]";
  
  // Thread ID
  oss << " [" << thread_id << "]";
  
  // Context
  if (!context.empty()) {
    oss << " (" << context << ")";
  }
  
  // Message
  oss << " " << message;
  
  // File and line (for debug builds)
  if (!file.empty() && line > 0) {
    oss << " [" << llvm::sys::path::filename(file) << ":" << line << "]";
  }
  
  return oss.str();
}

std::string LogEntry::formatJSON() const {
  std::ostringstream oss;
  
  auto time_t = std::chrono::system_clock::to_time_t(timestamp);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      timestamp.time_since_epoch()) % 1000;
  
  oss << "{";
  oss << "\"timestamp\":\"" << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
  oss << "." << std::setfill('0') << std::setw(3) << ms.count() << "Z\",";
  oss << "\"level\":\"" << logLevelToString(level) << "\",";
  oss << "\"thread\":\"" << thread_id << "\",";
  
  if (!context.empty()) {
    oss << "\"context\":\"" << context << "\",";
  }
  
  oss << "\"message\":\"" << message << "\"";
  
  if (!file.empty() && line > 0) {
    oss << ",\"file\":\"" << llvm::sys::path::filename(file) << "\",";
    oss << "\"line\":" << line;
  }
  
  oss << "}";
  return oss.str();
}

//===----------------------------------------------------------------------===//
// ConsoleSink
//===----------------------------------------------------------------------===//

ConsoleSink::ConsoleSink(bool use_colors) : use_colors_(use_colors) {}

void ConsoleSink::write(const LogEntry& entry) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (use_colors_) {
    // ANSI color codes
    switch (entry.level) {
      case LogLevel::TRACE: llvm::outs() << "\033[90m"; break; // Dark gray
      case LogLevel::DEBUG: llvm::outs() << "\033[36m"; break; // Cyan
      case LogLevel::INFO:  llvm::outs() << "\033[32m"; break; // Green
      case LogLevel::WARN:  llvm::outs() << "\033[33m"; break; // Yellow
      case LogLevel::ERROR: llvm::errs() << "\033[31m"; break; // Red
      case LogLevel::FATAL: llvm::errs() << "\033[35m"; break; // Magenta
    }
  }
  
  auto& stream = (entry.level >= LogLevel::ERROR) ? llvm::errs() : llvm::outs();
  stream << entry.format();
  
  if (use_colors_) {
    stream << "\033[0m"; // Reset color
  }
  
  stream << "\n";
}

void ConsoleSink::flush() {
  llvm::outs().flush();
  llvm::errs().flush();
}

//===----------------------------------------------------------------------===//
// FileSink
//===----------------------------------------------------------------------===//

FileSink::FileSink(const std::string& filename, size_t max_size_mb, int max_files)
    : base_filename_(filename), max_size_bytes_(max_size_mb * 1024 * 1024),
      max_files_(max_files), current_size_(0) {
  
  file_.open(filename, std::ios::app);
  if (file_.is_open()) {
    // Get current file size
    file_.seekp(0, std::ios::end);
    current_size_ = file_.tellp();
  }
}

FileSink::~FileSink() {
  if (file_.is_open()) {
    file_.close();
  }
}

void FileSink::write(const LogEntry& entry) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (!file_.is_open()) {
    return;
  }
  
  std::string formatted = entry.format() + "\n";
  file_ << formatted;
  current_size_ += formatted.size();
  
  rotateIfNeeded();
}

void FileSink::flush() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (file_.is_open()) {
    file_.flush();
  }
}

void FileSink::rotateIfNeeded() {
  if (current_size_ > max_size_bytes_) {
    rotate();
  }
}

void FileSink::rotate() {
  if (file_.is_open()) {
    file_.close();
  }
  
  // Rotate existing files
  for (int i = max_files_ - 1; i > 0; --i) {
    std::string from = base_filename_ + "." + std::to_string(i);
    std::string to = base_filename_ + "." + std::to_string(i + 1);
    
    if (llvm::sys::fs::exists(from)) {
      if (i == max_files_ - 1) {
        llvm::sys::fs::remove(from); // Remove oldest
      } else {
        llvm::sys::fs::rename(from, to);
      }
    }
  }
  
  // Move current log to .1
  if (llvm::sys::fs::exists(base_filename_)) {
    llvm::sys::fs::rename(base_filename_, base_filename_ + ".1");
  }
  
  // Open new log file
  file_.open(base_filename_, std::ios::out);
  current_size_ = 0;
}

//===----------------------------------------------------------------------===//
// JSONSink
//===----------------------------------------------------------------------===//

JSONSink::JSONSink(const std::string& filename) : first_entry_(true) {
  file_.open(filename, std::ios::out);
  if (file_.is_open()) {
    file_ << "[\n";
  }
}

JSONSink::~JSONSink() {
  if (file_.is_open()) {
    file_ << "\n]\n";
    file_.close();
  }
}

void JSONSink::write(const LogEntry& entry) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (!file_.is_open()) {
    return;
  }
  
  if (!first_entry_) {
    file_ << ",\n";
  } else {
    first_entry_ = false;
  }
  
  file_ << "  " << entry.formatJSON();
}

void JSONSink::flush() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (file_.is_open()) {
    file_.flush();
  }
}

//===----------------------------------------------------------------------===//
// Logger
//===----------------------------------------------------------------------===//

Logger& Logger::getInstance() {
  static Logger instance;
  return instance;
}

void Logger::addSink(std::unique_ptr<LogSink> sink) {
  std::lock_guard<std::mutex> lock(mutex_);
  sinks_.push_back(std::move(sink));
}

void Logger::clearSinks() {
  std::lock_guard<std::mutex> lock(mutex_);
  sinks_.clear();
}

void Logger::log(LogLevel level, const std::string& message,
                const std::string& context, const std::string& file, int line) {
  if (level < level_) {
    return;
  }
  
  LogEntry entry;
  entry.level = level;
  entry.message = message;
  entry.context = context;
  entry.file = file;
  entry.line = line;
  entry.timestamp = std::chrono::system_clock::now();
  entry.thread_id = std::this_thread::get_id();
  
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& sink : sinks_) {
    sink->write(entry);
  }
}

void Logger::trace(const std::string& message, const std::string& context) {
  log(LogLevel::TRACE, message, context);
}

void Logger::debug(const std::string& message, const std::string& context) {
  log(LogLevel::DEBUG, message, context);
}

void Logger::info(const std::string& message, const std::string& context) {
  log(LogLevel::INFO, message, context);
}

void Logger::warn(const std::string& message, const std::string& context) {
  log(LogLevel::WARN, message, context);
}

void Logger::error(const std::string& message, const std::string& context) {
  log(LogLevel::ERROR, message, context);
}

void Logger::fatal(const std::string& message, const std::string& context) {
  log(LogLevel::FATAL, message, context);
}

void Logger::flush() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& sink : sinks_) {
    sink->flush();
  }
}

void Logger::logCompilationMetrics(const std::string& phase,
                                  std::chrono::milliseconds duration,
                                  size_t memory_usage_mb) {
  std::ostringstream oss;
  oss << "Compilation phase '" << phase << "' completed in " << duration.count() << "ms";
  if (memory_usage_mb > 0) {
    oss << ", memory usage: " << memory_usage_mb << "MB";
  }
  info(oss.str(), "performance");
}

void Logger::logOptimizationStep(const std::string& pass_name,
                                const std::string& operation,
                                bool success,
                                const std::string& details) {
  std::ostringstream oss;
  oss << "Pass '" << pass_name << "' on '" << operation << "': " 
      << (success ? "SUCCESS" : "FAILED");
  if (!details.empty()) {
    oss << " - " << details;
  }
  
  if (success) {
    debug(oss.str(), "optimization");
  } else {
    warn(oss.str(), "optimization");
  }
}

//===----------------------------------------------------------------------===//
// LogContext
//===----------------------------------------------------------------------===//

LogContext::LogContext(const std::string& context) : previous_context_(current_context_) {
  if (!current_context_.empty()) {
    current_context_ += " -> " + context;
  } else {
    current_context_ = context;
  }
}

LogContext::~LogContext() {
  current_context_ = previous_context_;
}

std::string LogContext::getCurrentContext() {
  return current_context_;
}

//===----------------------------------------------------------------------===//
// PerformanceTimer
//===----------------------------------------------------------------------===//

PerformanceTimer::PerformanceTimer(const std::string& operation, LogLevel level)
    : operation_(operation), level_(level) {
  start_time_ = std::chrono::steady_clock::now();
  last_checkpoint_ = start_time_;
  
  Logger::getInstance().log(level_, "Started: " + operation_, LogContext::getCurrentContext());
}

PerformanceTimer::~PerformanceTimer() {
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_time_);
  
  std::ostringstream oss;
  oss << "Completed: " << operation_ << " (" << duration.count() << "ms)";
  Logger::getInstance().log(level_, oss.str(), LogContext::getCurrentContext());
}

void PerformanceTimer::checkpoint(const std::string& checkpoint_name) {
  auto now = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_checkpoint_);
  
  std::ostringstream oss;
  oss << operation_ << " -> " << checkpoint_name << " (" << duration.count() << "ms)";
  Logger::getInstance().log(level_, oss.str(), LogContext::getCurrentContext());
  
  last_checkpoint_ = now;
}

std::chrono::milliseconds PerformanceTimer::elapsed() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_time_);
}

//===----------------------------------------------------------------------===//
// StructuredLog
//===----------------------------------------------------------------------===//

StructuredLog::StructuredLog(LogLevel level) : level_(level), has_fields_(false) {}

StructuredLog::~StructuredLog() {
  std::ostringstream oss;
  oss << message_;
  
  if (has_fields_) {
    oss << " {" << fields_.str() << "}";
  }
  
  Logger::getInstance().log(level_, oss.str(), context_);
}

StructuredLog& StructuredLog::field(const std::string& key, const std::string& value) {
  if (has_fields_) fields_ << ", ";
  fields_ << key << ": \"" << value << "\"";
  has_fields_ = true;
  return *this;
}

StructuredLog& StructuredLog::field(const std::string& key, int value) {
  if (has_fields_) fields_ << ", ";
  fields_ << key << ": " << value;
  has_fields_ = true;
  return *this;
}

StructuredLog& StructuredLog::field(const std::string& key, double value) {
  if (has_fields_) fields_ << ", ";
  fields_ << key << ": " << value;
  has_fields_ = true;
  return *this;
}

StructuredLog& StructuredLog::field(const std::string& key, bool value) {
  if (has_fields_) fields_ << ", ";
  fields_ << key << ": " << (value ? "true" : "false");
  has_fields_ = true;
  return *this;
}

StructuredLog& StructuredLog::message(const std::string& msg) {
  message_ = msg;
  return *this;
}

StructuredLog& StructuredLog::context(const std::string& ctx) {
  context_ = ctx;
  return *this;
}