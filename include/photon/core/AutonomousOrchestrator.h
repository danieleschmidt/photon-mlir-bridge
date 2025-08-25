//===- AutonomousOrchestrator.h - Autonomous SDLC orchestrator ---------===//
//
// Generation 1: Autonomous compilation orchestrator for photonic systems
// Implements self-improving compilation with intelligent decision making
//
//===----------------------------------------------------------------------===//

#ifndef PHOTONIC_AUTONOMOUS_ORCHESTRATOR_H
#define PHOTONIC_AUTONOMOUS_ORCHESTRATOR_H

#include "photon/core/PhotonicCompiler.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>

namespace mlir {
namespace photonic {

/// Autonomous compilation task types
enum class TaskType {
  COMPILATION,
  OPTIMIZATION,
  VALIDATION,
  THERMAL_MANAGEMENT,
  PERFORMANCE_TUNING,
  QUANTUM_CALIBRATION
};

/// Task priority levels
enum class Priority {
  LOW = 0,
  NORMAL = 1,
  HIGH = 2,
  CRITICAL = 3,
  EMERGENCY = 4
};

/// Autonomous compilation task
struct AutonomousTask {
  size_t id;
  TaskType type;
  Priority priority;
  std::string model_path;
  PhotonicTargetConfig config;
  std::function<LogicalResult()> execution_function;
  std::chrono::steady_clock::time_point created_time;
  std::chrono::steady_clock::time_point deadline;
  int retry_count = 0;
  std::string description;
};

/// Learning-based performance metrics
struct PerformanceMetrics {
  double avg_compilation_time_ms = 0.0;
  double success_rate = 1.0;
  double thermal_efficiency = 0.8;
  double power_efficiency = 0.9;
  double phase_stability = 0.95;
  size_t total_tasks_completed = 0;
  size_t optimization_improvements = 0;
  double learning_rate = 0.01;
  
  // Quantum-enhanced metrics
  double quantum_coherence_time_us = 1000.0;
  double quantum_fidelity = 0.99;
  size_t quantum_error_corrections = 0;
};

/// Intelligent decision making system
class DecisionEngine {
public:
  DecisionEngine();
  
  /// Make autonomous decisions based on current state
  bool shouldOptimize(const PerformanceMetrics& metrics);
  bool shouldUpgradePrecision(const OptimizationStats& stats);
  bool shouldEnableQuantumMode(const PhotonicTargetConfig& config);
  TaskType selectNextTask(const std::vector<AutonomousTask>& pending_tasks);
  Priority calculatePriority(const AutonomousTask& task, const PerformanceMetrics& metrics);
  
  /// Learn from outcomes
  void updateFromResult(const AutonomousTask& task, LogicalResult result, 
                       const PerformanceMetrics& metrics);
  
  /// Thermal-aware scheduling
  bool isThermalSafe(double current_temp, const PhotonicTargetConfig& config);
  int calculateCooldownTime(double current_temp, double target_temp);
  
private:
  std::mutex decision_mutex_;
  std::vector<double> learning_weights_;
  size_t decision_count_ = 0;
  double confidence_threshold_ = 0.8;
};

/// Self-improving compilation orchestrator
class AutonomousOrchestrator {
public:
  AutonomousOrchestrator();
  ~AutonomousOrchestrator();
  
  /// Core orchestration interface
  LogicalResult initialize();
  void shutdown();
  bool isRunning() const { return orchestrator_running_.load(); }
  
  /// Task management
  size_t submitTask(const AutonomousTask& task);
  size_t scheduleCompilation(llvm::StringRef model_path, 
                           const PhotonicTargetConfig& config,
                           Priority priority = Priority::NORMAL);
  size_t scheduleOptimization(llvm::StringRef model_path,
                            const PhotonicTargetConfig& config);
  
  /// Autonomous operation modes
  void enableAutonomousMode(bool enable) { autonomous_mode_enabled_.store(enable); }
  void enableLearningMode(bool enable) { learning_mode_enabled_.store(enable); }
  void enableQuantumMode(bool enable) { quantum_mode_enabled_.store(enable); }
  
  /// Performance monitoring and improvement
  PerformanceMetrics getCurrentMetrics() const;
  LogicalResult optimizePerformance();
  LogicalResult calibrateQuantumSystem();
  LogicalResult performThermalOptimization();
  
  /// Configuration and tuning
  void setMaxConcurrentTasks(size_t max_tasks);
  void setLearningRate(double rate);
  void setPerformanceThresholds(double min_success_rate, double max_thermal_temp);
  
  /// Status and diagnostics
  std::vector<AutonomousTask> getPendingTasks() const;
  std::vector<AutonomousTask> getCompletedTasks() const;
  std::string getStatusReport() const;
  
  /// Callbacks for external monitoring
  using TaskCompletedCallback = std::function<void(const AutonomousTask&, LogicalResult)>;
  using PerformanceCallback = std::function<void(const PerformanceMetrics&)>;
  
  void setTaskCompletedCallback(TaskCompletedCallback callback);
  void setPerformanceCallback(PerformanceCallback callback);

private:
  // Core components
  std::unique_ptr<PhotonicCompiler> compiler_;
  std::unique_ptr<DecisionEngine> decision_engine_;
  std::unique_ptr<Logger> logger_;
  
  // Task management
  std::vector<AutonomousTask> pending_tasks_;
  std::vector<AutonomousTask> active_tasks_;
  std::vector<AutonomousTask> completed_tasks_;
  std::atomic<size_t> next_task_id_{1};
  
  // Threading and concurrency
  std::atomic<bool> orchestrator_running_{false};
  std::atomic<bool> autonomous_mode_enabled_{true};
  std::atomic<bool> learning_mode_enabled_{true};
  std::atomic<bool> quantum_mode_enabled_{false};
  std::atomic<size_t> max_concurrent_tasks_{4};
  
  std::vector<std::thread> worker_threads_;
  std::thread orchestrator_thread_;
  std::thread performance_monitor_thread_;
  
  std::mutex task_queue_mutex_;
  std::condition_variable task_available_cv_;
  std::condition_variable task_completed_cv_;
  
  // Performance tracking
  PerformanceMetrics current_metrics_;
  std::mutex metrics_mutex_;
  
  // Thermal management
  std::atomic<double> system_temperature_{25.0}; // Celsius
  std::atomic<bool> thermal_override_{false};
  
  // Callbacks
  TaskCompletedCallback task_completed_callback_;
  PerformanceCallback performance_callback_;
  std::mutex callback_mutex_;
  
  // Worker thread functions
  void orchestratorLoop();
  void workerLoop(size_t worker_id);
  void performanceMonitorLoop();
  void thermalManagementLoop();
  
  // Task execution
  LogicalResult executeTask(AutonomousTask& task);
  LogicalResult executeCompilationTask(AutonomousTask& task);
  LogicalResult executeOptimizationTask(AutonomousTask& task);
  LogicalResult executeValidationTask(AutonomousTask& task);
  LogicalResult executeQuantumCalibration(AutonomousTask& task);
  
  // Learning and adaptation
  void updatePerformanceMetrics(const AutonomousTask& task, 
                               LogicalResult result,
                               std::chrono::duration<double> execution_time);
  void adaptToPerformance();
  void optimizeTaskScheduling();
  
  // Utility methods
  void sortTasksByPriority();
  bool canExecuteTask(const AutonomousTask& task) const;
  void cleanupCompletedTasks();
  std::string getTaskTypeString(TaskType type) const;
  std::string getPriorityString(Priority priority) const;
  
  // Constants
  static constexpr size_t MAX_COMPLETED_TASKS_HISTORY = 1000;
  static constexpr double DEFAULT_LEARNING_RATE = 0.01;
  static constexpr double THERMAL_SAFETY_MARGIN = 5.0; // Celsius
  static constexpr int PERFORMANCE_UPDATE_INTERVAL_MS = 1000;
  static constexpr int THERMAL_CHECK_INTERVAL_MS = 500;
};

/// Factory functions for creating autonomous tasks
namespace autonomous {

/// Create compilation task with intelligent configuration
AutonomousTask createCompilationTask(llvm::StringRef model_path,
                                   const PhotonicTargetConfig& base_config,
                                   Priority priority = Priority::NORMAL);

/// Create optimization task
AutonomousTask createOptimizationTask(llvm::StringRef model_path,
                                    const PhotonicTargetConfig& config);

/// Create thermal management task
AutonomousTask createThermalManagementTask(double current_temp,
                                         double target_temp);

/// Create quantum calibration task
AutonomousTask createQuantumCalibrationTask(const PhotonicTargetConfig& config);

} // namespace autonomous

} // namespace photonic
} // namespace mlir

#endif // PHOTONIC_AUTONOMOUS_ORCHESTRATOR_H