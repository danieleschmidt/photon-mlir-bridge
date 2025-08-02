// Unit tests for PhotonicDevice class
// Tests the core photonic device abstraction

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <cmath>

// Mock PhotonicDevice for testing (actual headers would be included)
namespace photon {

struct DeviceConfig {
    int wavelength;
    std::pair<size_t, size_t> array_size;
    double max_power;
    double thermal_range_min;
    double thermal_range_max;
};

class PhotonicDevice {
public:
    explicit PhotonicDevice(const DeviceConfig& config) 
        : config_(config), phase_shifts_(config.array_size.first * config.array_size.second, 0.0) {}
    
    void setPhaseShift(size_t index, double phase) {
        if (index >= phase_shifts_.size()) {
            throw std::out_of_range("Phase shift index out of range");
        }
        if (std::abs(phase) > M_PI) {
            throw std::invalid_argument("Phase shift must be within [-π, π]");
        }
        phase_shifts_[index] = phase;
    }
    
    double getPhaseShift(size_t index) const {
        if (index >= phase_shifts_.size()) {
            throw std::out_of_range("Phase shift index out of range");
        }
        return phase_shifts_[index];
    }
    
    void calibrateThermal(double temperature) {
        if (temperature < config_.thermal_range_min || temperature > config_.thermal_range_max) {
            throw std::out_of_range("Temperature out of operating range");
        }
        // Mock thermal calibration
        thermal_offset_ = (temperature - 25.0) * 0.01; // 1% per degree
    }
    
    double getThermalOffset() const { return thermal_offset_; }
    
    size_t getArraySize() const { return phase_shifts_.size(); }

private:
    DeviceConfig config_;
    std::vector<double> phase_shifts_;
    double thermal_offset_ = 0.0;
};

} // namespace photon

namespace photon {
namespace test {

class PhotonicDeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.wavelength = 1550;
        config_.array_size = {64, 64};
        config_.max_power = 100.0;
        config_.thermal_range_min = 20.0;
        config_.thermal_range_max = 80.0;
        device_ = std::make_unique<PhotonicDevice>(config_);
    }

    DeviceConfig config_;
    std::unique_ptr<PhotonicDevice> device_;
};

TEST_F(PhotonicDeviceTest, DeviceInitialization) {
    EXPECT_EQ(device_->getArraySize(), 64 * 64);
    EXPECT_DOUBLE_EQ(device_->getThermalOffset(), 0.0);
}

TEST_F(PhotonicDeviceTest, SetPhaseShiftUpdatesCorrectly) {
    const double phase = M_PI_2;
    const size_t index = 5;
    
    device_->setPhaseShift(index, phase);
    
    EXPECT_DOUBLE_EQ(device_->getPhaseShift(index), phase);
}

TEST_F(PhotonicDeviceTest, SetMultiplePhaseShifts) {
    const std::vector<double> phases = {0.0, M_PI_4, M_PI_2, -M_PI_4, -M_PI_2};
    
    for (size_t i = 0; i < phases.size(); ++i) {
        device_->setPhaseShift(i, phases[i]);
    }
    
    for (size_t i = 0; i < phases.size(); ++i) {
        EXPECT_DOUBLE_EQ(device_->getPhaseShift(i), phases[i]);
    }
}

TEST_F(PhotonicDeviceTest, InvalidIndexThrowsException) {
    const size_t invalid_index = 10000;
    
    EXPECT_THROW(
        device_->setPhaseShift(invalid_index, 0.0),
        std::out_of_range
    );
    
    EXPECT_THROW(
        device_->getPhaseShift(invalid_index),
        std::out_of_range
    );
}

TEST_F(PhotonicDeviceTest, PhaseShiftRangeValidation) {
    const size_t index = 0;
    
    // Valid phase shifts should not throw
    EXPECT_NO_THROW(device_->setPhaseShift(index, M_PI));
    EXPECT_NO_THROW(device_->setPhaseShift(index, -M_PI));
    EXPECT_NO_THROW(device_->setPhaseShift(index, 0.0));
    
    // Invalid phase shifts should throw
    EXPECT_THROW(
        device_->setPhaseShift(index, M_PI + 0.1),
        std::invalid_argument
    );
    EXPECT_THROW(
        device_->setPhaseShift(index, -M_PI - 0.1),
        std::invalid_argument
    );
}

TEST_F(PhotonicDeviceTest, ThermalCalibration) {
    const double room_temperature = 25.0;
    const double hot_temperature = 60.0;
    
    device_->calibrateThermal(room_temperature);
    EXPECT_DOUBLE_EQ(device_->getThermalOffset(), 0.0);
    
    device_->calibrateThermal(hot_temperature);
    EXPECT_DOUBLE_EQ(device_->getThermalOffset(), 0.35); // 35 degrees * 0.01
}

TEST_F(PhotonicDeviceTest, ThermalRangeValidation) {
    EXPECT_THROW(
        device_->calibrateThermal(10.0), // Too cold
        std::out_of_range
    );
    
    EXPECT_THROW(
        device_->calibrateThermal(90.0), // Too hot
        std::out_of_range
    );
    
    // Valid temperatures should not throw
    EXPECT_NO_THROW(device_->calibrateThermal(20.0));
    EXPECT_NO_THROW(device_->calibrateThermal(80.0));
}

// Performance tests
TEST_F(PhotonicDeviceTest, PhaseShiftPerformance) {
    const size_t num_operations = 10000;
    const auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < num_operations; ++i) {
        device_->setPhaseShift(i % device_->getArraySize(), (i % 100) * 0.01);
    }
    
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Should complete within reasonable time (arbitrary threshold)
    EXPECT_LT(duration.count(), 100000); // Less than 100ms
}

// Edge case tests
TEST_F(PhotonicDeviceTest, BoundaryConditions) {
    const size_t last_index = device_->getArraySize() - 1;
    
    // Test first and last valid indices
    EXPECT_NO_THROW(device_->setPhaseShift(0, M_PI_2));
    EXPECT_NO_THROW(device_->setPhaseShift(last_index, -M_PI_2));
    
    EXPECT_DOUBLE_EQ(device_->getPhaseShift(0), M_PI_2);
    EXPECT_DOUBLE_EQ(device_->getPhaseShift(last_index), -M_PI_2);
}

} // namespace test
} // namespace photon