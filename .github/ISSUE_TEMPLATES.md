# GitHub Issue Templates

This document provides templates for GitHub issues. These should be created as `.github/ISSUE_TEMPLATE/` files.

## Bug Report Template (`bug_report.yml`)

```yaml
name: Bug Report
description: File a bug report to help us improve
title: "[Bug]: "
labels: ["bug", "triage"]
assignees: []

body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
        
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: A clear and concise description of what the bug is.
      placeholder: Tell us what you see!
    validations:
      required: true

  - type: textarea
    id: reproduction
    attributes:
      label: Reproduction Steps
      description: Steps to reproduce the behavior
      placeholder: |
        1. Compile model with '...'
        2. Run inference with '...'
        3. See error
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: Expected Behavior
      description: A clear and concise description of what you expected to happen.
    validations:
      required: true

  - type: textarea
    id: code-sample
    attributes:
      label: Code Sample
      description: Minimal code sample that reproduces the issue
      render: cpp
    validations:
      required: false

  - type: dropdown
    id: component
    attributes:
      label: Component
      description: Which component is affected?
      options:
        - Compiler
        - Runtime
        - Python Bindings
        - MLIR Dialect
        - Hardware Interface
        - Documentation
        - Build System
        - Other
    validations:
      required: true

  - type: input
    id: version
    attributes:
      label: Version
      description: What version of photon-mlir are you running?
      placeholder: "v0.1.0"
    validations:
      required: true

  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: |
        Please provide details about your environment:
      value: |
        - OS: [e.g. Ubuntu 22.04]
        - Compiler: [e.g. GCC 11.4]
        - LLVM Version: [e.g. 17.0.6]
        - Python Version: [e.g. 3.11.5]
        - Hardware: [e.g. Lightmatter Envise]
      render: markdown
    validations:
      required: true

  - type: textarea
    id: logs
    attributes:
      label: Relevant Log Output
      description: Please copy and paste any relevant log output
      render: shell
    validations:
      required: false

  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      description: By submitting this issue, you agree to follow our Code of Conduct
      options:
        - label: I agree to follow this project's Code of Conduct
          required: true
```

## Feature Request Template (`feature_request.yml`)

```yaml
name: Feature Request
description: Suggest an idea for photon-mlir
title: "[Feature]: "
labels: ["enhancement", "triage"]
assignees: []

body:
  - type: markdown
    attributes:
      value: |
        Thanks for your interest in improving photon-mlir!
        
  - type: textarea
    id: problem
    attributes:
      label: Problem Statement
      description: Is your feature request related to a problem? Please describe.
      placeholder: A clear and concise description of what the problem is...
    validations:
      required: true

  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
      description: Describe the solution you'd like
      placeholder: A clear and concise description of what you want to happen...
    validations:
      required: true

  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
      description: Describe alternatives you've considered
      placeholder: A clear and concise description of any alternative solutions...
    validations:
      required: false

  - type: dropdown
    id: category
    attributes:
      label: Feature Category
      description: What category does this feature belong to?
      options:
        - Compiler Optimization
        - New Hardware Support
        - API Enhancement
        - Performance Improvement
        - Developer Experience
        - Documentation
        - Testing
        - Other
    validations:
      required: true

  - type: dropdown
    id: priority
    attributes:
      label: Priority
      description: How important is this feature to you?
      options:
        - Low - Nice to have
        - Medium - Would be helpful
        - High - Important for my use case
        - Critical - Blocking my work
    validations:
      required: true

  - type: textarea
    id: use-case
    attributes:
      label: Use Case
      description: Describe your specific use case for this feature
      placeholder: What are you trying to achieve?
    validations:
      required: true

  - type: textarea
    id: additional-context
    attributes:
      label: Additional Context
      description: Add any other context, screenshots, or examples about the feature request
    validations:
      required: false

  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      description: By submitting this issue, you agree to follow our Code of Conduct
      options:
        - label: I agree to follow this project's Code of Conduct
          required: true
```

## Documentation Issue Template (`documentation.yml`)

```yaml
name: Documentation Issue
description: Report an issue with documentation
title: "[Docs]: "
labels: ["documentation", "triage"]
assignees: []

body:
  - type: dropdown
    id: doc-type
    attributes:
      label: Documentation Type
      description: What type of documentation is this about?
      options:
        - API Reference
        - User Guide
        - Tutorial
        - Examples
        - README
        - Code Comments
        - Architecture Documentation
    validations:
      required: true

  - type: input
    id: location
    attributes:
      label: Documentation Location
      description: Where is the documentation issue located?
      placeholder: "e.g., docs/guides/getting-started.md, line 42"
    validations:
      required: true

  - type: dropdown
    id: issue-type
    attributes:
      label: Issue Type
      description: What kind of documentation issue is this?
      options:
        - Missing Information
        - Incorrect Information
        - Unclear/Confusing
        - Outdated
        - Typo/Grammar
        - Broken Link
        - Missing Example
        - Other
    validations:
      required: true

  - type: textarea
    id: description
    attributes:
      label: Description
      description: Describe the documentation issue
      placeholder: What is wrong with the current documentation?
    validations:
      required: true

  - type: textarea
    id: suggestion
    attributes:
      label: Suggested Fix
      description: If you have a suggestion for how to fix this, please describe it
    validations:
      required: false

  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      description: By submitting this issue, you agree to follow our Code of Conduct
      options:
        - label: I agree to follow this project's Code of Conduct
          required: true
```

## Hardware Support Request Template (`hardware_support.yml`)

```yaml
name: Hardware Support Request
description: Request support for a new photonic hardware platform
title: "[Hardware]: "
labels: ["hardware", "enhancement", "triage"]
assignees: []

body:
  - type: input
    id: hardware-name
    attributes:
      label: Hardware Platform Name
      description: What is the name of the hardware platform?
      placeholder: "e.g., Lightmatter Envise Pro"
    validations:
      required: true

  - type: input
    id: vendor
    attributes:
      label: Vendor
      description: Who manufactures this hardware?
      placeholder: "e.g., Lightmatter, Intel, Custom"
    validations:
      required: true

  - type: textarea
    id: specifications
    attributes:
      label: Hardware Specifications
      description: Please provide key specifications
      value: |
        - Array Size: [e.g., 128x128]
        - Clock Rate: [e.g., 1 GHz]
        - Wavelength: [e.g., 1550 nm]
        - Precision: [e.g., INT8, FP16]
        - Power Consumption: [e.g., 50W]
        - Interface: [e.g., PCIe 4.0, Ethernet]
      render: markdown
    validations:
      required: true

  - type: dropdown
    id: availability
    attributes:
      label: Hardware Availability
      description: What is the availability status of this hardware?
      options:
        - Commercially Available
        - Pre-production/Beta
        - Research Prototype
        - Announced/Future
    validations:
      required: true

  - type: textarea
    id: documentation
    attributes:
      label: Available Documentation
      description: What documentation is available for this hardware?
      placeholder: |
        - SDK documentation links
        - Programming guides
        - Hardware specifications
        - Driver information
    validations:
      required: false

  - type: dropdown
    id: access
    attributes:
      label: Hardware Access
      description: Do you have access to this hardware for testing?
      options:
        - "Yes - I own this hardware"
        - "Yes - I have lab access"
        - "Limited - Can arrange testing"
        - "No - Requesting based on documentation"
    validations:
      required: true

  - type: textarea
    id: use-case
    attributes:
      label: Use Case
      description: What would you use this hardware for?
      placeholder: Describe your specific application or research needs
    validations:
      required: true

  - type: textarea
    id: timeline
    attributes:
      label: Timeline
      description: When do you need this support?
      placeholder: Any specific deadlines or project timelines?
    validations:
      required: false

  - type: checkboxes
    id: contribution
    attributes:
      label: Contribution
      description: Are you willing to help with the implementation?
      options:
        - label: I can provide hardware specifications
        - label: I can help with testing
        - label: I can contribute code
        - label: I can provide documentation

  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      description: By submitting this issue, you agree to follow our Code of Conduct
      options:
        - label: I agree to follow this project's Code of Conduct
          required: true
```

## Performance Issue Template (`performance.yml`)

```yaml
name: Performance Issue
description: Report a performance problem or regression
title: "[Performance]: "
labels: ["performance", "triage"]
assignees: []

body:
  - type: textarea
    id: description
    attributes:
      label: Performance Issue Description
      description: Describe the performance problem
      placeholder: What performance issue are you experiencing?
    validations:
      required: true

  - type: dropdown
    id: component
    attributes:
      label: Affected Component
      description: Which component has the performance issue?
      options:
        - Compilation Time
        - Runtime Performance
        - Memory Usage
        - Thermal Performance
        - Power Consumption
        - Throughput
        - Latency
        - Other
    validations:
      required: true

  - type: textarea
    id: benchmark
    attributes:
      label: Benchmark Results
      description: Please provide benchmark results or performance measurements
      render: shell
    validations:
      required: false

  - type: textarea
    id: model-info
    attributes:
      label: Model Information
      description: Information about the model/workload
      value: |
        - Model Type: [e.g., ResNet-50, BERT-Base]
        - Model Size: [e.g., 25M parameters]
        - Input Shape: [e.g., 1x3x224x224]
        - Batch Size: [e.g., 32]
      render: markdown
    validations:
      required: true

  - type: textarea
    id: hardware-config
    attributes:
      label: Hardware Configuration
      description: Hardware and system configuration
      value: |
        - Hardware: [e.g., Lightmatter Envise]
        - CPU: [e.g., Intel Xeon Gold 6248]
        - Memory: [e.g., 128GB DDR4]
        - GPU: [e.g., NVIDIA V100]
        - OS: [e.g., Ubuntu 22.04]
      render: markdown
    validations:
      required: true

  - type: textarea
    id: expected-performance
    attributes:
      label: Expected Performance
      description: What performance did you expect?
      placeholder: Based on documentation, similar hardware, or previous versions
    validations:
      required: true

  - type: textarea
    id: regression
    attributes:
      label: Performance Regression
      description: Is this a regression from a previous version?
      placeholder: If yes, what version worked better?
    validations:
      required: false

  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      description: By submitting this issue, you agree to follow our Code of Conduct
      options:
        - label: I agree to follow this project's Code of Conduct
          required: true
```

## Template Usage Instructions

1. Create `.github/ISSUE_TEMPLATE/` directory
2. Save each template as a `.yml` file with the corresponding name
3. GitHub will automatically present these templates when users create issues
4. Customize labels, assignees, and validation rules as needed
5. Regular review and updates based on project needs

## Additional Configuration

### Issue Template Chooser Config (`.github/ISSUE_TEMPLATE/config.yml`)

```yaml
blank_issues_enabled: false
contact_links:
  - name: Community Discussion
    url: https://github.com/yourusername/photon-mlir-bridge/discussions
    about: Ask questions and discuss ideas with the community
  - name: Security Vulnerability
    url: https://github.com/yourusername/photon-mlir-bridge/security/advisories/new
    about: Report security vulnerabilities privately
```

This configuration disables blank issues and provides alternative contact methods for discussions and security reports.