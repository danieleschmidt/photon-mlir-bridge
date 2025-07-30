# Pull Request Template

## 📝 Description

<!-- Provide a brief description of the changes in this PR -->

### Type of Change

<!-- Mark the relevant option with an "x" -->

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation (changes to documentation only)
- [ ] 🔧 Refactoring (code changes that neither fix a bug nor add a feature)
- [ ] ⚡ Performance improvement
- [ ] 🧪 Test addition or modification
- [ ] 🏗️ Build system or CI/CD changes
- [ ] 🔒 Security improvement

## 🔗 Related Issues

<!-- Link to the issue this PR addresses -->
Closes #(issue_number)

## 🧪 Testing

<!-- Describe the tests you ran to verify your changes -->

### Test Environment
- [ ] Local development environment
- [ ] CI/CD pipeline
- [ ] Hardware testing (specify hardware)
- [ ] Performance benchmarks

### Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] End-to-end tests added/updated
- [ ] Performance tests added/updated

### Test Results
<!-- Paste relevant test output or link to CI results -->

```
Test results here or link to CI build
```

## 📊 Performance Impact

<!-- If applicable, describe performance implications -->

### Benchmark Results
<!-- Include before/after performance measurements -->

| Metric | Before | After | Change |
|--------|--------|-------|---------|
| Compilation Time | | | |
| Runtime Performance | | | |
| Memory Usage | | | |
| Energy Efficiency | | | |

### Hardware Compatibility
- [ ] Tested on Lightmatter Envise
- [ ] Tested on MIT Photonic Processor
- [ ] Tested with simulation
- [ ] Not applicable

## 🔧 Implementation Details

<!-- Describe the technical implementation approach -->

### Key Changes
- 
- 
- 

### Architecture Impact
<!-- Does this change affect the overall architecture? -->

### Dependencies
<!-- List any new dependencies or version changes -->
- 
- 

## 📋 Checklist

### Code Quality
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

### MLIR/Compiler Specific
- [ ] MLIR dialect changes are properly documented
- [ ] TableGen definitions are validated
- [ ] Pass registration is correct
- [ ] Optimization passes are tested
- [ ] Error handling is comprehensive

### Python Bindings
- [ ] Python bindings are updated (if applicable)
- [ ] Type hints are provided
- [ ] Docstrings follow project conventions
- [ ] Examples are updated

### Documentation
- [ ] README updated (if applicable)
- [ ] API documentation updated
- [ ] User guides updated (if applicable)
- [ ] Examples updated (if applicable)
- [ ] CHANGELOG.md updated

### Security
- [ ] No sensitive information is exposed
- [ ] Input validation is implemented
- [ ] Security best practices followed
- [ ] Dependencies are secure and up-to-date

## 🖼️ Screenshots/Examples

<!-- If applicable, add screenshots or code examples -->

### Before
```cpp
// Old code example
```

### After
```cpp
// New code example
```

## 🚀 Deployment Notes

<!-- Any special deployment considerations -->

### Migration Required
- [ ] Database migration needed
- [ ] Configuration changes required
- [ ] Hardware recalibration needed
- [ ] No migration required

### Rollback Plan
<!-- Describe how to rollback if needed -->

## 📝 Additional Notes

<!-- Any additional information that reviewers should know -->

### Future Work
<!-- Are there follow-up tasks or improvements planned? -->

### Known Limitations
<!-- Any known limitations or trade-offs -->

## 👥 Reviewers

<!-- Request specific reviewers if needed -->
/cc @reviewer1 @reviewer2

---

**Reviewer Guidelines:**
- Focus on correctness, performance, and maintainability
- Check for proper error handling and edge cases
- Verify documentation and examples are accurate
- Consider impact on existing users and backwards compatibility
- Test locally if possible, especially for hardware-related changes