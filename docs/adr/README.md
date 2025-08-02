# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the photon-mlir-bridge project.

## What are ADRs?

Architecture Decision Records are short text documents that capture important architectural decisions made during the project lifecycle. Each ADR describes the context, decision, and consequences of a significant architectural choice.

## ADR Format

Each ADR follows this template structure:

```markdown
# ADR-XXXX: [Title]

**Status**: [Proposed | Accepted | Deprecated | Superseded]  
**Date**: YYYY-MM-DD  
**Deciders**: [List of people involved in the decision]

## Context

[Describe the architectural decision context and the forces at play]

## Decision

[Describe the architectural decision and the reasoning behind it]

## Consequences

[Describe the consequences of the decision, both positive and negative]

## Alternatives Considered

[List alternative approaches that were considered]

## References

[Links to supporting documentation, discussions, or related decisions]
```

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](./0001-mlir-framework-choice.md) | MLIR Framework Selection | Accepted | 2024-01-15 |
| [0002](./0002-photonic-dialect-design.md) | Photonic Dialect Architecture | Accepted | 2024-01-20 |
| [0003](./0003-thermal-compensation-strategy.md) | Thermal Compensation Approach | Accepted | 2024-02-01 |
| [0004](./0004-multi-device-partitioning.md) | Multi-Device Partitioning Algorithm | Accepted | 2024-02-15 |

## Creating New ADRs

1. Use the next sequential number (e.g., 0005)
2. Create a descriptive filename: `XXXX-short-title.md`
3. Follow the template structure
4. Update this README with the new ADR entry
5. Submit for review through standard PR process

## ADR Lifecycle

- **Proposed**: ADR is under discussion
- **Accepted**: ADR has been approved and is being implemented
- **Deprecated**: ADR is no longer recommended but may still be in use
- **Superseded**: ADR has been replaced by a newer decision

## Guidelines

- Focus on architectural decisions, not implementation details
- Keep ADRs concise and readable
- Update status as decisions evolve
- Link related ADRs for traceability
- Include measurable success criteria when applicable