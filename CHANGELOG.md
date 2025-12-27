# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-27

### Added
- **World Model Integration**: Implemented latent dynamics model and encoder for representation learning.
- **PPO Trainer**: Proximal Policy Optimization implementation with support for feature-mode and reward-mode from world model.
- **Project Structure**: Adopted modern `src-layout` with `pyproject.toml` configuration.
- **Dependency Management**: Migrated to `uv` for fast and reliable package management.
- **Testing**: Added basic smoke tests for World Model components using `pytest`.
- **Documentation**: Comprehensive documentation structure in `docs/` and root `README.md`.

### Changed
- Refactored code into `mujoco_lightworld` package.
- Updated `train.py` to use relative imports and support entry-point execution.

### Removed
- Legacy `requirements.txt` (superseded by `pyproject.toml`).
