# Meson/CMake Gap Analysis: GitHub Issues Design

## Overview

This document defines 10 GitHub issues to bridge the gap between the Apache Arrow C++ Meson and CMake build configurations. Each issue is self-contained: it adds the option, handles dependencies, wires into the build, and updates config.h.

Issues are ordered by priority. The guiding principles are:
- Fix broken/missing defaults first (things cmake enables by default that meson lacks)
- Then add new module support, ordered by likely usage
- GPU and codegen support last
- Prefer meson built-in functionality over reimplementing cmake-style options (e.g., sanitizers, ccache, SIMD are handled natively by meson)

### Explicitly excluded

- **jemalloc**: Defaults OFF in cmake and is low demand. Not worth a dedicated issue.
- **UCX**: Used by Flight for high-performance transport. Very niche, can be added later if requested.
- **Sanitizers, ccache/sccache, coverage, SIMD level control**: Handled natively by meson's built-in options (`-Db_sanitize=`, `-Db_coverage=`, etc.). No custom Arrow options needed.
- **MSVC-specific suffix options, dependency source management, static/shared CRT**: Build tooling details that can be handled incrementally as platform support matures.

## Issue 1: Add RE2 support to the Meson build

**Priority**: Highest -- affects correctness of already-supported features

**Problem**: `ARROW_WITH_RE2` is hardcoded to `false` in `cpp/src/arrow/util/meson.build:70`. Compute kernels that depend on regex (match_like, match_substring_regex, replace_substring_regex, etc. in `cpp/src/arrow/compute/kernels/scalar_string_ascii.cc`) are compiled without RE2 support even when `-Dcompute=enabled`.

**Scope**:
- The `re2.wrap` file already exists at `cpp/subprojects/re2.wrap`
- Auto-detect RE2 via `dependency('re2', required: false)` with subproject fallback
- Replace the hardcoded `false` for `ARROW_WITH_RE2` with actual detection result
- Ensure compute kernels link against RE2 when available
- No new meson option needed -- RE2 should be auto-detected like cmake's default behavior (`ARROW_WITH_RE2` defaults ON in cmake)

**Files to modify**:
- `cpp/meson.build` -- add `needs_re2` variable
- `cpp/src/arrow/util/meson.build` -- wire `ARROW_WITH_RE2` to actual value
- `cpp/src/arrow/meson.build` -- add RE2 to `arrow_compute_deps` (the compute library is built in this file, not in `compute/meson.build`)

**Verification**: Build with `-Dcompute=enabled` and confirm RE2-dependent kernels (e.g., `match_substring_regex`) pass their tests.

## Issue 2: Add mimalloc allocator support to the Meson build

**Priority**: High -- cmake has this ON by default, meson silently runs without it

**Problem**: `ARROW_MIMALLOC` is hardcoded to `false` in `cpp/src/arrow/util/meson.build:51`. CMake enables mimalloc by default (`ARROW_MIMALLOC` default ON in `DefineOptions.cmake:373`), meaning meson builds miss this allocator.

**Scope**:
- Handle `mimalloc` dependency (system or wrap -- check if wrapdb has a mimalloc wrap)
- Wire `ARROW_MIMALLOC` in config.h to detection result
- Build mimalloc-based memory pool (`cpp/src/arrow/memory_pool.cc` has `#ifdef ARROW_MIMALLOC` guards)
- Consider whether to auto-enable (matching cmake default) or add a meson option

**Files to modify**:
- `cpp/meson.build` -- add mimalloc handling
- `cpp/src/arrow/util/meson.build` -- wire config flag
- `cpp/src/arrow/meson.build` -- add mimalloc dependency to arrow core

**Verification**: Build and confirm `arrow::default_memory_pool()` reports mimalloc as the backend.

## Issue 3: Add ORC adapter support to the Meson build

**Priority**: Medium -- prior work exists in PR #46906

**Problem**: `needs_orc` is hardcoded to `false` in `cpp/meson.build:63`. No `orc` option exists in `meson.options`. No `cpp/src/arrow/adapters/orc/meson.build` exists.

**Prior work**: https://github.com/apache/arrow/pull/46906 -- should be picked up and continued.

**Scope**:
- Add `orc` option to `meson.options`
- Handle ORC dependency (system or wrap/subproject)
- Create `cpp/src/arrow/adapters/orc/meson.build`
- Wire `needs_orc` to the option instead of hardcoded `false`
- Wire into dataset module (conditional hooks already exist in `cpp/src/arrow/dataset/meson.build:64,150`)
- Note: cmake force-enables LZ4, Snappy, ZLIB, and ZSTD when ORC is enabled, because the Apache ORC library internally uses these codecs. The meson implementation should ensure these compression libraries are available when ORC is enabled.

**Files to modify**:
- `cpp/meson.options` -- add `orc` option
- `cpp/meson.build` -- wire `needs_orc` to option, add force-enable logic for compression deps (e.g., `needs_lz4 = needs_lz4 or needs_orc` and similarly for snappy, zlib, zstd)
- `cpp/src/arrow/adapters/orc/meson.build` -- new file
- `cpp/src/arrow/meson.build` -- include ORC adapter subdir conditionally

**Verification**: Build with `-Dorc=enabled` and confirm ORC read/write tests pass.

## Issue 4: Add Flight SQL support to the Meson build

**Priority**: Medium

**Problem**: `needs_flight_sql` is hardcoded to `false` in `cpp/meson.build:79`. No `flight_sql` option in `meson.options`. No `cpp/src/arrow/flight/sql/meson.build` exists. Flight RPC already has meson support, so this builds on that.

**Scope**:
- Add `flight_sql` option to `meson.options`
- Create `cpp/src/arrow/flight/sql/meson.build`
- Wire `needs_flight_sql` to the option
- Flight SQL depends on Flight -- the dependency chain (`needs_flight = ... or needs_flight_sql`) is already wired in `cpp/meson.build:80`
- Proto generation needed for `format/FlightSql.proto`. Follow the existing pattern in `cpp/src/arrow/flight/meson.build:52-77` which uses `custom_target` with `protoc`. Note: `cpp/src/arrow/flight/sql/protocol_internal.cc` includes the generated `FlightSql.pb.cc` directly (same ODR pattern as Flight's `Flight.grpc.pb.cc`).

**Files to modify**:
- `cpp/meson.options` -- add `flight_sql` option
- `cpp/meson.build` -- wire `needs_flight_sql` to option
- `cpp/src/arrow/flight/sql/meson.build` -- new file
- `cpp/src/arrow/flight/meson.build` -- include sql subdir conditionally

**Verification**: Build with `-Dflight_sql=enabled -Dflight=enabled` and confirm Flight SQL tests pass.

## Issue 5: Add S3 filesystem support to the Meson build

**Priority**: Medium

**Problem**: Enabling `-Ds3=enabled` hits `error('s3 filesystem support is not yet implemented in Meson')` at `cpp/src/arrow/meson.build:469`. The option exists but the implementation is a hard error.

**Scope**:
- Replace the `error()` call with actual S3 filesystem source compilation
- Handle AWS SDK dependency -- no wrap exists, will need system-installed AWS SDK
- Build S3 filesystem sources when enabled
- Wire into filesystem component

**Note**: cmake also has a separate `ARROW_S3_MODULE` option that builds S3 as a dynamic module rather than linking statically. The initial meson implementation can support direct linking only, with dynamic module support as a possible follow-up.

**Files to modify**:
- `cpp/src/arrow/meson.build` -- replace error with S3 sources and AWS SDK dependency
- Potentially `cpp/src/arrow/filesystem/meson.build` if filesystem sources are managed there

**Verification**: Build with `-Ds3=enabled` and confirm S3 filesystem tests pass (may require localstack or similar for integration tests).

## Issue 6: Add CUDA support to the Meson build

**Priority**: Medium-Low

**Problem**: `needs_cuda` is hardcoded to `false` in `cpp/meson.build:56`. No option, no `cpp/src/arrow/gpu/meson.build`.

**Scope**:
- Add `cuda` option to `meson.options`
- Meson has native CUDA language support via `add_languages('cuda')`
- Create `cpp/src/arrow/gpu/meson.build`
- Wire `needs_cuda` and `ARROW_CUDA` config flag
- CUDA depends on IPC -- `needs_ipc` in `cpp/meson.build` must be updated to include `needs_cuda` in its dependency chain

**Files to modify**:
- `cpp/meson.options` -- add `cuda` option
- `cpp/meson.build` -- wire `needs_cuda`, add `needs_cuda` to `needs_ipc` dependency chain, potentially add CUDA language
- `cpp/src/arrow/gpu/meson.build` -- new file
- `cpp/src/arrow/util/meson.build` -- config flag already wired via `needs_cuda`

**Verification**: Build with `-Dcuda=enabled` and confirm CUDA IPC tests pass on a system with a CUDA toolkit.

## Issue 7: Add OpenTelemetry support to the Meson build

**Priority**: Low

**Problem**: `needs_opentelemetry` is hardcoded to `false` in `cpp/meson.build:62`. A subproject directory exists at `cpp/subprojects/opentelemetry-cpp/` but isn't integrated.

**Scope**:
- Add `opentelemetry` option to `meson.options`
- Handle OpenTelemetry dependency. Note: `cpp/subprojects/opentelemetry-cpp/` is a full git clone directory (not a `.wrap` file), which is unusual -- every other subproject uses `.wrap` files. The implementer will need to either write an `opentelemetry-cpp.wrap` file or use `subproject('opentelemetry-cpp')` directly.
- Enabling OpenTelemetry also requires nlohmann_json and protobuf as additional dependencies
- Build `src/arrow/telemetry/` when enabled
- Wire `ARROW_WITH_OPENTELEMETRY` in config.h

**Files to modify**:
- `cpp/meson.options` -- add `opentelemetry` option
- `cpp/meson.build` -- wire `needs_opentelemetry`
- `cpp/src/arrow/meson.build` -- add telemetry subdir conditionally
- `cpp/src/arrow/util/meson.build` -- config flag already wired via `needs_opentelemetry`

**Verification**: Build with `-Dopentelemetry=enabled` and confirm telemetry tracing is functional.

## Issue 8: Add glog support to the Meson build

**Priority**: Low -- optional logging backend, developer convenience

**Problem**: `ARROW_USE_GLOG` is hardcoded to `false` in `cpp/src/arrow/util/meson.build:60`.

**Scope**:
- Auto-detect glog via `dependency('libglog', required: false)`
- Wire `ARROW_USE_GLOG` to detection result
- No meson option needed -- auto-detect like cmake's behavior

**Files to modify**:
- `cpp/src/arrow/util/meson.build` -- wire config flag
- `cpp/src/arrow/meson.build` -- add glog dependency when found

**Verification**: Build on a system with glog installed and confirm `ARROW_USE_GLOG` is set in config.h.

## Issue 9: Add musl libc detection to the Meson build

**Priority**: Low -- just a flag for Alpine/musl-based systems

**Problem**: `ARROW_WITH_MUSL` is hardcoded to `false` in `cpp/src/arrow/util/meson.build:68`.

**Scope**:
- Note: cmake's `ARROW_WITH_MUSL` is a manual option (default OFF), not auto-detected. The meson implementation can improve on this by auto-detecting musl. However, musl deliberately does not define `__MUSL__` or any similar macro, so detection requires alternative approaches (e.g., checking the output of `ldd --version`, examining the C library, or using a compile test for musl-specific behavior).
- Alternatively, a simple meson option matching cmake's manual approach may be more reliable.
- Wire `ARROW_WITH_MUSL` to the result.
- Small, self-contained change.

**Files to modify**:
- `cpp/meson.options` -- add `musl` option (if taking the manual approach)
- `cpp/src/arrow/util/meson.build` -- detect/wire config flag

**Verification**: Build on an Alpine/musl system and confirm `ARROW_WITH_MUSL` is set in config.h.

## Issue 10: Add Gandiva support to the Meson build

**Priority**: Lowest

**Problem**: `needs_gandiva` is hardcoded to `false` in `cpp/meson.build:81`. No option, no meson.build for Gandiva.

**Scope**:
- Add `gandiva` option to `meson.options`
- Handle LLVM dependency -- heavy, no wrap exists, system-installed only
- Depends on RE2 (Issue 1) being completed first
- Create meson.build for `cpp/src/gandiva/`
- Wire `needs_gandiva` and config flags

**Files to modify**:
- `cpp/meson.options` -- add `gandiva` option
- `cpp/meson.build` -- wire `needs_gandiva`
- `cpp/src/gandiva/meson.build` -- new file
- Depends on: Issue 1 (RE2)

**Verification**: Build with `-Dgandiva=enabled` and confirm Gandiva expression evaluation tests pass.

## Summary

| # | Issue | Priority | Status | Dependencies |
|---|-------|----------|--------|--------------|
| 1 | RE2 support | Highest | Hardcoded false | wrap exists |
| 2 | Mimalloc support | High | Hardcoded false | cmake default ON |
| 3 | ORC adapter | Medium | Hardcoded false | Prior PR #46906 |
| 4 | Flight SQL | Medium | Hardcoded false | Flight (done) |
| 5 | S3 filesystem | Medium | error() guard | Needs AWS SDK |
| 6 | CUDA support | Medium-Low | Hardcoded false | Meson has native CUDA |
| 7 | OpenTelemetry | Low | Hardcoded false | Subproject exists |
| 8 | glog support | Low | Hardcoded false | Auto-detect |
| 9 | musl detection | Low | Hardcoded false | Auto-detect |
| 10 | Gandiva | Lowest | Hardcoded false | Needs LLVM + RE2 |
